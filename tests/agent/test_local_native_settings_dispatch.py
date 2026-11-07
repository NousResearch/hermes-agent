"""Managed llama.cpp dispatch: saved per-artifact settings feed live requests.

The integration helper (``agent.chat_completion_local.merge_native_local_settings``)
resolves the staged artifact's saved native settings
(``native_model_settings.resolve_native_request_settings``) into the outgoing
request overrides: standard OpenAI fields merge WITHOUT overriding explicit
per-request values, llama.cpp engine extensions merge into ``extra_body`` per
key. Settings-store/artifact trouble must never break inference; only an
explicit per-request payload the native schema rejects surfaces as a
validation error. Synthetic GGUFs + temp HERMES_HOME only.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.chat_completion_local import merge_native_local_settings
from hermes_cli.local_runtime import bootstrap, native_model_settings as nms
from hermes_cli.local_runtime.model_settings import (
    AUTO,
    InvalidSettingValueError,
    ModelSettingsService,
    ModelSettingsStore,
)
from tests.hermes_cli._native_gguf_factory import write_gguf as _write_gguf


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    nms.clear_fingerprint_cache()
    yield root
    nms.clear_fingerprint_cache()


def _stage_dispatch_model(root) -> str:
    """Publisher-baked temperature/top_k plus saved top_p=0.9 / seed=AUTO."""
    _write_gguf(bootstrap.models_dir() / "Dispatch-Model.gguf", metadata={
        "general.sampling.temperature": 0.75,
        "general.sampling.top_k": 40,
    })
    key = nms.settings_key_for_model("Dispatch-Model")
    ModelSettingsService(ModelSettingsStore(root)).save(
        key, nms.NATIVE_CHAT_SCHEMA, {"top_p": 0.9, "seed": AUTO}, 0)
    return "Dispatch-Model"


def _agent(**over):
    fields = {"provider": "llamacpp", "base_url": "http://127.0.0.1:8080/v1",
              "model": "Dispatch-Model", "api_mode": "chat_completions"}
    fields.update(over)
    return SimpleNamespace(**fields)


# ── saved settings contribute ─────────────────────────────────────────────


def test_only_user_saved_settings_contribute_to_the_request(home):
    _stage_dispatch_model(home)
    out = merge_native_local_settings(_agent(), {})
    # saved top_p rides the standard surface; the saved seed AUTO is omitted.
    # Publisher GGUF metadata (temperature 0.75, top_k 40) is NOT injected:
    # llama-server already applies those as launch defaults, and a per-request
    # copy would override a hand-edited presets.ini (inheritance must survive).
    assert out == {"top_p": 0.9}


def test_saved_llamacpp_extension_merges_into_extra_body(home):
    _write_gguf(bootstrap.models_dir() / "Ext-Model.gguf")
    key = nms.settings_key_for_model("Ext-Model")
    ModelSettingsService(ModelSettingsStore(home)).save(
        key, nms.NATIVE_CHAT_SCHEMA, {"top_k": 0, "min_p": 0.05}, 0)
    out = merge_native_local_settings(_agent(model="Ext-Model"), {})
    assert out == {"extra_body": {"top_k": 0, "min_p": 0.05}}   # explicit zero kept


def test_unconfigured_user_request_is_unchanged_and_never_hashes(home, monkeypatch):
    """No settings store yet => nothing the user saved can exist: pass the
    request through untouched WITHOUT hashing a multi-GB GGUF on the hot path."""
    _write_gguf(bootstrap.models_dir() / "Fresh-Model.gguf", metadata={
        "general.sampling.temperature": 0.75, "general.sampling.top_k": 40})

    hashed: list[str] = []
    real = nms.artifact_fingerprint

    def _recording(model_id):
        hashed.append(model_id)       # a raise would be swallowed by the hook; record instead
        return real(model_id)

    monkeypatch.setattr(nms, "artifact_fingerprint", _recording)
    explicit = {"max_tokens": 64}
    assert merge_native_local_settings(_agent(model="Fresh-Model"), explicit) == explicit
    assert merge_native_local_settings(_agent(model="Fresh-Model"), {}) == {}
    assert hashed == []               # no store => no hashing on the hot path


def test_model_without_saved_row_inherits_server_defaults(home):
    _stage_dispatch_model(home)                       # store exists (other model saved)
    _write_gguf(bootstrap.models_dir() / "Other-Model.gguf", metadata={
        "general.sampling.temperature": 0.6})
    assert merge_native_local_settings(_agent(model="Other-Model"), {}) == {}


def test_explicit_request_values_win_and_extensions_survive(home):
    _stage_dispatch_model(home)
    explicit = {"temperature": 0.3, "extra_body": {"top_k": 8, "cache_prompt": True}}
    out = merge_native_local_settings(_agent(), explicit)
    assert out["temperature"] == 0.3           # explicit wins over saved/publisher
    assert out["top_p"] == 0.9                 # unmentioned settings still contribute
    # per-key merge: the explicit top_k wins and the unrelated extension lives.
    assert out["extra_body"] == {"top_k": 8, "cache_prompt": True}
    # the caller's dict is never mutated
    assert explicit == {"temperature": 0.3, "extra_body": {"top_k": 8, "cache_prompt": True}}


# ── explicit payload validation surfaces ─────────────────────────────────


def test_unsupported_explicit_payload_raises(home):
    _stage_dispatch_model(home)
    with pytest.raises(InvalidSettingValueError):
        merge_native_local_settings(_agent(), {"temperature": {"value": 1}})
    with pytest.raises(InvalidSettingValueError):
        merge_native_local_settings(_agent(), {"seed": 1.5})
    with pytest.raises(InvalidSettingValueError):
        merge_native_local_settings(_agent(), {"extra_body": {"top_k": -1}})


# ── never fails inference on our own problems ─────────────────────────────


def test_unknown_model_contributes_nothing(home):
    _stage_dispatch_model(home)
    assert merge_native_local_settings(_agent(model="Not-Staged"), {}) == {}
    # passthrough only — the request is returned exactly as it came
    explicit = {"temperature": 0.3, "max_tokens": 64}
    assert merge_native_local_settings(_agent(model="Not-Staged"), explicit) == explicit


def test_corrupt_store_never_breaks_request(home):
    _stage_dispatch_model(home)
    store = home / "local_runtime" / "model_settings.sqlite3"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_bytes(b"this is not sqlite")
    explicit = {"max_tokens": 128}
    assert merge_native_local_settings(_agent(), explicit) == explicit


def _write_raw_row(root, overrides):
    """A saved row the way an older build, tightened bounds or a hand edit leaves it: the store validates TYPES only, never keys or bounds."""
    key = nms.settings_key_for_model("Dispatch-Model")
    ModelSettingsStore(root).write(key, overrides, 0)


@pytest.mark.parametrize("bad_row", [
    {"temperature": 99.0},                      # above the schema's maximum
    {"temperature": 0.5, "no_such_setting": 1},  # a key the schema does not know
    {"top_p": -3.0},
])
def test_a_saved_row_that_no_longer_validates_never_breaks_inference(home, bad_row):
    """The row is OUR data, not the caller's payload: refusing the caller for it would break every chat request for that model."""
    _stage_dispatch_model(home)
    ModelSettingsStore(home).path.unlink()
    _write_raw_row(home, bad_row)
    explicit = {"max_tokens": 128}
    assert merge_native_local_settings(_agent(), explicit) == explicit
    assert merge_native_local_settings(_agent(), {}) == {}


def test_a_bad_saved_row_is_logged_not_swallowed_silently(home, caplog):
    import logging
    _stage_dispatch_model(home)
    ModelSettingsStore(home).path.unlink()
    _write_raw_row(home, {"temperature": 99.0})
    with caplog.at_level(logging.WARNING, logger="agent.chat_completion_local"):
        merge_native_local_settings(_agent(), {})
    assert any("temperature" in r.getMessage() and "ignored" in r.getMessage().lower() for r in caplog.records), \
        [r.getMessage() for r in caplog.records]


def test_an_invalid_explicit_payload_is_still_refused_when_the_saved_row_is_fine(home):
    """Control: provenance decides. The same exception type, raised for the caller's own payload, must still surface."""
    _stage_dispatch_model(home)
    with pytest.raises(InvalidSettingValueError):
        merge_native_local_settings(_agent(), {"temperature": 99.0})


def test_an_invalid_explicit_payload_passes_through_when_the_saved_row_is_the_broken_one(home):
    """Both are bad: the saved row is ours to absorb, and the caller's value goes on unchanged to the transport's own validation."""
    _stage_dispatch_model(home)
    ModelSettingsStore(home).path.unlink()
    _write_raw_row(home, {"temperature": 99.0})
    assert merge_native_local_settings(_agent(), {"temperature": 3.0}) == {"temperature": 3.0}


def test_a_valid_saved_row_still_applies_after_the_fix(home):
    """Control: the provenance probe must not turn off saved settings that are fine."""
    _stage_dispatch_model(home)
    assert merge_native_local_settings(_agent(), {}) == {"top_p": 0.9}


# ── scope gates ───────────────────────────────────────────────────────────


def test_profile_scope_respects_the_active_home(home, monkeypatch):
    root = home
    (root / ".env").write_text("", encoding="utf-8")  # home marker for profile resolution
    alt = root / "profiles" / "alt"
    alt.mkdir(parents=True)
    (alt / ".env").write_text("", encoding="utf-8")   # profile identity marker
    _stage_dispatch_model(root)                       # saved under the default profile

    monkeypatch.setenv("HERMES_HOME", str(alt))
    out = merge_native_local_settings(_agent(), {})
    # alt has no store of its own: nothing is injected, and the default
    # profile's saved top_p must not leak across profiles.
    assert out == {}


def test_non_managed_provider_is_untouched(home):
    _stage_dispatch_model(home)
    explicit = {"temperature": 0.3}
    remote = _agent(provider="custom", base_url="https://remote.example.com/v1")
    assert merge_native_local_settings(remote, explicit) == explicit


def test_non_chat_modes_are_untouched(home):
    _stage_dispatch_model(home)
    assert merge_native_local_settings(_agent(api_mode="anthropic_messages"), {}) == {}


# ── wiring: the chat_completions builder applies the merge ────────────────


def test_chat_completions_builder_applies_native_settings(home):
    _stage_dispatch_model(home)
    captured: dict = {}

    def _build_kwargs(**params):
        captured.update(params)
        return {"model": params["model"]}

    transport = SimpleNamespace(build_kwargs=_build_kwargs)
    agent = _agent(tools=[], session_id="s1", max_tokens=256,
                   _base_url_hostname="127.0.0.1", openrouter_min_coding_score=None,
                   providers_allowed=None, providers_ignored=None, providers_order=None,
                   provider_sort=None, provider_require_parameters=None,
                   provider_data_collection=None,
                   _get_transport=lambda: transport,
                   _is_qwen_portal=lambda: False,
                   _is_openrouter_url=lambda: False,
                   _base_url_lower="http://127.0.0.1:8080/v1",
                   _reasoning_endpoint_profile=lambda: None,
                   _prepare_messages_for_non_vision_model=lambda msgs: msgs,
                   _resolved_api_call_timeout=lambda: 30.0,
                   _max_tokens_param=lambda: 256,
                   _ollama_num_ctx=None,
                   _supports_reasoning_extra_body=lambda: False)
    from agent.chat_completion_helpers import _build_chat_completions_kwargs
    _build_chat_completions_kwargs(agent, [{"role": "user", "content": "hi"}],
                                   [], None, {}, None)
    assert captured["request_overrides"] == {"top_p": 0.9}
