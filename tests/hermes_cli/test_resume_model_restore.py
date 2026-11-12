"""Tests for CLI resume model restoration and /model session persistence.

Covers _restore_session_model, _persist_model_switch_to_session (cli.py) and
SessionDB.session_gateway_runtime (hermes_state.py) — the round trip that
makes `hermes --resume` reopen a session on the model/provider it actually
used instead of the ambient config default (#57588-class, #79536).
"""

import json

import pytest

import cli as cli_mod
from hermes_state import SessionDB


def _make_stub(**overrides):
    """Bare HermesCLI the way resume paths see it (no __init__)."""
    stub = object.__new__(cli_mod.HermesCLI)
    stub.model = "ambient-model"
    stub.provider = "openrouter"
    stub.requested_provider = "openrouter"
    stub.base_url = "https://openrouter.ai/api/v1"
    stub.api_key = "ambient-key"
    stub.api_mode = ""
    stub.agent = None
    stub._console_print = lambda s: None
    for key, value in overrides.items():
        setattr(stub, key, value)
    return stub


def _row(model="glm-4.7", model_config=None):
    return {
        "model": model,
        "model_config": json.dumps(model_config) if model_config else None,
    }


# ── SessionDB.session_gateway_runtime ───────────────────────────────


def test_session_gateway_runtime_prefers_nested_key():
    meta = _row(model_config={
        "gateway_runtime": {"provider": "custom:feather", "base_url": "https://f/v1"},
        "provider": "openrouter",
    })
    runtime = SessionDB.session_gateway_runtime(meta)
    assert runtime["provider"] == "custom:feather"
    assert runtime["base_url"] == "https://f/v1"


def test_session_gateway_runtime_falls_back_to_top_level_keys():
    # The TUI gateway's _runtime_model_config writes top-level keys only.
    meta = _row(model_config={"provider": "nous", "api_mode": "chat_completions"})
    runtime = SessionDB.session_gateway_runtime(meta)
    assert runtime == {"provider": "nous", "api_mode": "chat_completions"}


def test_session_gateway_runtime_tolerates_garbage():
    assert SessionDB.session_gateway_runtime(None) == {}
    assert SessionDB.session_gateway_runtime({}) == {}
    assert SessionDB.session_gateway_runtime({"model_config": "{not json"}) == {}
    assert SessionDB.session_gateway_runtime({"model_config": json.dumps([1, 2])}) == {}


# ── _restore_session_model ──────────────────────────────────────────


def test_restore_session_model_restores_model_and_provider():
    stub = _make_stub()
    stub._restore_session_model(_row(model_config={
        "gateway_runtime": {"provider": "custom:feather", "base_url": "https://f/v1"},
    }))
    assert stub.model == "glm-4.7"
    assert stub.provider == "custom:feather"
    assert stub.requested_provider == "custom:feather"
    assert stub.base_url == "https://f/v1"
    # Stale launch-time explicit overrides must not leak into the restored
    # provider's credential resolution.
    assert stub._explicit_api_key is None
    assert stub._explicit_base_url == "https://f/v1"


def test_restore_llamacpp_session_follows_live_managed_endpoint(monkeypatch):
    """Managed llama.cpp owns its live port. A snapshot of last boot's loopback
    URL must not pin the resumed client to a dead ephemeral endpoint."""
    live = {
        "provider": "custom",
        "base_url": "http://127.0.0.1:18434/v1",
        "api_key": "sk-live-managed",
        "api_mode": "chat_completions",
    }
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kw: live)
    stub = _make_stub()
    stub._restore_session_model(_row(
        model="Qwen3.8-27B-UD-IQ3_XXS",
        model_config={
            "provider": "llamacpp",
            "base_url": "http://127.0.0.1:51489/v1",
            "api_mode": "chat_completions",
        }))
    assert stub.provider == "llamacpp"
    assert stub.requested_provider == "llamacpp"
    assert stub.base_url == live["base_url"]
    assert stub._explicit_base_url in (None, "")
    assert stub.api_key == live["api_key"]


def test_restore_llamacpp_session_keeps_launch_base_url_for_same_provider(monkeypatch):
    """`hermes --provider llamacpp --base-url X --resume` is user intent for the SAME provider the
    session ran on; the live-endpoint re-resolution must not re-point it at the local supervisor."""
    calls = []
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kw: calls.append(kw) or {"base_url": "http://127.0.0.1:18434/v1"})
    user_url = "http://gpu-box:8080/v1"
    stub = _make_stub(provider="llamacpp", requested_provider="llamacpp", base_url=user_url,
                      _explicit_base_url=user_url)
    stub._restore_session_model(_row(
        model="Qwen3.8-27B-UD-IQ3_XXS",
        model_config={"provider": "llamacpp", "base_url": "http://127.0.0.1:51489/v1"}))
    assert stub.model == "Qwen3.8-27B-UD-IQ3_XXS"
    assert stub.base_url == user_url
    assert stub._explicit_base_url == user_url
    assert calls == []


def test_restore_session_model_explicit_cli_flag_wins():
    stub = _make_stub(model="cli-flag-model", _explicit_model_override=True)
    stub._restore_session_model(_row())
    assert stub.model == "cli-flag-model"
    assert stub.provider == "openrouter"


def test_restore_session_model_no_stored_model_is_noop():
    stub = _make_stub()
    stub._restore_session_model(_row(model=None))
    assert stub.model == "ambient-model"


def test_restore_session_model_matching_state_is_silent_noop():
    notes = []
    stub = _make_stub(model="glm-4.7", provider="custom:feather",
                      requested_provider="custom:feather",
                      _console_print=lambda s: notes.append(s))
    stub._restore_session_model(_row(model_config={
        "gateway_runtime": {"provider": "custom:feather"},
    }))
    assert not notes


def test_restore_session_model_swaps_running_agent_in_place():
    calls = {}

    class _Agent:
        def switch_model(self, **kwargs):
            calls.update(kwargs)

    stub = _make_stub(agent=_Agent())
    stub._restore_session_model(_row())
    assert calls["new_model"] == "glm-4.7"


# ── _persist_model_switch_to_session ────────────────────────────────


class _Result:
    new_model = "deepseek-v4-flash-free"
    target_provider = "custom:opencode-zen"
    base_url = "https://oz/v1"
    api_mode = ""


def test_persist_model_switch_writes_model_and_both_route_shapes():
    written = []

    class _DB:
        def update_session_model(self, sid, model, **route):
            written.append((sid, model, route))

    stub = _make_stub(_session_db=_DB(), session_id="s1")
    stub._persist_model_switch_to_session(_Result())
    assert written == [("s1", "deepseek-v4-flash-free", {
        "provider": "custom:opencode-zen", "base_url": "https://oz/v1", "api_mode": None})]


def test_persist_model_switch_clears_stale_route_keys(tmp_path, monkeypatch):
    """A later switch must not inherit the previous switch's api_mode/base_url.

    patch_session_model_config merges key-level and only deletes on explicit
    None — dropping falsy values from the patch left the FIRST switch's
    api_mode (e.g. anthropic_messages) alive under the SECOND switch's
    provider, corrupting the wire protocol on TUI/desktop resume.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="stale1", source="cli", model="m0")
    stub = _make_stub(_session_db=db, session_id="stale1")

    class _First:
        new_model = "claude-x"
        target_provider = "custom:feather"
        base_url = "https://feather/v1"
        api_mode = "anthropic_messages"

    class _Second:
        new_model = "gpt-5.4"
        target_provider = "openrouter"
        base_url = "https://openrouter.ai/api/v1"
        api_mode = ""  # openrouter default — must ERASE the anthropic mode

    stub._persist_model_switch_to_session(_First())
    stub._persist_model_switch_to_session(_Second())

    meta = db.get_session("stale1")
    config = json.loads(meta["model_config"])
    # Top-level keys: stale values deleted.
    assert config["provider"] == "openrouter"
    assert "api_mode" not in config, config  # stale anthropic_messages deleted
    # Nested gateway_runtime: stale values replaced with None (the merge
    # replaces the entire gateway_runtime dict, not deep-merging its keys).
    # The reader's `or None` / `if v` filtering treats None the same as
    # absent, so stale values are effectively erased.
    gw = config.get("gateway_runtime", {})
    assert gw.get("provider") == "openrouter"
    assert gw.get("api_mode") is None  # stale anthropic_messages erased
    runtime = SessionDB.session_gateway_runtime(meta)
    assert runtime["provider"] == "openrouter"
    assert "api_mode" not in runtime




def test_persist_model_switch_swallows_db_errors():
    class _DB:
        def update_session_model(self, *a):
            raise RuntimeError("disk full")

    stub = _make_stub(_session_db=_DB(), session_id="s1")
    stub._persist_model_switch_to_session(_Result())  # must not raise


def test_persist_model_switch_heals_bare_custom(tmp_path, monkeypatch):
    """Name an endpoint's configured owner; preserve an explicit unmatched endpoint."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = tmp_path / "config.yaml"
    config.write_text(
        "model:\n  default: ambient-model\n  provider: openai\n"
        "providers:\n  myendpoint:\n    base_url: https://my-endpoint/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("hermes_cli.local_runtime.endpoint._state_endpoint", lambda: None)
    written = {}

    class _DB:
        def update_session_model(self, sid, model, **route):
            written["model"] = model
            written["route"] = route

    class _BareResult:
        new_model = "qwen3.6-plus"
        target_provider = "custom"
        base_url = "https://my-endpoint/v1"
        api_mode = ""

    stub = _make_stub(_session_db=_DB(), session_id="s1")
    stub._persist_model_switch_to_session(_BareResult())
    assert written["route"]["provider"] == "custom:myendpoint"

    # An explicit unmatched endpoint remains a resumable bare custom route.
    config.write_text("model:\n  default: ambient-model\n  provider: openai\n", encoding="utf-8")
    written.clear()
    stub._persist_model_switch_to_session(_BareResult())
    assert written["route"]["provider"] == "custom"
    assert written["route"]["base_url"] == "https://my-endpoint/v1"


@pytest.mark.parametrize(
    "resolver_url,resolver_key,expected_key",
    [
        ("https://old.example.test/v1", "old-endpoint-key", None),
        ("https://new-unmatched.example.test/v1", "new-endpoint-key", "new-endpoint-key"),
    ],
)
def test_unmatched_custom_switch_restores_new_endpoint_with_bound_credentials(
    tmp_path, monkeypatch, resolver_url, resolver_key, expected_key,
):
    from hermes_cli import runtime_provider
    from types import SimpleNamespace

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "custom_providers: []" + chr(10) + "model:" + chr(10)
        + "  default: old-model" + chr(10) + "  provider: openai" + chr(10),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    old_url = "https://old.example.test/v1"
    new_url = "https://new-unmatched.example.test/v1"
    db = SessionDB(db_path=home / "state.db")
    db.create_session("custom-route", source="cli", model="old-model")
    db.update_session_model("custom-route", "old-model", provider="openai",
                            base_url=old_url, api_mode="chat_completions")

    # The resolver must not replace a persisted endpoint with its ambient default.
    monkeypatch.setattr(
        runtime_provider, "resolve_runtime_provider",
        lambda **_kwargs: {"api_key": resolver_key, "base_url": resolver_url,
                           "api_mode": "chat_completions"},
    )
    stub = _make_stub(
        _session_db=db, session_id="custom-route", model="review-model",
        provider="custom", requested_provider="custom", base_url=new_url,
        api_mode="chat_completions", api_key="ambient-secret",
        _explicit_model_override=False, _explicit_base_url=None,
    )

    class _UnmatchedCustom:
        new_model = "review-model"
        target_provider = "custom"
        base_url = new_url
        api_mode = "chat_completions"

    stub._persist_model_switch_to_session(_UnmatchedCustom())
    row = db.get_session("custom-route")
    config = json.loads(row["model_config"])
    runtime = config["gateway_runtime"]
    assert config["provider"] == runtime["provider"] == "custom"
    assert config["base_url"] == runtime["base_url"] == new_url
    assert config["api_mode"] == runtime["api_mode"] == "chat_completions"
    assert runtime["model"] == "review-model"

    resumed = _make_stub(
        model="old-model", provider="openai", requested_provider="openai",
        base_url=old_url, api_mode="chat_completions", api_key="ambient-secret",
        _explicit_model_override=False, _explicit_base_url=None,
    )
    resumed._restore_session_model(row, quiet=True)
    assert resumed.model == "review-model"
    assert resumed.provider == "custom"
    assert resumed.base_url == new_url
    assert resumed.api_mode == "chat_completions"
    assert resumed.api_key == expected_key


def test_provider_change_same_endpoint_resolver_failure_clears_ambient_credentials(
    tmp_path, monkeypatch,
):
    from hermes_cli import runtime_provider

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "custom_providers: []" + chr(10) + "model:" + chr(10)
        + "  default: old-model" + chr(10) + "  provider: openai" + chr(10),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    session_id = "same-endpoint-provider-change"
    endpoint = "https://same.example.test/v1"
    db = SessionDB(db_path=home / "state.db")
    db.create_session(session_id, source="cli", model="old-model")
    db.update_session_model(session_id, "old-model", provider="openai",
                            base_url=endpoint, api_mode="chat_completions")

    switch = _make_stub(
        _session_db=db, session_id=session_id, model="new-model",
        provider="custom", requested_provider="custom", base_url=endpoint,
        api_mode="chat_completions",
    )

    class _CustomSwitch:
        new_model = "new-model"
        target_provider = "custom"
        base_url = endpoint
        api_mode = "chat_completions"

    switch._persist_model_switch_to_session(_CustomSwitch())
    row = db.get_session(session_id)

    resolver_calls = []
    def fail_resolve(**kwargs):
        resolver_calls.append(kwargs)
        raise RuntimeError("injected credential resolver failure")
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", fail_resolve)

    class _Agent:
        def __init__(self, owner):
            self.owner, self.calls = owner, []
        def switch_model(self, **kwargs):
            self.calls.append({"kwargs": kwargs, "pool": self.owner._credential_pool})

    resumed = _make_stub(
        model="old-model", provider="openai", requested_provider="openai",
        base_url=endpoint, api_mode="chat_completions", api_key="ambient-secret",
        _explicit_model_override=False, _explicit_base_url=None,
        _credential_pool="ambient-pool",
    )
    resumed.agent = _Agent(resumed)
    resumed._restore_session_model(row, quiet=True)

    assert resolver_calls and resolver_calls[0]["requested"] == "custom"
    assert resumed.model == "new-model" and resumed.provider == "custom"
    assert resumed.base_url == endpoint and resumed.api_mode == "chat_completions"
    assert resumed.api_key is None and resumed._credential_pool is None
    assert resumed.agent.calls[0]["kwargs"] == {
        "new_model": "new-model", "new_provider": "custom", "api_key": "",
        "base_url": endpoint, "api_mode": "chat_completions",
    }
    assert resumed.agent.calls[0]["pool"] is None


def test_restore_session_model_heals_bare_custom_stored_rows(monkeypatch):
    """Rows persisted by older builds may carry bare 'custom' — heal or drop."""
    import hermes_cli.runtime_provider as rp
    monkeypatch.setattr(rp, "canonical_custom_identity",
                        lambda base_url=None, model=None: None)
    stub = _make_stub()
    stub._restore_session_model(_row(model_config={
        "gateway_runtime": {"provider": "custom"},
    }))
    # Provider dropped -> model restored but provider stays ambient.
    assert stub.model == "glm-4.7"
    assert stub.provider == "openrouter"


def test_restore_session_model_recovers_route_from_billing_columns(monkeypatch):
    """#136323 end-to-end: a CLI session that never ran /model has only the model column plus the
    billing class/endpoint. Resume must run it on the custom entry that endpoint belongs to, not
    the ambient default."""
    import hermes_cli.runtime_provider as rp
    monkeypatch.setattr(rp, "canonical_custom_identity",
                        lambda base_url=None, model=None: "custom:glm")
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kw: {"base_url": "http://127.0.0.1:1234/v1", "api_key": "sk-glm"})
    meta = _row(model="GLM-5.3")
    meta["billing_provider"] = "custom"
    meta["billing_base_url"] = "http://127.0.0.1:1234/v1"
    stub = _make_stub()
    stub._restore_session_model(meta)
    assert stub.model == "GLM-5.3"
    assert stub.provider == "custom:glm"
    assert stub.requested_provider == "custom:glm"
    assert stub.base_url == "http://127.0.0.1:1234/v1"
    assert stub.api_key == "sk-glm"


def test_restore_session_model_billing_custom_without_configured_entry_keeps_ambient(monkeypatch):
    """The billing endpoint no longer matches any configured entry — the heal resolves nothing, so
    the stored model is restored but the route stays ambient (the first turn surfaces the error)."""
    import hermes_cli.runtime_provider as rp
    monkeypatch.setattr(rp, "canonical_custom_identity",
                        lambda base_url=None, model=None: None)
    meta = _row(model="GLM-5.3")
    meta["billing_provider"] = "custom"
    meta["billing_base_url"] = "http://gone-endpoint/v1"
    stub = _make_stub()
    stub._restore_session_model(meta)
    assert stub.model == "GLM-5.3"
    assert stub.provider == "openrouter"
    assert stub.base_url == "https://openrouter.ai/api/v1"


def test_restore_session_model_rederives_per_model_wire_for_opencode_rows(monkeypatch):
    """A row persisted while an opencode-go session ran an anthropic_messages model (MiniMax) must not
    pin that wire onto a chat_completions model on resume — api_mode and the relay URL follow the
    stored model, and a fixed-wire provider's row is still honored verbatim (#96066)."""
    import hermes_cli.runtime_provider as rp
    monkeypatch.setattr(rp, "resolve_runtime_provider", lambda **kw: {"api_key": "go-key"})
    stub = _make_stub(provider="opencode-go", requested_provider="opencode-go",
                      base_url="https://opencode.ai/zen/go/v1", api_mode="chat_completions")
    stub._restore_session_model(_row(model="deepseek-v4-flash-vision-exp", model_config={
        "gateway_runtime": {"provider": "opencode-go", "base_url": "https://opencode.ai/zen/go",
                            "api_mode": "anthropic_messages"}}))
    assert (stub.api_mode, stub.base_url) == ("chat_completions", "https://opencode.ai/zen/go/v1")

    stub = _make_stub()
    stub._restore_session_model(_row(model="MiniMax-M2.5", model_config={
        "gateway_runtime": {"provider": "minimax", "base_url": "https://api.minimax.io/anthropic",
                            "api_mode": "anthropic_messages"}}))
    assert (stub.api_mode, stub.base_url) == ("anthropic_messages", "https://api.minimax.io/anthropic")


# ── round trip: persist → get_session shape → restore ───────────────


def test_round_trip_persist_then_restore(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="rt1", source="cli", model="ambient-model")

    stub = _make_stub(_session_db=db, session_id="rt1")
    stub._persist_model_switch_to_session(_Result())

    meta = db.get_session("rt1")
    restored = _make_stub()
    restored._restore_session_model(meta)
    assert restored.model == "deepseek-v4-flash-free"
    assert restored.provider == "custom:opencode-zen"
    assert restored.base_url == "https://oz/v1"


# ── update_session_model provider persistence (#79536) ──────────────


def test_update_session_model_persists_provider(tmp_path, monkeypatch):
    """update_session_model writes $.model + $.provider into model_config."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="s1", source="cli", model="m0")
    db.update_session_model("s1", "claude-x", provider="custom:feather")
    meta = db.get_session("s1")
    assert meta["model"] == "claude-x"
    config = json.loads(meta["model_config"])
    assert config["model"] == "claude-x"
    assert config["provider"] == "custom:feather"


def test_update_session_model_without_provider_preserves_existing(tmp_path, monkeypatch):
    """Without provider, existing $.provider is left untouched."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="s2", source="cli", model="m0")
    db.update_session_model("s2", "claude-x", provider="custom:feather")
    db.update_session_model("s2", "gpt-5.4")  # no provider
    meta = db.get_session("s2")
    config = json.loads(meta["model_config"])
    assert config["model"] == "gpt-5.4"
    assert config["provider"] == "custom:feather"  # preserved


def test_update_session_model_null_model_config_with_provider(tmp_path, monkeypatch):
    """Provider persistence works when model_config starts as NULL."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="s3", source="cli", model="m0")
    # model_config is NULL at creation — update_session_model must create it
    db.update_session_model("s3", "claude-x", provider="minimax")
    meta = db.get_session("s3")
    config = json.loads(meta["model_config"])
    assert config["model"] == "claude-x"
    assert config["provider"] == "minimax"


# ── session_gateway_runtime billing_provider fallback (#85721) ─────


def test_session_gateway_runtime_falls_back_to_billing_provider():
    """Sessions that never ran /model have only billing_provider."""
    meta = {
        "model": "glm-4.7",
        "model_config": None,
        "billing_provider": "minimax",
    }
    runtime = SessionDB.session_gateway_runtime(meta)
    assert runtime == {"provider": "minimax"}


def test_session_gateway_runtime_billing_provider_bare_bucket_ignored():
    """Bare billing buckets (auto/custom) are not routable — skip them."""
    for bare in ("auto", "custom"):
        meta = {
            "model": "m",
            "model_config": None,
            "billing_provider": bare,
        }
        assert SessionDB.session_gateway_runtime(meta) == {}


def test_session_gateway_runtime_explicit_provider_wins_over_billing():
    """Explicit model_config provider takes precedence over billing_provider."""
    meta = _row(model_config={"provider": "nous"})
    meta["billing_provider"] = "minimax"
    runtime = SessionDB.session_gateway_runtime(meta)
    assert runtime == {"provider": "nous"}


def test_session_gateway_runtime_billing_provider_fills_top_level_route_without_provider():
    """A TUI row with top-level base_url/api_mode but no provider keeps that endpoint AND takes the
    provider the session was billed to — the same merge the TUI gateway always did (#125942); a bare
    bucket still contributes nothing."""
    meta = _row(model_config={"base_url": "https://f/v1", "api_mode": "chat_completions"})
    meta["billing_provider"] = "minimax"
    assert SessionDB.session_gateway_runtime(meta) == {
        "provider": "minimax", "base_url": "https://f/v1", "api_mode": "chat_completions"}
    meta["billing_provider"] = "custom"
    assert SessionDB.session_gateway_runtime(meta) == {"base_url": "https://f/v1", "api_mode": "chat_completions"}


def test_session_gateway_runtime_bare_custom_falls_back_to_billing_base_url():
    """#136323: plain CLI turns persist only the billing class + the endpoint they served. The
    reader surfaces both so the bare-custom heal can recover custom:<name> instead of resume
    recombining the stored model with the ambient default."""
    meta = _row()
    meta["billing_provider"] = "custom"
    meta["billing_base_url"] = "http://127.0.0.1:1234/v1"
    assert SessionDB.session_gateway_runtime(meta) == {
        "provider": "custom", "base_url": "http://127.0.0.1:1234/v1"}
    # Without an endpoint there is nothing to attribute — stays filtered (ambient resume).
    del meta["billing_base_url"]
    assert SessionDB.session_gateway_runtime(meta) == {}


GLM_OLD, GLM_NEW = "http://10.0.0.5:8000/v1", "http://10.0.0.6:8000/v1"


def _billing_only_meta():
    meta = _row(model="GLM-5.3", model_config={"api_mode": "chat_completions"})
    meta["billing_provider"] = "custom"
    meta["billing_base_url"] = GLM_OLD
    return meta


def _custom_config(monkeypatch, *entries):
    import hermes_cli.runtime_provider as rp
    config = {"model": {"provider": "anthropic", "default": "claude-opus-5-5"}, "custom_providers": list(entries)}
    monkeypatch.setattr(rp, "load_config", lambda: config)
    monkeypatch.setattr(rp, "_get_model_config", lambda: config["model"])


def test_stored_session_route_follows_a_custom_entry_that_moved(monkeypatch):
    """The entry serving the chat's model now lives at a new URL: resume runs on the entry, which
    resolves its own endpoint, not on the URL the chat last billed."""
    from hermes_cli.cli_model_switch_mixin import stored_session_route
    _custom_config(monkeypatch, {"name": "glm", "base_url": GLM_NEW, "api_key": "glm-key", "model": "GLM-5.3"})

    route = stored_session_route(_billing_only_meta(), current_model="claude-opus-5-5", current_provider="anthropic")

    assert route[:3] == ("GLM-5.3", "custom:glm", None)
    assert route[4] is True


def test_stored_session_route_drops_an_endpoint_no_entry_serves(monkeypatch):
    from hermes_cli.cli_model_switch_mixin import stored_session_route
    _custom_config(monkeypatch)

    route = stored_session_route(_billing_only_meta(), current_model="claude-opus-5-5", current_provider="anthropic")

    assert route == ("GLM-5.3", None, None, None, False)


def test_session_gateway_runtime_bare_auto_ignores_billing_base_url():
    """auto bills through the router — no attributable endpoint, stays filtered."""
    meta = _row()
    meta["billing_provider"] = "auto"
    meta["billing_base_url"] = "http://127.0.0.1:1234/v1"
    assert SessionDB.session_gateway_runtime(meta) == {}


def test_restore_session_model_restores_billing_provider_fallback():
    """End-to-end: _restore_session_model uses billing_provider fallback."""
    stub = _make_stub()
    stub._restore_session_model({
        "model": "glm-4.7",
        "model_config": None,
        "billing_provider": "minimax",
    })
    assert stub.model == "glm-4.7"
    assert stub.provider == "minimax"


def test_same_model_provider_restores_saved_endpoint_and_wire(monkeypatch):
    calls = []
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider",
                        lambda **kw: calls.append(kw) or {
                            "api_key": "resolved", "base_url": "https://saved.invalid/v1"})
    stub = _make_stub(model="same", provider="custom:feather", requested_provider="custom:feather",
                      base_url="https://ambient.invalid/v1", api_mode="chat_completions")
    stub._restore_session_model(_row(model="same", model_config={"gateway_runtime": {
        "model": "same", "provider": "custom:feather", "base_url": "https://saved.invalid/v1",
        "api_mode": "anthropic_messages"}}))
    assert (stub.model, stub.provider, stub.base_url, stub.api_mode) == (
        "same", "custom:feather", "https://saved.invalid/v1", "anthropic_messages")
    assert calls and stub.api_key == "resolved"


def test_same_provider_explicit_launch_endpoint_still_wins(monkeypatch):
    calls = []
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider",
                        lambda **kw: calls.append(kw) or {})
    stub = _make_stub(model="same", provider="custom:feather", requested_provider="custom:feather",
                      base_url="https://explicit.invalid/v1", api_mode="chat_completions",
                      _explicit_base_url="https://explicit.invalid/v1")
    stub._restore_session_model(_row(model="same", model_config={"gateway_runtime": {
        "model": "same", "provider": "custom:feather", "base_url": "https://saved.invalid/v1",
        "api_mode": "anthropic_messages"}}))
    assert stub.base_url == "https://explicit.invalid/v1"
    assert stub.api_mode == "chat_completions"
    assert calls == []


def test_stored_session_route_heals_an_empty_provider_row_unless_the_main_slot_serves_it(monkeypatch):
    """#132648, CLI twin of the TUI heal. A row with a model but no provider resumes on the entry that
    declares the model, with that entry's own URL and wire (the row's may be stamps from turns that went to
    the main slot). A row on the main slot's own model stays there."""
    from hermes_cli.cli_model_switch_mixin import stored_session_route
    _custom_config(monkeypatch, {"name": "glm", "base_url": GLM_NEW, "api_key": "***", "model": "GLM-5.3"},
                   {"name": "proxy", "base_url": GLM_OLD, "api_key": "***", "model": "claude-opus-5-5"})
    stamped = _row(model="GLM-5.3", model_config={"base_url": GLM_OLD, "api_mode": "anthropic_messages"})

    assert stored_session_route(stamped, current_model="claude-opus-5-5", current_provider="anthropic")[:4] == (
        "GLM-5.3", "custom:glm", None, None)
    assert stored_session_route(_row(model="claude-opus-5-5"), current_model="x", current_provider="anthropic")[1] is None
