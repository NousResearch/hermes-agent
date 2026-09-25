"""#115079: an INFERRED provider route must not silently persist to config.yaml.

Standing ruling (PR #107366): "Possessing a credential is not selecting a provider."
When a bare ``/model <name>`` is handed to a provider by ``detect_provider_for_model``
(step e) — whose only authorization gate is credential possession — the session switch
may proceed, but writing ``model.provider`` into config.yaml records a route the user
never named. Persistence therefore fails closed unless the user has NAMED the provider:
it is the auth-store active provider (a login/selection), or the config is fresh (the
first pick must persist, ``resolve_persist_behavior``'s documented intent).

The E2E rows drive the real pipeline with ``DASHSCOPE_API_KEY`` present/absent: the
injection itself must move ``provider_inferred`` (a no-op injection invalidates the
guard). Hermetic like ``test_model_switch_configured_provider_routing.py``: catalogs,
aliases, validation and metadata probes are patched; the credential ladder, routing and
the config write are real.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from unittest.mock import patch

import yaml

from hermes_cli.model_switch import ModelSwitchResult, persist_model_selection, switch_model

# cli and tui_gateway.server are imported at COLLECTION time (upstream pattern:
# test_chat_q_exit_clear.py:8, test_audio_playback_guard.py:35), not inside the
# test bodies. Their import triggers hermes_bootstrap's PM dependency activation,
# which probes the PM payload manifest at <checkout-parent>/manifest.json — from a
# git worktree under ~/.hermes/worktrees that probe sits inside the guarded root
# and the real-home tripwire refuses it (the trip is PM's own environment
# resolution, not Hermes state I/O, and it fires identically on plain origin/main;
# see tests/hermes_cli/test_apply_model_switch_result_context.py running
# per-test-import from a worktree). Collection precedes the guard fixture, so the
# probe runs unguarded exactly like every other cli-importing test file.
import cli  # noqa: E402
from tui_gateway import server as _tui_server  # noqa: E402

_ACCEPTED = {"accepted": True, "persist": True, "recognized": True, "message": None}

# A configured (NOT fresh) install: deepseek is the standing route, and an ambient
# DASHSCOPE_API_KEY seeds the credential pool for the unconfigured alibaba provider.
_SEED = (
    "model:\n"
    "  default: deepseek-chat\n"
    "  provider: deepseek\n"
    "agent:\n"
    "  system_prompt: keepme\n"
)


def _seed_home(tmp_path, monkeypatch, config_text: str | None = _SEED, *, dashscope: bool):
    """Isolated HERMES_HOME with a seeded config.yaml and a live deepseek session key;
    the ambient alibaba (DashScope) key is the injected variable under test.
    ``config_text=None`` leaves config.yaml ABSENT (a genuine first-run home)."""
    home = tmp_path / "home"
    home.mkdir(parents=True)
    if config_text is not None:
        (home / "config.yaml").write_text(config_text, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-session")
    if dashscope:
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope-ambient")
    return home


@contextlib.contextmanager
def _offline():
    """Patch out every catalog/network lookup the switch pipeline may reach, mirroring
    ``test_model_switch_configured_provider_routing._run_switch``. The credential ladder
    (env keys), routing and the config write stay REAL."""
    with patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
         patch("hermes_cli.models.cached_provider_model_ids", return_value=[]), \
         patch("hermes_cli.models.model_ids", return_value=[]), \
         patch("hermes_cli.models_validate.validate_requested_model", return_value=_ACCEPTED), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None):
        yield


def _model_block(home) -> dict:
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["model"]


# ---------------------------------------------------------------------------
# (a) the refusal: inferred cross-provider route → no config write, named refusal
# ---------------------------------------------------------------------------

def test_inferred_cross_provider_persist_refuses_and_names_the_provider(tmp_path, monkeypatch):
    """DASHSCOPE_API_KEY alone must not earn a ``provider: alibaba`` line in config.yaml."""
    home = _seed_home(tmp_path, monkeypatch, dashscope=True)
    with _offline():
        result = switch_model(
            raw_input="qwen3.6-plus", current_provider="deepseek", current_model="deepseek-chat",
            is_global=True)
    assert result.success is True, result.error_message
    assert result.target_provider == "alibaba"
    assert result.provider_inferred is True

    refusal = persist_model_selection(result)

    assert isinstance(refusal, str) and refusal
    assert "alibaba" in refusal.lower()
    block = _model_block(home)
    assert (block["default"], block["provider"]) == ("deepseek-chat", "deepseek")
    # The refusal must tell the user how to confirm, and the session switch still won.
    assert "/model" in refusal and "--provider" in refusal


# ---------------------------------------------------------------------------
# (b) control: the user NAMED the provider → persists
# ---------------------------------------------------------------------------

def test_explicitly_named_provider_persists(tmp_path, monkeypatch):
    home = _seed_home(tmp_path, monkeypatch, dashscope=True)
    with _offline():
        result = switch_model(
            raw_input="qwen3.6-plus", current_provider="deepseek", current_model="deepseek-chat",
            is_global=True, explicit_provider="alibaba")
    assert result.success is True, result.error_message
    assert result.provider_inferred is False

    assert persist_model_selection(result) is None

    block = _model_block(home)
    assert (block["default"], block["provider"]) == ("qwen3.6-plus", "alibaba")


# ---------------------------------------------------------------------------
# (c) control: target is the auth-store active provider → persists
# ---------------------------------------------------------------------------

def test_persists_when_target_is_the_auth_store_active_provider(tmp_path, monkeypatch):
    home = _seed_home(tmp_path, monkeypatch, dashscope=True)
    (home / "auth.json").write_text(
        json.dumps({"version": 3, "providers": {}, "active_provider": "alibaba"}),
        encoding="utf-8")
    with _offline():
        result = switch_model(
            raw_input="qwen3.6-plus", current_provider="deepseek", current_model="deepseek-chat",
            is_global=True)
    assert result.provider_inferred is True

    assert persist_model_selection(result) is None

    block = _model_block(home)
    assert (block["default"], block["provider"]) == ("qwen3.6-plus", "alibaba")


# ---------------------------------------------------------------------------
# (d) control: fresh install (no model.default, no model.provider) → persists
# ---------------------------------------------------------------------------

def test_fresh_install_first_pick_persists(tmp_path, monkeypatch):
    """Documented intent (``resolve_persist_behavior``): the first pick must persist so it
    does not evaporate into a stray ``*_API_KEY`` on the next launch."""
    home = _seed_home(
        tmp_path, monkeypatch,
        "model:\n  persist_switch_by_default: false\n", dashscope=True)
    with _offline():
        result = switch_model(
            raw_input="qwen3.6-plus", current_provider="deepseek", current_model="deepseek-chat",
            is_global=True)
    assert result.provider_inferred is True

    assert persist_model_selection(result) is None

    block = _model_block(home)
    assert (block["default"], block["provider"]) == ("qwen3.6-plus", "alibaba")


# ---------------------------------------------------------------------------
# (e) control: same-provider model-only switch → persists (no provider route named)
# ---------------------------------------------------------------------------

def test_same_provider_model_only_switch_persists(tmp_path, monkeypatch):
    home = _seed_home(tmp_path, monkeypatch, dashscope=False)
    with _offline(), patch(
            "hermes_cli.models.detect_provider_for_model",
            return_value=("deepseek", "deepseek-chat-v2")):
        result = switch_model(
            raw_input="deepseek-chat-v2", current_provider="deepseek",
            current_model="deepseek-chat", is_global=True)
    assert result.success is True, result.error_message
    assert result.provider_changed is False
    # Round-1 semantics: the flag is PROVENANCE, not a provider CHANGE — detection fired
    # and the raw input never named the provider, so it reads inferred. The rename still
    # persists because the persist gate additionally compares the target against the
    # DISK provider: the write moves no route, so there is nothing to authorize (#115079).
    assert result.provider_inferred is True

    assert persist_model_selection(result) is None

    block = _model_block(home)
    assert (block["default"], block["provider"]) == ("deepseek-chat-v2", "deepseek")


def test_second_turn_global_persists_no_provider_the_session_inferred(tmp_path, monkeypatch):
    """F4 (round-1 review) — the turn-2 carryover bypass, the incident's exact class:
    turn 1 ``/model qwen3.6-plus`` inferred alibaba (session-only, refused); turn 2 types
    a DIFFERENT bare model with ``--global``. Step e now DETECTS from the current session
    provider (alibaba — already inferred, nothing the user named), the old cross-provider
    condition reads "no provider change" → the flag was not set → the write made the
    inferred provider durable with no user naming. The flag is now provenance-only and
    the persist gate compares the TARGET against the DISK provider, so the write is
    refused and the disk keeps ``openrouter``."""
    home = _seed_home(
        tmp_path, monkeypatch,
        "model:\n  default: some-other-model\n  provider: openrouter\n", dashscope=True)
    with _offline(), patch(
            "hermes_cli.model_switch.list_provider_models", return_value=[]), patch(
            "hermes_cli.models.detect_provider_for_model",
            return_value=("alibaba", "qwen3.6-max")):
        # Production fires detect on the LIVE-catalog hit (``current_provider_catalog_match``)
        # or the static ladder; the rows patch the same seams the sibling rename row (e) uses
        # (step d's aggregator catalog patched empty so turn 1 reaches step e like production).
        # Turn 1: inferred route, session-scoped (never persisted — refused by the gate).
        turn1 = switch_model(
            raw_input="qwen3.6-plus", current_provider="openrouter",
            current_model="some-other-model", is_global=False)
        assert turn1.success is True, turn1.error_message
        assert turn1.target_provider == "alibaba" and turn1.provider_inferred is True
        # Turn 2: same session (provider is now alibaba), a different model, --global.
        turn2 = switch_model(
            raw_input="qwen3.6-max", current_provider="alibaba",
            current_model="qwen3.6-plus", is_global=True)
    assert turn2.success is True, turn2.error_message
    assert turn2.target_provider == "alibaba"
    # Pre-fix this was False (cross-provider term dropped the provenance on turn 2).
    assert turn2.provider_inferred is True, "carryover turn must keep the inferred provenance"

    refusal = persist_model_selection(turn2)

    assert isinstance(refusal, str) and refusal and "alibaba" in refusal.lower()
    block = _model_block(home)
    assert block["provider"] == "openrouter"  # pre-fix: rewritten to alibaba
    assert block["default"] == "some-other-model"


def test_explicit_flag_over_ambient_key_persists_carryover_target(tmp_path, monkeypatch):
    """Companion guard to the carryover row: ``--provider`` IS a selection — the same
    target the refusal above blocks must persist when the user names the provider on the
    command line, even with the ambient key present and a different provider on disk."""
    home = _seed_home(
        tmp_path, monkeypatch,
        "model:\n  default: some-other-model\n  provider: openrouter\n", dashscope=True)
    with _offline():
        result = switch_model(
            raw_input="qwen3.6-max", current_provider="alibaba",
            current_model="qwen3.6-plus", is_global=True, explicit_provider="alibaba")
    assert result.success is True, result.error_message
    assert result.provider_inferred is False

    assert persist_model_selection(result) is None

    block = _model_block(home)
    assert (block["default"], block["provider"]) == ("qwen3.6-max", "alibaba")


# ---------------------------------------------------------------------------
# (f) env-key E2E control rows: the injection must MOVE the provenance flag
# ---------------------------------------------------------------------------

def test_env_key_injection_moves_the_provenance_flag_both_ways(tmp_path, monkeypatch):
    # WITH the ambient key: detection names alibaba → inferred → persist refused.
    home = _seed_home(tmp_path / "with", monkeypatch, dashscope=True)
    with _offline():
        with_key = switch_model(
            raw_input="qwen3.6-plus", current_provider="deepseek", current_model="deepseek-chat",
            is_global=True)
    assert with_key.success is True, with_key.error_message
    assert with_key.target_provider == "alibaba"
    assert with_key.provider_inferred is True
    assert isinstance(persist_model_selection(with_key), str)
    assert _model_block(home)["provider"] == "deepseek"

    # WITHOUT it: the credential gate skips the guess, detection returns None, the switch
    # stays put — provenance clean, nothing refused.
    home_no = _seed_home(tmp_path / "without", monkeypatch, dashscope=False)
    with _offline():
        without_key = switch_model(
            raw_input="qwen3.6-plus", current_provider="deepseek", current_model="deepseek-chat",
            is_global=True)
    assert without_key.success is True, without_key.error_message
    assert without_key.target_provider == "deepseek"
    assert without_key.provider_inferred is False
    assert persist_model_selection(without_key) is None
    assert _model_block(home_no)["provider"] == "deepseek"


# ---------------------------------------------------------------------------
# (h) a TYPED provider name is a selection, not an inference
# ---------------------------------------------------------------------------

def test_typed_provider_name_is_a_selection_and_persists(tmp_path, monkeypatch):
    """``/model alibaba`` reaches step e through detect's NAMING branch (models.py:
    "explicitly named provider: let the credential step report it"), so the provenance
    check must not read the provider DIFF as inference: naming the provider IS selecting
    it, provenance stays clean and ``--global`` persists (#115079 review)."""
    home = _seed_home(tmp_path, monkeypatch, dashscope=True)
    with _offline():
        result = switch_model(
            raw_input="alibaba", current_provider="deepseek", current_model="deepseek-chat",
            is_global=True)
    assert result.success is True, result.error_message
    assert result.target_provider == "alibaba"
    assert result.provider_inferred is False

    assert persist_model_selection(result) is None

    block = _model_block(home)
    assert block["provider"] == "alibaba"
    assert block["default"]  # step 0's bare-provider-name default model, not the old route

# ---------------------------------------------------------------------------
# (i) the naming helper must NEVER suppress the flag for a bare MODEL name —
#     that is the incident class rows (a)/(f) keep refusing
# ---------------------------------------------------------------------------

def test_model_names_never_suppress_the_inference_flag():
    """Unit-level complement of the (a)/(f) E2E rows: the helper decides provenance, so
    pin its boundary directly — an alias entry or a ``vendor/``-prefixed first token that
    resolves to the DETECTED provider counts as named; anything else (above all a bare
    model name) does not."""
    from hermes_cli.model_switch import _Switch, _raw_input_names_detected_provider
    st = _Switch(
        raw_input="", current_provider="deepseek", current_model="deepseek-chat",
        current_base_url="", current_api_key="", is_global=True, explicit_provider="",
        user_providers=None, custom_providers=None)
    named = [
        ("alibaba", "alibaba"),           # bare provider id
        ("DashScope", "alibaba"),         # _PROVIDER_ALIASES entry
        ("alibaba/qwen3.6-plus", "alibaba"),   # provider/model first token
        ("alibaba:qwen3.6-plus", "alibaba"),   # vendor:model form (kept raw until step c)
        # The two alias tables diverge on these keys: detect names the provider under
        # its _PROVIDER_ALIASES id, providers.ALIASES normalizes BOTH that id and the
        # typed token to a third (models.dev) canonical. Both sides must be
        # normalized before comparing, or the typed name reads as un-selected.
        ("moonshot", "kimi-coding"),      # both normalize to kimi-for-coding
        ("zen", "opencode-zen"),          # both normalize to opencode
        ("github", "copilot"),            # both normalize to github-copilot
        ("kilo-code", "kilocode"),        # both normalize to kilo
        # providers.ALIASES does not bridge these at all — detect names them through
        # the catalog table (models_catalog_static._PROVIDER_ALIASES) and the plugin
        # profiles, so the helper must consult that naming source too (round-3 review).
        ("google", "gemini"),             # catalog alias google → gemini
        ("google-vertex", "vertex"),      # catalog alias google-vertex → vertex
    ]
    for raw, detected in named:
        assert _raw_input_names_detected_provider(raw, detected, st) is True, (raw, detected)
    not_named = [
        ("qwen3.6-plus", "alibaba"),      # the incident class: a bare MODEL name
        ("deepseek-v4.1-flash", "alibaba"),
        ("deepseek", "alibaba"),          # naming a DIFFERENT provider names nothing here
        ("kimi-k2.5", "kimi-coding"),     # a bare MODEL name from the SAME alias family
        ("alibaba-cn", "alibaba"),        # a DISTINCT id, not an alias of alibaba
    ]
    for raw, detected in not_named:
        assert _raw_input_names_detected_provider(raw, detected, st) is False, (raw, detected)

def test_models_dev_alias_provider_ids_still_count_as_named(monkeypatch):
    """Round-3 residual: models.dev ITSELF lists ``google`` and ``google-vertex`` as
    provider ids, so ``resolve_provider_full`` resolves the typed name to the pdef whose
    id is the alias (source ``models.dev``), never reaching the plugin-profile rung that
    reports ``gemini``/``vertex``. detect's NAMING branch still fires through the catalog
    table — so the helper must bridge that same naming source or the user who typed the
    provider's own name eats the 'never selected by you' refusal with the wrong copy.
    Reproduces production reality: every populated install serves the models.dev cache.
    """
    from hermes_cli.model_switch import _Switch, _raw_input_names_detected_provider

    class _MDevInfo:
        def __init__(self, name):
            self.name, self.env, self.api, self.doc = name, (), f"https://{name}.test/v1", ""

    def shadowed(name, allow_network=True):
        # mirrors the real cached models.dev entry for these two keys (probed 2026-09)
        return _MDevInfo(name) if name in ("google", "google-vertex") else None

    monkeypatch.setattr("agent.models_dev.get_provider_info", shadowed)
    st = _Switch(
        raw_input="", current_provider="deepseek", current_model="deepseek-chat",
        current_base_url="", current_api_key="", is_global=True, explicit_provider="",
        user_providers=None, custom_providers=None)
    for raw, detected in [("google", "gemini"), ("google-vertex", "vertex")]:
        assert _raw_input_names_detected_provider(raw, detected, st) is True, (raw, detected)
    # fail-toward-flagged is unchanged under the shadow: an unknown token stays flagged
    assert _raw_input_names_detected_provider("googlex", "gemini", st) is False

def test_registry_alias_sweep_every_naming_key_counts_as_named():
    """Round-3 sweep guard: the catalog alias table is detect's OWN naming source
    (``detect_static_provider_for_model`` step 0), so EVERY key whose naming branch
    actually fires must read as NAMED in the persist-gate helper — no per-key bridges.
    A future alias added to the table stays covered by construction; a bare model name
    is still never named (guard row at the end)."""
    from hermes_cli.model_switch import _Switch, _raw_input_names_detected_provider
    from hermes_cli.models import (
        _PROVIDER_ALIASES, _PROVIDER_LABELS, _PROVIDER_MODELS,
        detect_static_provider_for_model)

    st = _Switch(
        raw_input="", current_provider="deepseek", current_model="deepseek-chat",
        current_base_url="", current_api_key="", is_global=True, explicit_provider="",
        user_providers=None, custom_providers=None)
    checked = 0
    for alias, canonical in _PROVIDER_ALIASES.items():
        if canonical in {"custom", "openrouter"}:
            continue  # step 0 refuses to name these: naming branch cannot fire
        if canonical not in _PROVIDER_LABELS or not _PROVIDER_MODELS.get(canonical):
            continue  # no label/catalog: step 0 cannot fire
        detected = detect_static_provider_for_model(alias, "deepseek")
        if not detected or detected[0] != canonical:
            continue  # naming branch did not fire for this key
        checked += 1
        assert _raw_input_names_detected_provider(alias, canonical, st) is True, (alias, canonical)
    assert checked >= 70, f"sweep vacuous: only {checked} naming keys checked of {len(_PROVIDER_ALIASES)}"
    # the incident class survives the sweep: a bare MODEL name is still flagged
    assert _raw_input_names_detected_provider("qwen3.6-plus", "alibaba", st) is False

# ---------------------------------------------------------------------------
# (j) documented fail-closed rule: an unreadable config is NOT a fresh install
# ---------------------------------------------------------------------------

def test_unreadable_config_is_not_fresh_and_refuses(tmp_path, monkeypatch):
    """A gate that opened on "the config looks fresh" when the config merely could not be
    read would hand persistence to exactly the route with the least evidence behind it.
    Real failure path, no simulation: ``config.yaml`` is a DIRECTORY, so the raw read the
    freshness clause performs raises (IsADirectoryError through the real config API).
    Pre-fix this FAILED OPEN: ``load_config`` swallowed the error into a defaults view
    whose ``model`` block is falsy, and the gate read that as "fresh install"."""
    from hermes_cli.model_switch import inferred_provider_persist_refusal
    home = tmp_path / "home"
    home.mkdir(parents=True)
    (home / "config.yaml").mkdir()  # unreadable as a file — raises through read AND parse
    monkeypatch.setenv("HERMES_HOME", str(home))
    with patch("hermes_cli.auth.get_active_provider", return_value=None):
        refusal = inferred_provider_persist_refusal("alibaba", "CONFIRM-HINT")
    assert refusal is not None  # a refusal, not an exception escaping the gate
    assert "alibaba" in refusal.lower()
    assert refusal.endswith("CONFIRM-HINT")

    # Explicit config_path is honoured by the freshness clause too (same file the write
    # would target), and a directory there fails closed the same way.
    fresh_home = tmp_path / "fresh-home"
    fresh_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(fresh_home))  # default config absent
    broken = tmp_path / "gateway-profile" / "config.yaml"
    broken.mkdir(parents=True)
    with patch("hermes_cli.auth.get_active_provider", return_value=None):
        refusal = inferred_provider_persist_refusal(
            "alibaba", "CONFIRM-HINT", config_path=broken)
    assert refusal is not None and refusal.endswith("CONFIRM-HINT")

# ---------------------------------------------------------------------------
# (j2) genuinely absent config file → fresh → the first pick persists
# ---------------------------------------------------------------------------

def test_absent_config_file_is_fresh_and_persists(tmp_path, monkeypatch):
    """Complement of the directory row: NO config.yaml at all is the real first-run
    state — freshness must authorize it (and the write creates the file)."""
    home = _seed_home(tmp_path, monkeypatch, config_text=None, dashscope=True)
    assert not (home / "config.yaml").exists()
    with _offline():
        result = switch_model(
            raw_input="qwen3.6-plus", current_provider="deepseek", current_model="deepseek-chat",
            is_global=True)
    assert result.provider_inferred is True
    assert persist_model_selection(result) is None
    block = _model_block(home)
    assert (block["default"], block["provider"]) == ("qwen3.6-plus", "alibaba")

# ---------------------------------------------------------------------------
# (k) freshness is judged on the RAW file the write targets, not the merged view
# ---------------------------------------------------------------------------

def test_freshness_reads_the_written_file_explicit_config_path_honoured(tmp_path, monkeypatch):
    """The gateway persists to a per-PROFILE ``config_path`` (slash_commands_model.py)
    while ``HERMES_HOME`` points at the default profile. The gate must judge freshness
    from the file it will actually write: a configured profile file is NOT fresh even
    though the default-home config is absent (merged/default view would say fresh)."""
    from hermes_cli.model_switch import inferred_provider_persist_refusal
    home = _seed_home(tmp_path, monkeypatch, config_text=None, dashscope=True)
    assert not (home / "config.yaml").exists()  # default home looks fresh
    profile = tmp_path / "profile-b" / "config.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text(_SEED, encoding="utf-8")  # configured (NOT fresh) profile file
    with patch("hermes_cli.auth.get_active_provider", return_value=None):
        refusal = inferred_provider_persist_refusal(
            "alibaba", "CONFIRM-HINT", config_path=profile)
    assert refusal is not None
    assert "alibaba" in refusal.lower()

    # Same call with the profile file's model block emptied → fresh → authorized.
    profile.write_text("model:\n  persist_switch_by_default: false\n", encoding="utf-8")
    with patch("hermes_cli.auth.get_active_provider", return_value=None):
        assert inferred_provider_persist_refusal(
            "alibaba", "CONFIRM-HINT", config_path=profile) is None

    # And the persist path threads it end-to-end: on the CONFIGURED profile file the
    # refusal is returned and the write touches nothing.
    profile.write_text(_SEED, encoding="utf-8")
    with _offline():
        result = switch_model(
            raw_input="qwen3.6-plus", current_provider="deepseek", current_model="deepseek-chat",
            is_global=True)
    assert result.provider_inferred is True
    refusal = persist_model_selection(result, config_path=profile)
    assert isinstance(refusal, str) and refusal
    assert profile.read_text(encoding="utf-8") == _SEED

# ---------------------------------------------------------------------------
# Refusal message shape (the shared helper T2 also builds on)
# ---------------------------------------------------------------------------

def test_refusal_message_names_provider_and_ends_with_the_confirm_hint(tmp_path, monkeypatch):
    from hermes_cli.model_switch import inferred_provider_persist_refusal
    _seed_home(tmp_path, monkeypatch, dashscope=False)  # configured (not fresh) home
    with patch("hermes_cli.auth.get_active_provider", return_value=None):
        refusal = inferred_provider_persist_refusal("alibaba", "CONFIRM-HINT")
    assert refusal is not None
    assert "alibaba" in refusal.lower()
    assert refusal.endswith("CONFIRM-HINT")

    # Authorized readers: the auth-store active provider is a user selection.
    with patch("hermes_cli.auth.get_active_provider", return_value=" Alibaba "):
        assert inferred_provider_persist_refusal("alibaba", "CONFIRM-HINT") is None

def test_refusal_message_collapses_trailing_hint_whitespace(tmp_path, monkeypatch):
    """GPT-OSS review nit: a surface passing a hint with trailing whitespace must not
    weld a double space (or a dangling newline) onto the shared message."""
    from hermes_cli.model_switch import inferred_provider_persist_refusal
    _seed_home(tmp_path, monkeypatch, dashscope=False)  # configured (not fresh) home
    with patch("hermes_cli.auth.get_active_provider", return_value=None):
        refusal = inferred_provider_persist_refusal("alibaba", "CONFIRM-HINT  \n ")
    assert refusal is not None
    assert refusal.endswith("CONFIRM-HINT")  # pre-fix: ended with the raw trailing run
    assert "  " not in refusal  # one space joins body and hint, never two

# ---------------------------------------------------------------------------
# F2 (round-1 review): gate (i) must compare CANONICAL provider ids
# ---------------------------------------------------------------------------

def test_auth_store_alias_form_active_provider_authorizes(tmp_path, monkeypatch):
    """The auth store writes ``active_provider`` in whatever form the login flow used —
    an alias like ``dashscope`` — while detect hands back the canonical id (``alibaba``).
    A user who DID authenticate to that provider must not eat the "never selected by
    you" refusal because of a spelling difference: the compare goes through
    ``normalize_provider`` on both sides (#115079 round 1)."""
    from hermes_cli.model_switch import inferred_provider_persist_refusal
    _seed_home(tmp_path, monkeypatch, dashscope=False)  # configured (not fresh) home
    with patch("hermes_cli.auth.get_active_provider", return_value="dashscope"):
        assert inferred_provider_persist_refusal("alibaba", "CONFIRM-HINT") is None
    # And the reverse spelling direction resolves the same way.
    with patch("hermes_cli.auth.get_active_provider", return_value="alibaba"):
        assert inferred_provider_persist_refusal("dashscope", "CONFIRM-HINT") is None
    # A genuinely different provider stays refused (no alias bridge for strangers).
    with patch("hermes_cli.auth.get_active_provider", return_value="openrouter"):
        assert inferred_provider_persist_refusal("alibaba", "CONFIRM-HINT") is not None


# ---------------------------------------------------------------------------
# (g) the three persist call sites surface the refusal on their own warning channel
# ---------------------------------------------------------------------------

def _refusing_result(**overrides) -> ModelSwitchResult:
    base = dict(
        success=True, new_model="qwen3.6-plus", target_provider="alibaba",
        provider_changed=True, provider_inferred=True, is_global=True)
    return ModelSwitchResult(**{**base, **overrides})


def test_cli_surface_prints_the_refusal_instead_of_a_saved_line(tmp_path, monkeypatch):
    from hermes_cli import cli_model_switch_mixin as mixin
    printed: list[str] = []
    monkeypatch.setattr(cli, "_cprint", lambda *a, **k: printed.append(" ".join(map(str, a))))
    monkeypatch.setattr(
        "hermes_cli.model_switch.persist_model_selection", lambda *a: "REFUSED: alibaba")
    monkeypatch.setattr(
        cli.HermesCLI, "_persist_model_switch_to_session", lambda *a, **k: None)
    monkeypatch.setattr(mixin, "_print_switch_summary", lambda *a, **k: None)
    stub = type("Stub", (), {
        "agent": None, "model": "deepseek-chat", "_pending_one_turn_model_restore": None,
        "_stage_and_swap_model": lambda self, r, o: True})()

    mixin._commit_model_switch(stub, _refusing_result(), persist_global=True)

    assert any("REFUSED" in line for line in printed)
    assert not any("Saved to config.yaml" in line for line in printed)


def test_gateway_surface_reports_the_refusal_as_the_global_error(tmp_path, monkeypatch):
    from gateway.slash_commands_model import (
        GatewayModelCommandsMixin, _ModelSwitchContext, _persist_model_switch_to_config)
    monkeypatch.setattr(
        "hermes_cli.model_switch.persist_model_selection", lambda *a: "REFUSED: alibaba")
    result = _refusing_result()
    config_path = tmp_path / "config.yaml"

    refusal = asyncio.run(_persist_model_switch_to_config(result, config_path))
    assert refusal == "REFUSED: alibaba"

    class _Store:
        async def set_model_override(self, session_key, override):
            self.saved = override

    class _Runner(GatewayModelCommandsMixin):
        config = None

        def _evict_cached_agent(self, session_key):
            pass

    store = _Store()
    runner = _Runner()
    runner.async_session_store = store
    runner._session_model_overrides = {}
    ctx = _ModelSwitchContext(
        session_key="k", source=None, config_path=config_path, persist_global=True)
    ctx.current_model = "deepseek-chat"

    global_error = asyncio.run(
        runner._record_model_switch(result, ctx, source=None, one_turn=False, picker=False))

    assert global_error == "REFUSED: alibaba"
    # The refused switch keeps its session override instead of config.yaml becoming the
    # durable authority (#100314's failure mode runs backwards here — claiming global while
    # nothing was written is exactly what the refusal prevents).
    assert store.saved["model"] == "qwen3.6-plus"


def test_tui_surface_propagates_the_refusal_into_the_switch_warning(tmp_path, monkeypatch):
    _seed_home(tmp_path, monkeypatch, dashscope=False)  # HERMES_HOME + config sandbox
    server = _tui_server  # collection-time import (see module header)
    result = _refusing_result(warning_message="pre-existing warning")
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: result)
    monkeypatch.setattr(
        "hermes_cli.model_switch.persist_model_selection", lambda *a: "REFUSED: alibaba")
    monkeypatch.setattr("tui_gateway.server._emit", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("tui_gateway.server._restart_slash_worker", lambda *a, **k: None)
    monkeypatch.setattr("tui_gateway.server._session_info", lambda *a, **k: None)

    out = server._apply_model_switch(
        "sid", {"agent": None}, "qwen3.6-plus --global", confirm_expensive_model=True)

    assert out["value"] == "qwen3.6-plus"
    assert "REFUSED: alibaba" in out["warning"]
    assert "pre-existing warning" in out["warning"]
