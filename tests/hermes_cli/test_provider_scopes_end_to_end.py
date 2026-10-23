"""Named provider scopes: N credentials for ONE provider endpoint stay independent.

A "scope" is a named ``providers:`` entry that reuses a provider's endpoint under its own
``key_env`` (``opencode-zen-work`` / ``opencode-zen-personal``). The wiring already resolved
per-scope credentials; what broke was identity recovery — with two scopes on the SAME
``base_url`` the URL reverse lookup is indiferenciable and returned whichever entry came
first in ``providers:``, so a resumed session billed the other scope's key (#118285).

These tests drive the real chain (config loader → named-provider lookup →
``resolve_runtime_provider`` / picker payload) against a temp ``HERMES_HOME``: a mocked
chain is what hid the mix-up.
"""

from __future__ import annotations

import pytest

URL = "https://opencode.ai/zen/v1"
WORK_KEY = "sk-work-scope"
PERSONAL_KEY = "sk-personal-scope"

CONFIG = f"""\
model:
  provider: custom:opencode-zen-personal
  default: kimi-k2.6
providers:
  opencode-zen-work:
    name: OpenCode Zen (work)
    api: {URL}
    key_env: OPENCODE_ZEN_WORK_API_KEY
    default_model: kimi-k2.6
  opencode-zen-personal:
    name: OpenCode Zen (personal)
    api: {URL}
    key_env: OPENCODE_ZEN_PERSONAL_API_KEY
    default_model: kimi-k2.6
  acme-zen:
    name: Acme (own scope name)
    api: {URL}
    key_env: OPENCODE_ZEN_WORK_API_KEY
    default_model: kimi-k2.6
"""


@pytest.fixture
def scoped_home(tmp_path, monkeypatch):
    """Two scopes of one provider: same endpoint, different credentials."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("OPENCODE_ZEN_API_KEY", raising=False)
    monkeypatch.setenv("OPENCODE_ZEN_WORK_API_KEY", WORK_KEY)
    monkeypatch.setenv("OPENCODE_ZEN_PERSONAL_API_KEY", PERSONAL_KEY)
    return home


def test_each_scope_resolves_its_own_credential(scoped_home):
    """Same endpoint, two credentials: each scope must produce its OWN key."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    work = resolve_runtime_provider(requested="custom:opencode-zen-work", target_model="kimi-k2.6")
    personal = resolve_runtime_provider(requested="custom:opencode-zen-personal", target_model="kimi-k2.6")

    assert work["api_key"] == WORK_KEY
    assert personal["api_key"] == PERSONAL_KEY
    assert work["base_url"].rstrip("/") == personal["base_url"].rstrip("/") == URL.rstrip("/")


def test_per_model_wire_is_a_property_of_the_model_not_the_scope(scoped_home):
    """Per-model wire selection follows the model (and its endpoint family), not the scope's
    name: a scope named after the family and one with an unrelated name route a given model
    the same way."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    def _api_modes(model):
        return {resolve_runtime_provider(requested=f"custom:{scope}", target_model=model)["api_mode"]
                for scope in ("opencode-zen-work", "opencode-zen-personal", "acme-zen")}

    assert len(_api_modes("gpt-5.6-luna")) == 1
    assert _api_modes("kimi-k2.6") != _api_modes("gpt-5.6-luna")


def test_builtin_provider_honours_model_key_env(scoped_home):
    """The one-key-at-a-time alternative for a built-in provider: ``model.key_env`` names the
    scope's variable and the built-in resolves it through the profile secret scope."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    for env_name, key in (("OPENCODE_ZEN_WORK_API_KEY", WORK_KEY),
                          ("OPENCODE_ZEN_PERSONAL_API_KEY", PERSONAL_KEY)):
        scoped_home.joinpath("config.yaml").write_text(
            f"model:\n  provider: opencode-zen\n  default: kimi-k2.6\n  key_env: {env_name}\n",
            encoding="utf-8")
        runtime = resolve_runtime_provider(requested="opencode-zen", target_model="kimi-k2.6")
        assert runtime["provider"] == "opencode-zen"
        assert runtime["api_key"] == key
        assert runtime["source"] == env_name


def test_explicit_scope_wins_over_the_shared_url_when_recovering_identity(scoped_home):
    """The regression: recovery returned the first entry on that URL, so resuming the
    ``personal`` session healed to ``custom:opencode-zen-work`` and billed work's key."""
    from hermes_cli import runtime_provider as rp

    assert rp.canonical_custom_identity(
        base_url=URL, config_provider="opencode-zen-personal", model="kimi-k2.6") == "custom:opencode-zen-personal"
    assert rp.canonical_custom_identity(
        base_url=URL, config_provider="opencode-zen-work", model="kimi-k2.6") == "custom:opencode-zen-work"


def test_display_name_heals_to_its_own_scope(scoped_home):
    """The picker/desktop pass the display name; it must land on the same scope's key."""
    from hermes_cli import runtime_provider as rp
    from hermes_cli.runtime_provider_custom import _get_named_custom_provider

    identity = rp.canonical_custom_identity(base_url=URL, config_provider="OpenCode Zen (personal)")
    assert identity == "custom:opencode-zen-personal"
    assert str((_get_named_custom_provider(identity) or {}).get("key_env")) == "OPENCODE_ZEN_PERSONAL_API_KEY"


def test_picker_lists_one_row_per_scope(scoped_home):
    """Both scopes are selectable rows, and the configured one is the current row."""
    from hermes_cli.inventory import build_models_payload, load_picker_context

    payload = build_models_payload(load_picker_context(), for_picker=True, probe_custom_providers=False,
                                   non_blocking_catalogs=True)
    rows = {r["slug"]: r for r in payload["providers"]}

    assert {"opencode-zen-work", "opencode-zen-personal"} <= set(rows)
    assert rows["opencode-zen-personal"]["is_current"] is True
    assert rows["opencode-zen-work"]["is_current"] is False


def test_cli_resume_heals_a_bare_row_to_the_scope_the_row_names(scoped_home):
    """The CLI restore path must not fall back to the first entry on the URL either.

    ``stored_session_route`` reads the row through ``session_gateway_runtime``, which prefers the
    nested ``gateway_runtime`` shape; that shape can hold the bare billing class ``custom`` while the
    row's top-level key still names the scope. Healing from the endpoint alone sent the resume to
    ``opencode-zen-work`` — the wrong scope's key (#118285).
    """
    import json

    from hermes_cli.cli_model_switch_mixin import stored_session_route

    row = {
        "model": "kimi-k2.6",
        "model_config": json.dumps({
            "gateway_runtime": {"provider": "custom", "base_url": URL, "api_mode": "chat_completions"},
            "provider": "custom:opencode-zen-personal",
            "base_url": URL,
        }),
    }

    route = stored_session_route(row, current_model="other-model", current_provider="other-provider")

    assert route[1] == "custom:opencode-zen-personal"
