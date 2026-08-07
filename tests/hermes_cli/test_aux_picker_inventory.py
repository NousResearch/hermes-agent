"""Auxiliary-task pickers share one provider-inventory substrate.

Every aux picker (``hermes model`` → Configure auxiliary models, the
``hermes tools`` vision picker, and any future one) must route through
``hermes_cli.inventory.build_aux_picker_rows()`` so it shows the same
provider universe as ``/model``.

Two independent contributor PRs fixed the same two call sites for exactly
this reason:

- #52642 (@deepjia) — user ``providers:`` / ``custom_providers:`` entries
  were invisible because the aux picker never forwarded them.
- #66624 (@Drexuxux) — providers with a fully rate-limited credential pool
  were hidden because the aux picker never forwarded ``for_picker``.

Both were per-call-site kwarg patches, so the next aux picker would have
reintroduced the gap. These tests pin the *shared substrate* behaviour and
guard the seam itself, not the kwargs at any one site.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


CONFIG = {
    "model": {"provider": "openrouter", "default": "anthropic/claude-opus-4.6"},
    "model_catalog": {"excluded_providers": ["copilot"]},
    "providers": {
        "my-llm": {
            "name": "My LLM",
            "base_url": "https://myllm.example.com/v1",
            "key_env": "MYLLM_KEY",
            "discover_models": False,
            "models": {"big-model": {}, "small-model": {}},
        }
    },
    "custom_providers": [
        {
            "name": "Legacy Box",
            "base_url": "https://legacy.example.com/v1",
            "key_env": "LEGACY_KEY",
            "model": "legacy-1",
            "discover_models": False,
        }
    ],
}


@pytest.fixture
def configured_home(tmp_path, monkeypatch):
    """A HERMES_HOME with one ``providers:`` entry and one legacy
    ``custom_providers:`` entry, both credentialled via env."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump(CONFIG))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("MYLLM_KEY", "sk-mine")
    monkeypatch.setenv("LEGACY_KEY", "sk-legacy")
    return home


# ─── The substrate contract ─────────────────────────────────────────────


def test_aux_picker_surfaces_user_defined_providers(configured_home):
    """Both config schemas for a user's own endpoint reach an aux picker.

    This is #52642's bug: the aux picker built its own kwargs and passed
    neither ``user_providers`` nor ``custom_providers``, so a user who had
    configured their own endpoint could not route any auxiliary task to it.
    """
    from hermes_cli.inventory import build_aux_picker_rows

    slugs = {r["slug"] for r in build_aux_picker_rows()}

    assert "my-llm" in slugs, (
        "a keyed providers: entry must be selectable for auxiliary tasks"
    )
    assert "custom:legacy-box" in slugs, (
        "a legacy custom_providers: entry must be selectable for auxiliary tasks"
    )


def test_aux_picker_requests_exhausted_pool_visibility(configured_home):
    """#66624: a provider whose credential pool is entirely rate-limited
    must stay visible. Rate limits are per-model and the aux picker writes a
    config the user runs later, once the cooldown has cleared."""
    from hermes_cli import inventory

    seen = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        return []

    with patch("hermes_cli.model_switch.list_authenticated_providers", _capture):
        inventory.build_aux_picker_rows()

    assert seen.get("for_picker") is True


def test_desktop_model_options_requests_picker_visibility(configured_home):
    """The Desktop/TUI shared catalog must preserve exhausted providers too."""
    from hermes_cli import inventory

    seen = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        return []

    with patch("hermes_cli.model_switch.list_authenticated_providers", _capture):
        inventory.build_model_options_payload(inventory.load_picker_context())

    assert seen.get("for_picker") is True


# ─── Shared rendering ───────────────────────────────────────────────────






# ─── Seam guard ─────────────────────────────────────────────────────────


def test_aux_pickers_route_through_the_shared_substrate(configured_home):
    """Neither aux picker may call ``list_authenticated_providers`` directly.

    This is the regression that produced #52642 and #66624 as two separate
    contributor PRs against the same two call sites: a picker calling the
    low-level function re-derives its own kwargs and drops whichever slice
    of user config the author didn't think about. Both pickers must reach
    the provider list only through ``build_aux_picker_rows``.
    """
    import hermes_cli.main as main
    import hermes_cli.tools_config as tools_config

    direct_calls = []
    substrate_calls = []

    def _direct(**kwargs):
        direct_calls.append(kwargs)
        return []

    def _substrate(**kwargs):
        substrate_calls.append(kwargs)
        return []

    # Abort each picker at its first prompt, right after it has asked for its
    # provider list. The vision picker returns early on empty rows; the
    # aux-task picker still renders auto/custom/back, so cancel its menu.
    with (
        patch("hermes_cli.model_switch.list_authenticated_providers", _direct),
        patch("hermes_cli.inventory.build_aux_picker_rows", _substrate),
        patch("hermes_cli.main._prompt_provider_choice", return_value=None),
    ):
        main._aux_select_for_task("compression")
        tools_config._configure_vision_provider_model({}, {})

    assert len(substrate_calls) == 2, (
        "both the aux-task picker and the vision picker must source providers "
        f"from build_aux_picker_rows (got {len(substrate_calls)} calls)"
    )
    assert direct_calls == [], (
        "aux pickers must not call list_authenticated_providers directly — "
        "route through hermes_cli.inventory.build_aux_picker_rows so custom "
        "providers, exclusions, and picker visibility stay consistent"
    )
