"""Managed-scope metadata on the dashboard config API (#135859).

``GET /api/config/schema`` must tell Settings surfaces which leaves the managed
layer pins (so they render read-only), and ``PUT /api/config`` must name the
pinned keys it refused instead of answering a bare ``{"ok": true}`` while
``save_config`` silently strips them to a stderr-only note.
"""

from __future__ import annotations

import asyncio
import contextlib
import textwrap

import pytest


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    return home, managed


def _write(path, body):
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()


@pytest.fixture
def router(monkeypatch):
    """The config endpoints with their late-bound web-server seams stubbed out.

    ``_profile_scope``/``_CONFIG_MUTATION_LOCK``/the broadcast are web-server
    wiring this contract does not exercise; ``load_config``/``save_config`` stay
    real so the strip-then-report path runs against a tmp HERMES_HOME.
    """
    import hermes_cli.web_routers.config_env as mod

    monkeypatch.setattr(mod, "_profile_scope", lambda profile: contextlib.nullcontext())
    monkeypatch.setattr(mod, "_config_profile_scope", lambda profile: contextlib.nullcontext())
    monkeypatch.setattr(mod, "_CONFIG_MUTATION_LOCK", contextlib.nullcontext())
    monkeypatch.setattr(mod, "_broadcast_gateway_session_info", lambda: None)
    monkeypatch.setattr(mod, "_schema_with_dynamic_provider_options", dict)
    return mod


def test_schema_lists_managed_keys_and_source(homes, router):
    home, managed = homes
    _write(home / "config.yaml", "model:\n  default: user/model\n")
    _write(managed / "config.yaml", "model:\n  default: managed/model\n")

    out = asyncio.run(router.get_schema())

    assert out["managed_keys"] == ["model.default"]
    assert out["managed_source"] == str(managed)


def test_schema_reports_empty_managed_metadata_without_a_scope(tmp_path, monkeypatch, router):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "absent"))
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()

    out = asyncio.run(router.get_schema())

    assert out["managed_keys"] == []
    assert out["managed_source"] is None


def test_put_reports_edited_pinned_keys_and_does_not_persist_them(homes, router):
    from hermes_cli.web_models import ConfigUpdate

    home, managed = homes
    _write(home / "config.yaml", "model:\n  default: managed/model\n")
    _write(managed / "config.yaml", "model:\n  default: managed/model\n")

    out = asyncio.run(router.update_config(ConfigUpdate(config={"model": "user/model"})))

    assert out["ok"] is True
    assert out["managed_rejected"] == ["model.default"]
    # The pinned leaf never takes the PUT body's value on disk (save_config
    # strips it; the managed overlay re-supplies it on the next load).
    assert "user/model" not in (home / "config.yaml").read_text(encoding="utf-8")


def test_put_resending_the_managed_value_is_not_a_rejection(homes, router):
    """The GET response is overlay-merged, so an untouched form resends the
    managed values verbatim — that round-trip must stay a clean save."""
    from hermes_cli.web_models import ConfigUpdate

    home, managed = homes
    _write(home / "config.yaml", "model:\n  default: managed/model\n")
    _write(managed / "config.yaml", "model:\n  default: managed/model\n")

    out = asyncio.run(router.update_config(ConfigUpdate(config={"model": "managed/model"})))

    assert out == {"ok": True}


def test_sparse_autosave_diff_omitting_the_pinned_key_is_not_a_rejection(homes, router):
    """Desktop autosaves send only the fields that changed; an absent pinned
    key means "unchanged", not "attempted edit"."""
    from hermes_cli.web_models import ConfigUpdate

    home, managed = homes
    _write(home / "config.yaml", "agent:\n  max_turns: 5\n")
    _write(managed / "config.yaml", "model:\n  default: managed/model\n")

    out = asyncio.run(router.update_config(ConfigUpdate(config={"agent": {"max_turns": 42}})))

    assert out == {"ok": True}
    assert "max_turns: 42" in (home / "config.yaml").read_text(encoding="utf-8")


def test_put_without_a_managed_scope_keeps_the_bare_ok_contract(tmp_path, monkeypatch, router):
    from hermes_cli.web_models import ConfigUpdate

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "absent"))
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()

    out = asyncio.run(router.update_config(ConfigUpdate(config={"agent": {"max_turns": 3}})))

    assert out == {"ok": True}
