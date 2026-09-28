"""Shared-plugins directory fallback for per-directory profiles (#87238).

Per-profile ``HERMES_HOME`` should keep skills scoped while plugins stay
shared via ``HERMES_SHARED_PLUGINS`` (or ``plugins.shared_dir`` in
``config.yaml``). Per-profile ``plugins/`` still wins when present.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins_discovery import collect_directory_manifests, resolve_manifest_winners
from hermes_cli.plugins_manifest import PluginManifest


def _write_plugin(base: Path, name: str) -> Path:
    plugin_dir = base / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": name, "version": "1.0.0"}), encoding="utf-8"
    )
    (plugin_dir / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    return plugin_dir


def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": []}}), encoding="utf-8")
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    return home


def test_shared_dir_env_is_scanned_when_per_profile_plugins_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty per-profile plugins/ dir must fall back to HERMES_SHARED_PLUGINS."""
    home = _isolated_home(tmp_path, monkeypatch)
    shared = tmp_path / "shared-plugins"
    _write_plugin(shared, "shared-only")
    monkeypatch.setenv("HERMES_SHARED_PLUGINS", str(shared))

    names = {m.name for m in collect_directory_manifests()}

    assert "shared-only" in names


def test_shared_dir_config_key_is_scanned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``plugins.shared_dir`` in config.yaml honors the same fallback path."""
    home = _isolated_home(tmp_path, monkeypatch)
    shared = tmp_path / "shared-cfg"
    _write_plugin(shared, "shared-cfg-plugin")
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": [], "shared_dir": str(shared)}}),
        encoding="utf-8",
    )

    names = {m.name for m in collect_directory_manifests()}

    assert "shared-cfg-plugin" in names


def test_env_var_wins_over_config_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HERMES_SHARED_PLUGINS overrides plugins.shared_dir (env wins)."""
    home = _isolated_home(tmp_path, monkeypatch)
    env_dir = tmp_path / "from-env"
    cfg_dir = tmp_path / "from-cfg"
    _write_plugin(env_dir, "env-wins-plugin")
    _write_plugin(cfg_dir, "cfg-loser-plugin")
    monkeypatch.setenv("HERMES_SHARED_PLUGINS", str(env_dir))
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": [], "shared_dir": str(cfg_dir)}}),
        encoding="utf-8",
    )

    names = {m.name for m in collect_directory_manifests()}

    assert "env-wins-plugin" in names
    assert "cfg-loser-plugin" not in names


def test_per_profile_plugin_overrides_shared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-profile plugins/ wins when the same plugin exists in both locations."""
    home = _isolated_home(tmp_path, monkeypatch)
    shared = tmp_path / "shared"
    _write_plugin(shared, "dual")
    per_profile = home / "plugins" / "dual"
    per_profile.mkdir(parents=True, exist_ok=True)
    (per_profile / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "dual", "version": "9.9.9"}), encoding="utf-8"
    )
    (per_profile / "__init__.py").write_text(
        "def register(ctx):\n    pass\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_SHARED_PLUGINS", str(shared))

    manifests = collect_directory_manifests()
    dual = [m for m in manifests if m.name == "dual"]
    # Two raw manifests (shared + user); per-profile wins because it's scanned later, so
    # resolve_manifest_winners picks the per-profile (user) one.
    assert len(dual) == 2
    user_winner = next(m for m in dual if m.source == "user")
    assert user_winner.version == "9.9.9"


def test_shared_dir_absent_does_not_break_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting the env var to a missing path is a no-op, not an error."""
    _isolated_home(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_SHARED_PLUGINS", str(tmp_path / "does-not-exist"))

    # Must not raise and must not invent plugins.
    assert collect_directory_manifests() == []


def test_shared_impostor_cannot_claim_bundled_key(tmp_path: Path) -> None:
    """A shared manifest from a differently named dir must not shadow a bundled key."""
    bundled = PluginManifest(
        name="kanban", version="1.0.0", source="bundled",
        path=str(tmp_path / "bundled" / "kanban"), key="kanban",
    )
    impostor = PluginManifest(
        name="kanban", version="1.0.0", source="shared",
        path=str(tmp_path / "shared" / "some-other-dir"), key="kanban",
    )
    winners = resolve_manifest_winners([bundled, impostor])
    assert winners["kanban"].source == "bundled"

    # A same-named shared copy is still the documented override.
    override = PluginManifest(
        name="kanban", version="2.0.0", source="shared",
        path=str(tmp_path / "shared" / "kanban"), key="kanban",
    )
    winners = resolve_manifest_winners([bundled, override])
    assert winners["kanban"].source == "shared"


def test_shared_source_uses_directory_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shared manifest must load via _load_directory_module, not the entry-point loader."""
    import types

    from hermes_cli.plugins import PluginManager

    home = _isolated_home(tmp_path, monkeypatch)
    plugin_dir = _write_plugin(tmp_path / "shared", "shared-probe")
    manifest = PluginManifest(
        name="shared-probe", version="1.0.0", source="shared",
        path=str(plugin_dir), key="shared-probe",
    )
    module = types.ModuleType("shared_probe")
    module.register = lambda ctx: None  # noqa: E731

    monkeypatch.setattr(
        PluginManager, "_load_directory_module", lambda self, m, **kw: module
    )

    def _fail_entrypoint(self, m, **kw):
        raise AssertionError("shared must not reach the entry-point loader")

    monkeypatch.setattr(PluginManager, "_load_entrypoint_module", _fail_entrypoint)

    mgr = PluginManager()
    assert mgr.home_path == home
    mgr._load_plugin(manifest)
    assert mgr._plugins["shared-probe"].enabled
    assert mgr._plugins["shared-probe"].module is module
