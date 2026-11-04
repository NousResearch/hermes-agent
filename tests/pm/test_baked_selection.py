"""A Docker image records the extras it bakes, and boot recovery restores them.

Without a recorded selection, the first extra a container installs on use
(``fal``) became the whole selection: the generation built from it replaced the
image environment and dropped every baked extra (aiohttp, so no API server).
"""

from __future__ import annotations

from pathlib import Path

import pytest

PYPROJECT = (
    '[project]\nname="baked"\nversion="1"\n'
    '[project.optional-dependencies]\nall=[]\nmessaging=[]\notlp=[]\nfal=[]\nazure-identity=[]\n'
)


@pytest.fixture
def install(tmp_path, monkeypatch):
    from pm import paths
    from pm.packages import Venv

    root = tmp_path / "hermes"
    root.mkdir()
    (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    store = tmp_path / "tools"
    store.mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: root)
    monkeypatch.setattr(paths, "store_root", lambda: store)
    monkeypatch.setattr(Venv, "expected_stamp",
                        lambda self, extras, plugin_dirs=None: "stamp:" + ",".join(sorted(extras)))
    return root, store


def test_build_records_the_extra_list_as_the_store_selection(install, monkeypatch):
    import pm
    import pm.features
    from pm import build_env
    from pm.lock import Facts

    root, store = install
    monkeypatch.setattr(pm, "build_environment", lambda **kwargs: root / "venv/bin/python")
    # The frozen build's list is the graph; an anchor inventory would drop meta extras like ``all``.
    monkeypatch.setattr(pm.features, "installed_extras", lambda *a, **k: pytest.fail("no inventory expected"))
    assert build_env.main(["--source", str(root), "--out", str(root / "venv"), "--record-selection",
                           "--extra", "all", "--extra", "messaging", "--extra", "otlp", "--extra", "all"]) == 0
    fact = Facts(store / "facts.json").get("venv")
    assert fact == {"stamp": "stamp:all,messaging,otlp", "extras": ["all", "messaging", "otlp"]}


def test_record_selection_refuses_builds_without_an_extra_list(install):
    from pm import build_env

    root, _ = install
    with pytest.raises(SystemExit):
        build_env.main(["--source", str(root), "--out", str(root / "venv"), "--record-selection", "--all-extras"])


@pytest.mark.parametrize("store_extras, expected", [
    (None, ["fal"]),  # main before the fix: the lazy extra becomes the whole selection
    (["all", "messaging", "otlp"], ["all", "fal", "messaging", "otlp"]),
])
def test_first_lazy_extra_extends_the_recorded_image_selection(install, store_extras, expected):
    from pm.install import _facts, _target_selection
    from pm.lock import Facts
    from pm.packages import Venv

    root, store = install
    if store_extras is not None:
        Facts(store / "facts.json").record_state("venv", "image", store_extras)
    fact = _facts().get("venv") or {}  # sync_venv's fallback when the volume records nothing
    enabled, _, _ = _target_selection(Venv(root), fact, extras=["fal"], inputs={"plugin_dirs": []},
                                      repair=False, shipped=None, frozen=None)
    assert enabled == expected


@pytest.mark.parametrize("recorded", [["fal"], []])
def test_refresh_restores_baked_extras_a_volume_lost(install, monkeypatch, recorded):
    import pm.client
    import pm.install
    from pm.environments import runtime_facts_path
    from pm.lock import Facts
    from pm.recovery import refresh_dependencies

    root, store = install
    Facts(store / "facts.json").record_state("venv", "image", ["all", "messaging", "otlp"])
    Facts(runtime_facts_path(root)).record_state("venv", "lazy", recorded)
    calls = []
    monkeypatch.setattr(pm.client, "sync_venv", lambda extras=None, **kwargs: calls.append((extras, kwargs)))
    # The broken generation matches its own stamp: a plain refresh would call it current.
    monkeypatch.setattr(pm.install, "venv_is_current", lambda **kwargs: True)

    assert refresh_dependencies(root) == "restored all, messaging, otlp"
    assert calls == [(["all", "messaging", "otlp"], {"explicit": True, "automatic": ["all", "messaging", "otlp"]})]


def test_refresh_leaves_a_complete_selection_alone(install, monkeypatch):
    import pm.client
    import pm.install
    from pm.environments import runtime_facts_path
    from pm.lock import Facts
    from pm.recovery import refresh_dependencies

    root, store = install
    Facts(store / "facts.json").record_state("venv", "image", ["all", "messaging"])
    Facts(runtime_facts_path(root)).record_state("venv", "lazy", ["all", "fal", "messaging"])
    monkeypatch.setattr(pm.client, "sync_venv", lambda *a, **k: pytest.fail("no rebuild expected"))
    monkeypatch.setattr(pm.install, "venv_is_current", lambda **kwargs: True)
    assert refresh_dependencies(root) == "current"


def test_offline_refresh_keeps_baked_extras_for_the_next_lazy_install(install, monkeypatch):
    import pm.client
    import pm.install
    from pm.environments import runtime_facts_path
    from pm.install import _target_selection
    from pm.lock import Facts
    from pm.packages import Venv
    from pm.recovery import refresh_dependencies

    root, store = install
    Facts(store / "facts.json").record_state("venv", "image", ["all", "messaging", "otlp"])
    Facts(runtime_facts_path(root)).record_state("venv", "lazy", ["fal"])

    def offline(*args, **kwargs):
        raise RuntimeError("uv sync exited 1: no network")

    monkeypatch.setattr(pm.client, "sync_venv", offline)
    monkeypatch.setattr(pm.install, "venv_is_current", lambda **kwargs: True)

    assert refresh_dependencies(root) == "fallback"
    fact = Facts(runtime_facts_path(root)).get("venv")
    assert fact is not None and fact["extras"] == ["fal", "all", "messaging", "otlp"]
    assert "environment" not in fact  # the image environment boots
    # A lazy install before the next boot must not drop the baked extras again.
    enabled, _, _ = _target_selection(Venv(root), fact, extras=["fal"], inputs={"plugin_dirs": []},
                                      repair=False, shipped=None, frozen=None)
    assert enabled == ["all", "fal", "messaging", "otlp"]


@pytest.mark.parametrize("venv", [{"stamp": "s", "extras": "fal"}, {"stamp": "s", "extras": None},
                                  {"stamp": "s", "extras": [1]}, {"stamp": "s"},
                                  {"extras": ["fal"]}, {"stamp": "", "extras": ["fal"]},
                                  {"stamp": 7, "extras": ["fal"]}])
def test_refresh_refuses_an_invalid_recorded_selection(install, monkeypatch, venv):
    import json

    import pm.client
    from pm.environments import runtime_facts_path
    from pm.lock import Facts
    from pm.recovery import refresh_dependencies

    root, store = install
    Facts(store / "facts.json").record_state("venv", "image", ["all", "messaging"])
    path = runtime_facts_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": 1, "packages": {"venv": venv}}), encoding="utf-8")
    monkeypatch.setattr(pm.client, "sync_venv", lambda *a, **k: pytest.fail("must not sync over a corrupt fact"))
    with pytest.raises(ValueError, match="invalid recorded dependency state"):
        refresh_dependencies(root)


def test_refresh_matches_recorded_extras_by_pep685_name(install, monkeypatch):
    import pm.client
    import pm.install
    from pm.environments import runtime_facts_path
    from pm.lock import Facts
    from pm.recovery import refresh_dependencies

    root, store = install
    Facts(store / "facts.json").record_state("venv", "image", ["all", "azure-identity"])
    Facts(runtime_facts_path(root)).record_state("venv", "lazy", ["all", "Azure_Identity", "fal"])
    monkeypatch.setattr(pm.client, "sync_venv", lambda *a, **k: pytest.fail("no rebuild expected"))
    monkeypatch.setattr(pm.install, "venv_is_current", lambda **kwargs: True)
    assert refresh_dependencies(root) == "current"


def test_refresh_ignores_baked_extras_the_source_no_longer_declares(install, monkeypatch):
    import pm.client
    import pm.install
    from pm.environments import runtime_facts_path
    from pm.lock import Facts
    from pm.recovery import refresh_dependencies

    root, store = install
    Facts(store / "facts.json").record_state("venv", "image", ["all", "hindsight"])
    Facts(runtime_facts_path(root)).record_state("venv", "lazy", ["all"])
    monkeypatch.setattr(pm.client, "sync_venv", lambda *a, **k: pytest.fail("no rebuild expected"))
    monkeypatch.setattr(pm.install, "venv_is_current", lambda **kwargs: True)
    assert refresh_dependencies(root) == "current"


def test_refresh_boots_the_image_environment_when_it_is_the_selection(install, monkeypatch):
    """An upgrade must not rebuild a copy of the image's venv onto the data volume."""
    import pm.client
    import pm.install
    from pm.environments import runtime_facts_path
    from pm.lock import Facts
    from pm.packages import uv_cache_path
    from pm.recovery import refresh_dependencies

    root, store = install
    Facts(store / "facts.json").record_state("venv", "stamp:all,fal,messaging", ["all", "fal", "messaging"])
    Facts(runtime_facts_path(root)).record_state("venv", "previous-image", ["fal"])
    (uv_cache_path() / "archive-v0").mkdir(parents=True)
    monkeypatch.setattr(pm.client, "sync_venv", lambda *a, **k: pytest.fail("no rebuild expected"))
    venv_is_current = pm.install.venv_is_current
    monkeypatch.setattr(pm.install, "venv_is_current", lambda **kwargs: False)

    assert refresh_dependencies(root) == "image"
    assert Facts(runtime_facts_path(root)).get("venv") is None
    assert not uv_cache_path().exists()

    # The next boot of the same image finds nothing to do: no rebuild on every boot.
    monkeypatch.setattr(pm.install, "venv_is_current", venv_is_current)
    (root / "venv").mkdir()
    (root / "venv" / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    assert refresh_dependencies(root) == "current"


def test_refresh_without_room_for_a_rebuild_boots_the_image_and_keeps_the_selection(install, monkeypatch):
    import pm.client
    import pm.install
    import pm.packages
    from pm.environments import runtime_facts_path
    from pm.lock import Facts
    from pm.recovery import refresh_dependencies

    root, store = install
    Facts(store / "facts.json").record_state("venv", "stamp:all,messaging", ["all", "messaging"])
    Facts(runtime_facts_path(root)).record_state("venv", "previous-image", ["all", "fal", "messaging"])
    monkeypatch.setattr(pm.packages, "uv_cache_seed_bytes", lambda: 1 << 62)
    monkeypatch.setattr(pm.client, "sync_venv", lambda *a, **k: pytest.fail("no rebuild expected"))
    monkeypatch.setattr(pm.install, "venv_is_current", lambda **kwargs: False)

    assert refresh_dependencies(root) == "fallback"
    fact = Facts(runtime_facts_path(root)).get("venv")
    assert fact is not None and fact["extras"] == ["all", "fal", "messaging"]
    assert "environment" not in fact


def test_refresh_rebuilds_a_selection_with_plugin_members_instead_of_booting_the_image(install, monkeypatch):
    """The image ships no plugins: booting it would drop a plugin's Python dependencies."""
    import pm.client
    import pm.install
    import pm.workspace
    from pm.environments import runtime_facts_path
    from pm.lock import Facts
    from pm.packages import Venv
    from pm.recovery import refresh_dependencies

    root, store = install
    members = [root / "plugins" / "with-deps"]
    monkeypatch.setattr(pm.workspace, "build_member_dirs", lambda extras, python=None: members)
    monkeypatch.setattr(Venv, "expected_stamp", lambda self, extras, plugin_dirs=None: "stamp:" + ",".join(
        sorted(extras)) + "".join(f"+{d.name}" for d in (pm.workspace.build_member_dirs(extras)
                                                        if plugin_dirs is None else plugin_dirs)))
    Facts(store / "facts.json").record_state("venv", "stamp:all,fal,messaging", ["all", "fal", "messaging"])
    Facts(runtime_facts_path(root)).record_state("venv", "previous-image", ["all", "fal", "messaging"])
    rebuilt = []
    monkeypatch.setattr(pm.client, "sync_venv", lambda *a, **k: rebuilt.append(a))
    monkeypatch.setattr(pm.install, "venv_is_current", lambda **kwargs: False)

    assert refresh_dependencies(root) == "rebuilt"
    assert rebuilt


def _disk(free: int):
    from types import SimpleNamespace

    return lambda path: SimpleNamespace(total=1 << 40, used=(1 << 40) - free, free=free)


def test_refresh_that_fills_the_volume_frees_the_uv_cache_so_the_agent_boots(install, monkeypatch):
    """A failed rebuild on a full volume must not leave it full: every later boot would fail again."""
    import shutil

    import pm.client
    import pm.install
    import pm.packages
    from pm.environments import runtime_facts_path
    from pm.lock import Facts
    from pm.packages import uv_cache_path
    from pm.recovery import refresh_dependencies

    root, store = install
    Facts(store / "facts.json").record_state("venv", "stamp:all,messaging", ["all", "messaging"])
    Facts(runtime_facts_path(root)).record_state("venv", "previous-image", ["all", "fal", "messaging"])
    (uv_cache_path() / "archive-v0").mkdir(parents=True)
    monkeypatch.setattr(pm.packages, "uv_cache_seed_bytes", lambda: 0)
    free = [1 << 40]

    def fills_the_volume(*args, **kwargs):
        free[0] = 0
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(shutil, "disk_usage", lambda path: _disk(free[0])(path))
    monkeypatch.setattr(pm.client, "sync_venv", fills_the_volume)
    monkeypatch.setattr(pm.install, "venv_is_current", lambda **kwargs: False)

    assert refresh_dependencies(root) == "fallback"
    assert not uv_cache_path().exists()
    fact = Facts(runtime_facts_path(root)).get("venv")
    assert fact["extras"] == ["all", "fal", "messaging"] and "environment" not in fact


def test_refresh_that_fails_with_room_to_spare_keeps_the_uv_cache(install, monkeypatch):
    """An offline or broken build is no reason to throw away the cache the retry needs."""
    import shutil

    import pm.client
    import pm.install
    import pm.packages
    from pm.environments import runtime_facts_path
    from pm.lock import Facts
    from pm.packages import uv_cache_path
    from pm.recovery import refresh_dependencies

    root, store = install
    Facts(store / "facts.json").record_state("venv", "stamp:all,messaging", ["all", "messaging"])
    Facts(runtime_facts_path(root)).record_state("venv", "previous-image", ["all", "fal", "messaging"])
    (uv_cache_path() / "archive-v0").mkdir(parents=True)
    monkeypatch.setattr(pm.packages, "uv_cache_seed_bytes", lambda: 0)
    monkeypatch.setattr(shutil, "disk_usage", _disk(1 << 40))

    def offline(*args, **kwargs):
        raise RuntimeError("uv sync exited 1: no network")

    monkeypatch.setattr(pm.client, "sync_venv", offline)
    monkeypatch.setattr(pm.install, "venv_is_current", lambda **kwargs: False)

    assert refresh_dependencies(root) == "fallback"
    assert (uv_cache_path() / "archive-v0").is_dir()


def test_refresh_whose_plugin_discovery_fails_never_keeps_the_previous_generation(install, monkeypatch):
    """A generation resolved against the previous lock must never boot the new code, even when
    reading an enabled plugin's manifest raises before the rebuild starts."""
    import pm.client
    import pm.install
    import pm.workspace
    from pm.environments import runtime_facts_path
    from pm.lock import Facts
    from pm.package import InstallError
    from pm.packages import Venv
    from pm.recovery import refresh_dependencies

    root, store = install

    def broken_manifest(extras, python=None):
        raise InstallError("venv", "plugin 'bad' declares an unsupported manifest_version")

    monkeypatch.setattr(pm.workspace, "build_member_dirs", broken_manifest)
    monkeypatch.setattr(Venv, "expected_stamp", lambda self, extras, plugin_dirs=None: "stamp:" + ",".join(
        sorted(extras)) + ("" if plugin_dirs is not None else str(pm.workspace.build_member_dirs(extras))))
    Facts(store / "facts.json").record_state("venv", "stamp:all,fal,messaging", ["all", "fal", "messaging"])
    Facts(runtime_facts_path(root)).record_state("venv", "previous-image", ["all", "fal"],
                                                 environment=root / "previous" / "venv")
    monkeypatch.setattr(pm.client, "sync_venv", lambda *a, **k: pytest.fail("no rebuild expected"))

    assert refresh_dependencies(root) == "fallback"
    fact = Facts(runtime_facts_path(root)).get("venv")
    assert "environment" not in fact  # the image environment boots, not the previous generation
    assert fact["extras"] == ["all", "fal", "messaging"]


def test_refresh_rebuilds_a_current_selection_its_fallback_unselected(install, monkeypatch):
    """A plain restart whose plugin check fails boots the image environment; once the check
    passes again, the user's selection must be rebuilt, not read as current on the image."""
    import shutil

    import pm.client
    import pm.workspace
    from pm.environments import install_state_dir, runtime_facts_path
    from pm.lock import Facts
    from pm.package import InstallError
    from pm.packages import Venv
    from pm.recovery import refresh_dependencies

    root, store = install
    broken = [True]

    def members(extras, python=None):
        if broken[0]:
            raise InstallError("venv", "plugin 'bad' declares an unsupported manifest_version")
        return []

    monkeypatch.setattr(pm.workspace, "build_member_dirs", members)
    monkeypatch.setattr(Venv, "expected_stamp", lambda self, extras, plugin_dirs=None: "stamp:" + ",".join(
        sorted(extras)) + str(pm.workspace.build_member_dirs(extras) if plugin_dirs is None else plugin_dirs))
    monkeypatch.setattr(shutil, "disk_usage", _disk(1 << 40))
    (root / "venv").mkdir()
    (root / "venv" / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    generation = install_state_dir(root) / "environments" / "otlp" / "venv"
    generation.mkdir(parents=True)
    (generation / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    Facts(store / "facts.json").record_state("venv", "stamp:all,fal,messaging[]", ["all", "fal", "messaging"])
    # Built for this image: only the plugin check fails on this boot.
    Facts(runtime_facts_path(root)).record_state("venv", "stamp:all,fal,messaging,otlp[]",
                                                 ["all", "fal", "messaging", "otlp"], environment=generation)
    rebuilt = []
    monkeypatch.setattr(pm.client, "sync_venv", lambda *a, **k: rebuilt.append(a))

    assert refresh_dependencies(root) == "fallback"
    broken[0] = False
    assert refresh_dependencies(root) == "rebuilt"
    assert rebuilt
