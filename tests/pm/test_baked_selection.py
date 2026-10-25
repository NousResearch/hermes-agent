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
    assert calls == [(["all", "messaging", "otlp"], {"explicit": True})]


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
