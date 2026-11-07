"""Every sync that can build PM's first generation of a pre-PM tree carries the old venv's extras.

A main-era install (a release before PM) holds its features in the checkout's own ``venv/`` with no
PM ledger. Whichever path runs its first PM sync (the update's completion bootstrap, the historical
updater takeover, the launch migration, the stdlib ``venv_sync`` entry), the generation it
publishes must keep those features: nothing carries them afterwards, so a sync that records only
the core drops Telegram, Discord and the rest for good while the update reports success.
"""
from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

from pm import paths
from pm.environments import install_state_dir, runtime_facts_path
from pm.lock import Facts


class _Venv:
    """The venv package with a build backend that commits a generation of this install."""

    def __init__(self, root: Path):
        self.root, self.applied = root, []

    def project_root(self):
        return self.root

    def expected_stamp(self, enabled, **inputs):
        return "stamp:" + ",".join(sorted(enabled)) + f":{len(inputs.get('plugin_dirs') or ())}"

    def apply(self, enabled, **_inputs):
        self.applied.append(sorted(enabled))
        environment = install_state_dir(self.root) / "environments" / str(len(self.applied)) / "venv"
        environment.mkdir(parents=True)
        (environment / "pyvenv.cfg").write_text("home = fixture\n")
        return {"environment": environment}


@pytest.fixture()
def legacy_tree(monkeypatch, tmp_path):
    """A checkout with a main-era venv holding the messaging SDKs, and no PM facts yet."""
    import pm.client
    from pm import install

    root = tmp_path / "checkout"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname="hermes-agent"\nversion="1"\n[project.optional-dependencies]\n'
        'all=[]\ntelegram=[]\ndiscord=[]\nfal=[]\n', encoding="utf-8")
    site = root / "venv" / "lib" / "python3.11" / "site-packages"
    for anchor in ("telegram", "discord", "fal_client"):
        (site / anchor).mkdir(parents=True)
    monkeypatch.setattr(paths, "repo_root", lambda: root)
    package = _Venv(root)
    monkeypatch.setattr(install, "get_package", lambda name: package)
    # The client's in-process path, against PM's real selection and publication logic.
    monkeypatch.setattr(pm.client, "is_runtime", lambda: True)
    monkeypatch.setattr(pm.client, "ensure_tools_for_sync", lambda: None)
    # Plugin eviction judges requires-python against the pinned interpreter; this tree has no plugins.
    monkeypatch.setattr("pm.plugin_eviction._interpreter_version", lambda: "3.14.0")
    from hermes_cli import gitlock, venv_sync

    for name in ("arm_completion", "collect_superseded_generations", "publish_launchers",
                 "refuse_foreign_owned_venv"):
        monkeypatch.setattr(venv_sync, name, lambda *a, **k: None)
    monkeypatch.setattr(gitlock, "convert_treeless_checkout_first", lambda root: None)
    assert not runtime_facts_path(root).exists()
    return root


def _completion_bootstrap(root: Path, tmp_path: Path, monkeypatch) -> None:
    """``hermes update``'s completion bootstrap, on a pre-PM tree with no recorded selection."""
    from hermes_cli import update_completion

    result = tmp_path / "result.json"

    def prepared_child(*_args, **_kwargs):
        result.write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(update_completion.subprocess, "call", prepared_child)
    monkeypatch.setattr("pm.receipt.worker_context", lambda update_id: nullcontext())
    request = {"source": str(root), "receipt": {"update_id": "u1"}, "bytecode_cache": str(tmp_path / "bc")}
    assert update_completion._prepare(request, tmp_path / "request.json", result) == 0


def _historical_takeover(root: Path, tmp_path: Path, monkeypatch) -> None:
    """A pre-handoff updater's takeover of the new tree."""
    from hermes_cli import _update_takeover, update_stage

    monkeypatch.setattr(update_stage, "ensure_panel", lambda root: None)
    monkeypatch.setattr(_update_takeover, "_arm_fleet_obligation", lambda root: None)
    monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda root: Path(sys.executable))
    _update_takeover.prepare({"root": str(root), "update_id": "u1"})


def _launch_migration(root: Path, tmp_path: Path, monkeypatch) -> None:
    """The first launch after a pull finishes the source update's dependency sync."""
    from hermes_cli import venv_sync

    venv_sync._sync_source_dependencies(root, arm=False)


def _venv_sync_entry(root: Path, tmp_path: Path, monkeypatch) -> None:
    """``python -m hermes_cli.venv_sync``: the stdlib-only sync entry."""
    from hermes_cli import venv_sync

    assert venv_sync.sync(root) == {"state": "synced", "ok": True}


@pytest.mark.parametrize("first_sync", [_completion_bootstrap, _historical_takeover, _launch_migration,
                                        _venv_sync_entry], ids=lambda entry: entry.__name__.strip("_"))
def test_the_first_pm_generation_keeps_the_features_the_old_venv_held(legacy_tree, tmp_path, monkeypatch,
                                                                       first_sync):
    first_sync(legacy_tree, tmp_path, monkeypatch)

    recorded = (Facts(runtime_facts_path(legacy_tree)).get("venv") or {}).get("extras")
    assert recorded and {"all", "telegram", "discord", "fal"} <= set(recorded), recorded


def test_a_recorded_selection_owns_the_extras_after_the_first_generation(legacy_tree, tmp_path, monkeypatch):
    """Once PM recorded a selection, the old venv (never deleted) is not scanned again: what the
    ledger no longer holds never comes back from it."""
    _completion_bootstrap(legacy_tree, tmp_path, monkeypatch)
    facts = runtime_facts_path(legacy_tree)
    data = json.loads(facts.read_text(encoding="utf-8"))
    data["packages"]["venv"]["extras"] = ["all", "fal"]
    facts.write_text(json.dumps(data), encoding="utf-8")

    _completion_bootstrap(legacy_tree, tmp_path, monkeypatch)

    assert (Facts(facts).get("venv") or {}).get("extras") == ["all", "fal"]
