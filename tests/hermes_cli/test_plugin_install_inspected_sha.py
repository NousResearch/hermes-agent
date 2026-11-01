"""An install dialog's ``plugins.manage install`` is held to the commit its ``inspect`` read.

The dialog shows the user one commit; the branch may move before Install is clicked. The agent half (and
the desktop half the host copies out of that checkout) must never be a tree nobody reviewed: a moved
branch is refused. An install at the reviewed tip stays an unpinned checkout of the branch, so ``update``
still follows it. Real RPC dispatch, real clone of a local repo.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import hermes_yaml as yaml
from tests.hermes_cli.plugin_worker_support import (
    isolated_python as isolated_python,  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import
    plugin_world as plugin_world,  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import
)
from tui_gateway import server


@pytest.fixture(autouse=True)
def _offline_pm(plugin_world):
    (plugin_world.home / "config.yaml").unlink()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, marker: str) -> str:
    (repo / "marker.txt").write_text(marker, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.email=t@example.invalid", "-c", "user.name=t", "commit", "-qm", marker)
    return _git(repo, "rev-parse", "HEAD")


def _install(**params):
    return server.handle_request({"id": "1", "method": "plugins.manage",
                                  "params": {"action": "install", "identifier": "https://example.invalid/acme/demo.git",
                                             "enable": False, **params}})


@pytest.fixture
def moved_repo(tmp_path, monkeypatch):
    """``(home, reviewed_sha, tip_sha, repo)``: a repo the dialog inspected at ``reviewed`` whose branch then moved."""
    from hermes_cli import plugins_cmd
    repo = tmp_path / "demo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "plugin.yaml").write_text(yaml.safe_dump({"name": "demo", "version": "1.0.0"}), encoding="utf-8")
    (repo / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    reviewed = _commit(repo, "reviewed")
    tip = _commit(repo, "moved")
    real = plugins_cmd._clone_plugin_repo
    monkeypatch.setattr(plugins_cmd, "_clone_plugin_repo",
                        lambda dest, _url, revision, subdir=None: real(dest, repo.as_uri(), revision, subdir))
    from hermes_constants import get_hermes_home
    return get_hermes_home(), reviewed, tip, repo


def test_install_refuses_a_branch_that_moved_since_the_dialog_inspected_it(moved_repo):
    home, reviewed, tip, _repo = moved_repo

    refused = _install(inspected_sha=reviewed)

    message = refused["error"]["message"]
    assert "source changed since you reviewed it" in message and tip[:12] in message
    # The client re-inspects on this structured marker, never on the message text.
    assert refused["error"]["data"] == {"reason": "source_moved", "placed": False}
    assert not (home / "plugins" / "demo").exists()


def test_a_clone_failure_echoing_the_refusal_text_carries_no_marker(moved_repo, monkeypatch):
    """A git server controls its clone error's text, so that text can say anything; only the backend's
    own comparison of commits tags the refusal."""
    from hermes_cli import plugins_cmd
    _home, reviewed, _tip, _repo = moved_repo

    def hostile(*_args, **_kwargs):
        raise plugins_cmd.PluginOperationError(
            "Git clone failed: remote: The source changed since you reviewed it: it is now at commit 0a415bc5c1d2. "
            "invalid params for plugins.manage: inspected_sha: Extra inputs are not permitted")
    monkeypatch.setattr(plugins_cmd, "_clone_plugin_repo", hostile)

    refused = _install(inspected_sha=reviewed)["error"]

    # Refused before anything was published: nothing was placed, and no ``source_moved`` from its text.
    assert refused["code"] == 5026 and refused["data"] == {"placed": False}


def test_a_refused_enable_names_the_tree_that_landed(moved_repo, monkeypatch):
    """The install placed the tree, then its enable was refused: the client must know the agent half is
    on disk, so the matching desktop half still goes down beside it."""
    from hermes_cli import plugins_cmd
    from hermes_cli.plugins_admission import AdmissionRefused
    home, _reviewed, tip, _repo = moved_repo

    def refuse(*_args, **_kwargs):
        raise AdmissionRefused("dependency conflict")
    monkeypatch.setattr(plugins_cmd, "_set_plugin_enabled", refuse)

    refused = _install(inspected_sha=tip, enable=True)["error"]

    assert refused["code"] == 5026 and "enable refused" in refused["message"]
    assert refused["data"] == {"plugin_name": "demo"}
    assert (home / "plugins" / "demo").is_dir()


@pytest.mark.parametrize("refusal", ["unknown catalog entry", "removed source", "scan block"])
def test_a_refusal_before_publication_says_nothing_was_placed(moved_repo, monkeypatch, refusal):
    from hermes_cli import plugins_cmd, plugins_cmd_catalog
    params = {}
    if refusal == "unknown catalog entry":
        monkeypatch.setattr(plugins_cmd_catalog, "get_live_catalog_entry", lambda _name: None)
        params = {"catalog_name": "no-such-plugin"}
    elif refusal == "scan block":
        def blocked(*_args, **_kwargs):
            raise plugins_cmd.PluginScanBlocked("dangerous code")
        monkeypatch.setattr(plugins_cmd, "_scan_merged_tree", blocked)
    else:
        def removed(*_args, **_kwargs):
            raise plugins_cmd.PluginOperationError("removed from the Hermes plugin catalog")
        monkeypatch.setattr(plugins_cmd_catalog, "raise_if_removed", removed)

    refused = _install(force=True, **params)["error"]

    assert refused["code"] == 5026 and refused["data"] == {"placed": False}, refused


@pytest.mark.parametrize("fails", ["publication", "selection read"])
def test_a_failure_once_the_tree_went_to_pm_never_says_nothing_was_placed(moved_repo, monkeypatch, fails):
    """A forced replacement that fails during or after publication may have replaced the tree: only a refusal
    raised before it may tell the client nothing was placed (its desktop half then stays put)."""
    import pm.client
    from hermes_cli import plugins_cmd
    home, _reviewed, tip, _repo = moved_repo
    if fails == "publication":
        def broken(**_kwargs):
            raise RuntimeError("worker gone")
        monkeypatch.setattr(pm.client, "sync_venv", broken)
    else:
        def unreadable(*_args, **_kwargs):
            raise ValueError("config.yaml: plugins.enabled must be a list")
        monkeypatch.setattr(plugins_cmd, "_set_plugin_enabled", unreadable)

    refused = _install(inspected_sha=tip, enable=True, force=True)["error"]

    assert refused["code"] == 5026 and "placed" not in (refused.get("data") or {}), refused
    if fails == "selection read":
        assert (home / "plugins" / "demo").is_dir()


def test_reopening_the_dialog_after_a_refusal_reads_the_moved_source_again(moved_repo):
    _home, reviewed, tip, repo = moved_repo
    inspect = {"id": "1", "method": "plugins.manage",
               "params": {"action": "inspect", "identifier": "https://example.invalid/acme/demo.git"}}
    _git(repo, "update-ref", "refs/heads/main", reviewed)
    assert server.handle_request(inspect)["result"]["sha"] == reviewed
    _git(repo, "update-ref", "refs/heads/main", tip)  # the branch moves while the dialog is open
    assert "source changed" in _install(inspected_sha=reviewed)["error"]["message"]

    assert server.handle_request(inspect)["result"]["sha"] == tip


def test_install_at_the_inspected_tip_stays_an_unpinned_branch_checkout(moved_repo):
    home, _reviewed, tip, _repo = moved_repo

    result = _install(inspected_sha=tip)["result"]

    assert result["ok"] is True
    target = home / "plugins" / "demo"
    assert (target / "marker.txt").read_text() == "moved"
    record = json.loads((home / "plugins" / ".install-metadata.json").read_text())["demo"]
    assert (record["revision"], record["pinned"]) == (tip, False)
    # On the branch, not detached at a sha: a later `update` can still pull it forward.
    assert _git(target, "symbolic-ref", "--short", "HEAD") == "main"
