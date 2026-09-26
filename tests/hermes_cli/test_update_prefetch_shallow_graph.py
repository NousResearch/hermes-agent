"""Shallow installs prefetch the commit graph before the update's branch fetch (#123254).

A plain ``git fetch origin <branch>`` from a depth-1 clone that fell behind makes the
server send nearly the whole object graph (merge-heavy history: every side branch
merged since the boundary drags its full ancestry behind it), which the 300 s
network cap kills on every attempt — the post-update unshallow step never gets to
run, so the install is stuck. The update path now unshallows to a commit-only
graph first, making the branch fetch incremental.

These run ``_prefetch_shallow_commit_graph`` against real git checkouts (a depth-1
clone of a local origin); only the failure and already-unshallowed cases fake the
commit-graph fetch.
"""

from __future__ import annotations

import subprocess

import hermes_cli.main as cli_main
from hermes_cli import gitlock, update_cmd


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _origin(tmp_path, monkeypatch):
    for key, value in {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
        "GIT_CONFIG_NOSYSTEM": "1",
    }.items():
        monkeypatch.setenv(key, value)
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "uploadpack.allowFilter", "true")
    for n in range(1, 4):
        (origin / "f.txt").write_text(f"{n}\n", encoding="utf-8")
        _git(origin, "add", "f.txt")
        _git(origin, "commit", "-q", "-m", f"c{n}")
    _git(origin, "tag", "v0.21.4")
    return origin


def _shallow_checkout(tmp_path, origin, monkeypatch):
    clone = tmp_path / "checkout"
    _git(tmp_path, "clone", "-q", "--depth", "1", origin.as_uri(), str(clone))
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", clone)
    return clone


def _is_shallow(repo) -> str:
    return subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"], cwd=repo,
        check=True, capture_output=True, text=True).stdout.strip()


def test_shallow_checkout_prefetches_the_commit_graph_and_unshallows(tmp_path, monkeypatch, capsys):
    origin = _origin(tmp_path, monkeypatch)
    checkout = _shallow_checkout(tmp_path, origin, monkeypatch)
    assert _is_shallow(checkout) == "true"

    update_cmd._prefetch_shallow_commit_graph(["git"])

    assert "Prefetched the shallow commit graph" in capsys.readouterr().out
    assert _is_shallow(checkout) == "false"


def test_full_checkout_skips_the_prefetch(tmp_path, monkeypatch, capsys):
    origin = _origin(tmp_path, monkeypatch)
    checkout = tmp_path / "checkout"
    _git(tmp_path, "clone", "-q", origin.as_uri(), str(checkout))
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", checkout)

    def fail(*_args, **_kwargs):
        raise AssertionError("full checkouts must not run the commit-graph prefetch")

    monkeypatch.setattr(gitlock, "fetch_full_commit_graph", fail)
    update_cmd._prefetch_shallow_commit_graph(["git"])

    assert "Prefetched the shallow commit graph" not in capsys.readouterr().out


def test_prefetch_failure_falls_through_to_the_plain_fetch(tmp_path, monkeypatch, capsys):
    origin = _origin(tmp_path, monkeypatch)
    checkout = _shallow_checkout(tmp_path, origin, monkeypatch)

    def stall(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["git", "fetch"], 900)

    monkeypatch.setattr(gitlock, "fetch_full_commit_graph", stall)
    update_cmd._prefetch_shallow_commit_graph(["git"])  # must not raise

    out = capsys.readouterr().out
    assert "Could not prefetch the shallow commit graph" in out
    assert "continuing with the plain fetch" in out
    assert _is_shallow(checkout) == "true"


def test_prefetch_reports_nothing_when_the_checkout_was_already_unshallowed(tmp_path, monkeypatch, capsys):
    origin = _origin(tmp_path, monkeypatch)
    _shallow_checkout(tmp_path, origin, monkeypatch)

    monkeypatch.setattr(gitlock, "fetch_full_commit_graph", lambda *_args, **_kwargs: False)
    update_cmd._prefetch_shallow_commit_graph(["git"])

    assert "Prefetched the shallow commit graph" not in capsys.readouterr().out
