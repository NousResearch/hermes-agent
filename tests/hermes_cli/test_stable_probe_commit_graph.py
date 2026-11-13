"""A timed-out ancestry probe writes one changed-path-free commit-graph and retries locally
before falling back to the network (regression for #139091)."""
import subprocess

import pytest

from hermes_cli import source_check, source_releases


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def history(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    shas = []
    for n in range(3):
        (origin / "f").write_text(str(n))
        _git(origin, "add", "f")
        _git(origin, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", str(n))
        shas.append(_git(origin, "rev-parse", "HEAD"))
    return origin, shas  # shas[0] is the "release", HEAD (shas[2]) is ahead of it


def _first_probe_times_out(monkeypatch, subcommand):
    """Only the first ``git <subcommand>`` call times out; later calls run real git."""
    real_run = subprocess.run
    calls = []

    def run(cmd, *args, **kwargs):
        if cmd[:2] == ["git", subcommand]:
            calls.append(cmd)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(source_releases.subprocess, "run", run)
    return calls


def test_timed_out_probe_answers_locally_after_the_graph(history, monkeypatch):
    """The retry after the graph write answers ancestry locally; the network stays untouched."""
    origin, shas = history
    calls = _first_probe_times_out(monkeypatch, "merge-base")
    monkeypatch.setattr(source_check, "_github_compare", lambda *a, **k: pytest.fail("local retry answered"))
    # HEAD contains the release: the retried probe proves it and keeps HEAD without the network.
    assert source_releases._head_containing(["git"], origin, shas[0], "o/r") == shas[2]
    assert calls == [["git", "merge-base", "--is-ancestor", shas[0], shas[2]]] * 2
    # The self-heal is the changed-path-free graph write the #127711 guard still allows.
    assert (origin / ".git" / "objects" / "info" / "commit-graph").exists()


def test_unwritable_graph_keeps_the_relation_unknown(history, monkeypatch):
    """A graph that cannot be written leaves the unanswered probe exactly as unknown as before."""
    origin, shas = history
    real_run = subprocess.run

    def run(cmd, *args, **kwargs):
        if cmd[:3] == ["git", "commit-graph", "write"]:
            return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="unsupported")
        if cmd[:2] == ["git", "merge-base"]:
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(source_releases.subprocess, "run", run)
    monkeypatch.setattr(source_check, "_github_compare", lambda *a, **k: {"status": "behind"})
    assert source_releases._head_containing(["git"], origin, shas[2], "o/r") is None
