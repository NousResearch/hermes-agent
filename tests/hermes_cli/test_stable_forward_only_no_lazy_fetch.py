"""A partial clone on git < 2.44 must not run a local ancestry probe that lazy-fetches.

Git < 2.44 ignores ``GIT_NO_LAZY_FETCH`` (see ``NO_LAZY_FETCH_ENV``), and in a partial
clone *every* local object lookup -- ``merge-base --is-ancestor``, ``cat-file -e``,
``rev-parse --verify``, ``rev-list --missing`` -- downloads the missing history from the
promisor remote. The probe's 10s timeout then kills git mid-fetch and no update is
applied. ``_head_containing`` must decide such a checkout without that probe.
"""
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import source_check, source_releases

_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def _git(*args, cwd=None):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          cwd=cwd, env=_ENV, check=True, capture_output=True, text=True).stdout.strip()


def _packs(clone):
    """Object packs present, read without git: a promisor lazy fetch lands a new one."""
    return sorted(p.name for p in (Path(clone) / ".git" / "objects" / "pack").glob("*.pack"))


@pytest.fixture
def partial_clone(tmp_path):
    """(clone, local_head, unfetched_upstream_tip) for a blobless clone of a local upstream."""
    seed, up, clone = tmp_path / "seed", tmp_path / "up.git", tmp_path / "clone"
    _git("init", "-q", "-b", "main", str(seed))
    for i in range(2):
        (seed / f"f{i}.txt").write_text(f"v{i}\n" * 50, encoding="utf-8")
        _git("add", "-A", cwd=seed)
        _git("commit", "-qm", f"c{i}", cwd=seed)
    _git("clone", "-q", "--bare", str(seed), str(up))
    _git("config", "uploadpack.allowFilter", "true", cwd=up)
    _git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=up)
    _git("clone", "-q", "--filter=blob:none", "--no-checkout", up.as_uri(), str(clone))
    (seed / "new.txt").write_text("upstream moved\n", encoding="utf-8")
    _git("add", "-A", cwd=seed)
    _git("commit", "-qm", "upstream", cwd=seed)
    _git("push", "-q", str(up), "main", cwd=seed)
    return clone, _git("rev-parse", "HEAD", cwd=clone), _git("rev-parse", "main", cwd=up)


def test_gate_flags_partial_clone_on_old_git_only():
    def run_for(version, promisor):
        def run(*args):
            return SimpleNamespace(stdout=promisor if "remote.origin.promisor" in args else version)
        return run

    # Git < 2.44 ignores the variable: a partial clone's local probe lazy-fetches.
    assert source_releases._ancestry_probe_may_lazy_fetch(run_for("git version 2.43.0\n", "true\n")) is True
    # Git >= 2.44 honors it, so the probe fails fast on a missing object.
    assert source_releases._ancestry_probe_may_lazy_fetch(run_for("git version 2.44.0\n", "true\n")) is False
    # No promisor remote (a full clone) can lazy-fetch nothing.
    assert source_releases._ancestry_probe_may_lazy_fetch(run_for("git version 2.43.0\n", "")) is False


def test_partial_clone_on_old_git_never_runs_the_local_probe(partial_clone, monkeypatch):
    clone, head, upstream_tip = partial_clone
    # Force "git earlier than 2.44" so the gate is exercised on any host's git version.
    monkeypatch.setattr(source_releases, "_NO_LAZY_FETCH_MIN_GIT", (99, 0))
    monkeypatch.setattr(source_check, "_github_compare", lambda *a, **k: {"status": "ahead"})
    commands = []
    real = subprocess.run

    def spy(cmd, *args, **kwargs):
        commands.append(list(cmd))
        return real(cmd, *args, **kwargs)

    monkeypatch.setattr(source_releases.subprocess, "run", spy)
    before = _packs(clone)
    assert source_releases._head_containing(["git"], clone, upstream_tip, "o/r") == head
    # The probe that would download the missing history never ran...
    assert not any("merge-base" in cmd for cmd in commands)
    # ...so no pack was fetched from the promisor remote.
    assert _packs(clone) == before
    # GitHub proving HEAD is behind is still the only case that drops HEAD.
    monkeypatch.setattr(source_check, "_github_compare", lambda *a, **k: {"status": "behind"})
    assert source_releases._head_containing(["git"], clone, upstream_tip, "o/r") is None
    assert _packs(clone) == before


def test_git_2_44_keeps_the_local_probe(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    origin.mkdir()
    _git("init", "-q", "-b", "main", str(origin))
    shas = []
    for n in range(3):
        (origin / "f").write_text(str(n))
        _git("add", "f", cwd=origin)
        _git("commit", "-qm", str(n), cwd=origin)
        shas.append(_git("rev-parse", "HEAD", cwd=origin))
    clone = tmp_path / "full"
    _git("clone", "-q", str(origin), str(clone))
    _git("checkout", "-q", shas[2], cwd=clone)

    monkeypatch.setattr(source_releases, "_ancestry_probe_may_lazy_fetch", lambda run: False)
    monkeypatch.setattr(source_check, "_github_compare", lambda *a, **k: pytest.fail("a present object needs no network"))
    commands = []
    real = subprocess.run

    def spy(cmd, *args, **kwargs):
        commands.append(list(cmd))
        return real(cmd, *args, **kwargs)

    monkeypatch.setattr(source_releases.subprocess, "run", spy)
    assert source_releases._head_containing(["git"], clone, shas[0], "o/r") == shas[2]
    assert any("merge-base" in cmd for cmd in commands)
