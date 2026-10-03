"""`hermes update` migrates a treeless (tree:0) install to a blobless clone (#129514).

A tree:0 clone stores no trees at all, so every path-filtered walk lazy-fetches one
tree per commit from the promisor remote, and each on-demand fetch writes its own
promisor pack and schedules maintenance — the unbounded storm that filled 434 GB.
`migrate_treeless_checkout` flips the filter to `blob:none` and refetches the trees
once, so path-filtered walks answer locally. A full clone and a checkout carrying any
other filter must stay untouched (#122353: repeat the clone's own filter, never
re-arm or silently change one).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli.gitlock import migrate_treeless_checkout

_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def _git(*args, cwd=None, env=_ENV, check=True):
    result = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t.invalid", *args],
        cwd=cwd, env=env, capture_output=True, text=True)
    if check and result.returncode:
        raise AssertionError(f"git {args} failed: {result.stderr}")
    return result.stdout.strip()


@pytest.fixture
def upstream(tmp_path: Path) -> Path:
    seed, up = tmp_path / "seed", tmp_path / "up.git"
    _git("init", "-q", "-b", "main", str(seed))
    for index in range(5):
        (seed / "history.txt").write_text(f"commit {index}\n", encoding="utf-8")
        _git("add", "-A", cwd=seed)
        _git("commit", "-qm", f"history {index}", cwd=seed)
    _git("clone", "-q", "--bare", str(seed), str(up))
    _git("config", "uploadpack.allowFilter", "true", cwd=up)
    _git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=up)
    _git("symbolic-ref", "HEAD", "refs/heads/main", cwd=up)
    return up


def _clone(up: Path, tmp_path: Path, name: str, *clone_args: str) -> Path:
    checkout = tmp_path / name
    _git("clone", "-q", *clone_args, up.as_uri(), str(checkout))
    return checkout


def _filter_of(checkout: Path) -> str | None:
    result = _git("config", "--get", "remote.origin.partialclonefilter", cwd=checkout, check=False)
    return result or None


def _path_probe_succeeds_locally(checkout: Path) -> bool:
    """A path-filtered rev-list answers without lazy fetches (trees are local)."""
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-list", "HEAD", "--", "history.txt"],
        env={**_ENV, "GIT_NO_LAZY_FETCH": "1"}, capture_output=True, text=True)
    return result.returncode == 0


def test_treeless_clone_migrates_to_blobless_and_answers_paths_locally(upstream, tmp_path):
    checkout = _clone(upstream, tmp_path, "treeless", "--filter=tree:0")
    assert _filter_of(checkout) == "tree:0"
    # The bug this guards: on the treeless clone the probe fails hard with lazy
    # fetches forbidden, because every tree it would walk is missing (#129514).
    assert not _path_probe_succeeds_locally(checkout)

    assert migrate_treeless_checkout(checkout, "main") is True

    assert _filter_of(checkout) == "blob:none"
    assert _path_probe_succeeds_locally(checkout)
    # Promisor packs stay marked, so later fetches don't hit the git 2.53+ crash (#124272).
    packs = list((checkout / ".git" / "objects" / "pack").glob("pack-*.pack"))
    assert all(p.with_suffix(".promisor").exists() for p in packs)


def test_migration_is_idempotent_after_the_filter_flipped(upstream, tmp_path):
    checkout = _clone(upstream, tmp_path, "treeless", "--filter=tree:0")
    # No branch given: the checked-out branch is refetched.
    assert migrate_treeless_checkout(checkout) is True
    # The second run sees blob:none (not tree:0) and does nothing.
    assert migrate_treeless_checkout(checkout) is False
    assert _filter_of(checkout) == "blob:none"


def test_a_failed_refetch_is_retried_by_the_next_run(upstream, tmp_path):
    checkout = _clone(upstream, tmp_path, "treeless", "--filter=tree:0")
    real_url = _git("remote", "get-url", "origin", cwd=checkout)
    _git("remote", "set-url", "origin", (tmp_path / "gone.git").as_uri(), cwd=checkout)
    assert migrate_treeless_checkout(checkout, "main") is True
    # The filter already flipped, but the trees never arrived.
    assert _filter_of(checkout) == "blob:none"
    assert not _path_probe_succeeds_locally(checkout)

    _git("remote", "set-url", "origin", real_url, cwd=checkout)
    assert migrate_treeless_checkout(checkout, "main") is True
    assert _path_probe_succeeds_locally(checkout)
    assert migrate_treeless_checkout(checkout, "main") is False


def test_the_update_bootstrap_interpreter_completes_the_refetch(upstream, tmp_path):
    # Both update hand-offs (update_completion._prepare, _update_takeover.prepare) call this
    # from an isolated interpreter with no site-packages, before the dependency sync. A
    # refetch that needs a third-party module there never runs: the filter flips but the
    # trees never arrive, which is the per-tree storm in a new shape.
    checkout = _clone(upstream, tmp_path, "treeless", "--filter=tree:0")
    root = Path(__file__).resolve().parents[2]
    script = ("import sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path; "
              "from hermes_cli.gitlock import migrate_treeless_checkout; "
              "print(migrate_treeless_checkout(Path(sys.argv[2])))")
    result = subprocess.run([sys.executable, "-I", "-S", "-c", script, str(root), str(checkout)],
                            env=_ENV, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.stdout.strip() == "True", result.stderr
    assert _filter_of(checkout) == "blob:none"
    assert _path_probe_succeeds_locally(checkout)
    assert migrate_treeless_checkout(checkout, "main") is False


def test_full_clone_is_untouched(upstream, tmp_path):
    checkout = _clone(upstream, tmp_path, "full")
    assert _filter_of(checkout) is None
    assert migrate_treeless_checkout(checkout, "main") is False
    assert _filter_of(checkout) is None
    # The promisor keys must not appear (#122353: a fetch never re-arms them).
    assert _git("config", "--get", "remote.origin.promisor", cwd=checkout, check=False) == ""


def test_user_chosen_blobless_filter_is_untouched(upstream, tmp_path):
    checkout = _clone(upstream, tmp_path, "blobless", "--filter=blob:none")
    assert _filter_of(checkout) == "blob:none"
    assert migrate_treeless_checkout(checkout, "main") is False
    assert _filter_of(checkout) == "blob:none"


def test_maintenance_guards_stay_after_migration(upstream, tmp_path):
    checkout = _clone(upstream, tmp_path, "treeless", "--filter=tree:0")
    assert migrate_treeless_checkout(checkout, "main") is True
    for key in ("maintenance.commit-graph.enabled", "gc.writeCommitGraph", "fetch.writeCommitGraph"):
        assert _git("config", "--get", key, cwd=checkout) == "false"
    # Automatic maintenance itself stays ON: its post-fetch gc --auto is what
    # folds lazy-fetch packs between updates (gitlock's own contract).
    assert _git("config", "--get", "maintenance.auto", cwd=checkout, check=False) == ""
