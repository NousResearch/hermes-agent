"""`hermes update` must preserve local commits when histories diverge.

When the checkout sits on the update target and its history has diverged, the
updater pins the pre-update HEAD under ``refs/hermes-update-backups/`` and
merges the fetched target. A merge conflict aborts and stops the update; local
work remains reachable and the recovery ref is retained. The installer update
paths are covered in ``tests/scripts/install/test_install_diverged_rescue_ref.py``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import update_cmd


GIT = ["git"]


def _git(repo, *args, check=True):
    return subprocess.run(
        GIT + list(args), cwd=repo, capture_output=True, text=True, check=check)


def _commit(repo, name, text):
    (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "commit", "-q", "-m", f"add {name}")
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture()
def diverged_checkout(tmp_path):
    """A checkout on ``main`` carrying a local commit its ``origin/main`` does not have."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q", "-b", "main")
    _commit(upstream, "shared.txt", "shared\n")
    _commit(upstream, "upstream-only.txt", "upstream\n")

    checkout = tmp_path / "checkout"
    _git(tmp_path, "clone", "-q", str(upstream), str(checkout))
    _git(checkout, "reset", "-q", "--hard", "HEAD~1")          # back to the shared commit
    local_sha = _commit(checkout, "local-fix.txt", "local\n")  # diverges from origin/main
    return checkout, local_sha


def _rescue_refs(checkout):
    out = _git(checkout, "for-each-ref", "--format=%(refname) %(objectname)",
               "refs/hermes-update-backups/").stdout
    return dict(line.split() for line in out.splitlines() if line.strip())


def _assert_merge_kept_local_commit(checkout, local_sha, output):
    head = _git(checkout, "rev-parse", "HEAD").stdout.strip()
    origin = _git(checkout, "rev-parse", "origin/main").stdout.strip()
    assert head != origin, "a divergent update must create a merge commit, not reset to origin"
    assert _git(checkout, "merge-base", "--is-ancestor", origin, head, check=False).returncode == 0
    assert _git(checkout, "merge-base", "--is-ancestor", local_sha, head, check=False).returncode == 0
    refs = [ref for ref, sha in _rescue_refs(checkout).items() if sha == local_sha]
    assert len(refs) == 1, f"the pre-merge commit needs exactly one rescue ref: {_rescue_refs(checkout)}"
    assert refs[0].startswith("refs/hermes-update-backups/diverged-main-")
    assert refs[0] in output, "the user must be told where the recovery ref went"


def test_hermes_update_merges_local_commit_and_keeps_a_recovery_ref(
        diverged_checkout, monkeypatch, capsys):
    """The real apply path: ff-only fails, then local and upstream commits are merged."""
    checkout, local_sha = diverged_checkout
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", checkout)

    update_cmd._pull_updates(
        GIT, "main", None, prompt_for_restore=False, gw_input_fn=None,
        discard_local_changes=False, keep_stash=False)

    out = capsys.readouterr().out
    _assert_merge_kept_local_commit(checkout, local_sha, out)
    assert "merging the fetched target" in out
    assert "resetting to match remote" not in out
    assert "git reset --hard" not in out


def test_hermes_update_stops_on_merge_conflict_and_retains_recovery_ref(
        diverged_checkout, monkeypatch, capsys):
    """A conflicting same-branch merge aborts without replacing local code."""
    checkout, local_sha = diverged_checkout
    upstream = checkout.parent / "upstream"
    (checkout / "shared.txt").write_text("local change\n", encoding="utf-8")
    local_sha = _commit(checkout, "shared.txt", "local change\n")
    (upstream / "shared.txt").write_text("upstream change\n", encoding="utf-8")
    _git(upstream, "add", "shared.txt")
    _git(upstream, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "commit", "-q", "-m", "conflicting upstream")
    upstream_sha = _git(upstream, "rev-parse", "HEAD").stdout.strip()
    # Fetch the newly-created upstream commit so the checkout owns the object before
    # moving origin/main; a ref update alone is invalid in a freshly-cloned repo.
    _git(checkout, "fetch", "-q", "origin", "main")
    assert _git(checkout, "rev-parse", "origin/main").stdout.strip() == upstream_sha
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", checkout)

    with pytest.raises(SystemExit) as exc:
        update_cmd._pull_updates(
            GIT, "main", None, prompt_for_restore=False, gw_input_fn=None,
            discard_local_changes=False, keep_stash=False)

    assert exc.value.code == 1
    assert _git(checkout, "rev-parse", "HEAD").stdout.strip() == local_sha
    assert _git(checkout, "merge-base", "--is-ancestor", upstream_sha, local_sha,
                check=False).returncode == 1
    assert _git(checkout, "diff", "--name-only").stdout.strip() == ""
    assert (checkout / "shared.txt").read_text(encoding="utf-8") == "local change\n"
    assert not (checkout / ".git" / "MERGE_HEAD").exists()
    refs = _rescue_refs(checkout)
    assert any(sha == local_sha for sha in refs.values())
    out = capsys.readouterr().out
    assert "Merge conflict" in out
    assert "Recovery ref retained" in out


@pytest.mark.parametrize(
    "abort_failure",
    [
        "returncode",
        "exception",
        "probe_returncode",
        "probe_exception",
        "probe_path_exception",
        "probe_malformed_marker",
    ],
)
def test_hermes_update_stops_when_merge_abort_is_unproven_and_keeps_conflict_state(
        diverged_checkout, monkeypatch, capsys, abort_failure):
    """An abort error or unreadable merge state must not be reported as clean rollback."""
    checkout, local_sha = diverged_checkout
    upstream = checkout.parent / "upstream"
    (checkout / "shared.txt").write_text("local change\n", encoding="utf-8")
    local_sha = _commit(checkout, "shared.txt", "local change\n")
    (upstream / "shared.txt").write_text("upstream change\n", encoding="utf-8")
    _git(upstream, "add", "shared.txt")
    _git(upstream, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "commit", "-q", "-m", "conflicting upstream")
    upstream_sha = _git(upstream, "rev-parse", "HEAD").stdout.strip()
    _git(checkout, "fetch", "-q", "origin", "main")
    assert _git(checkout, "rev-parse", "origin/main").stdout.strip() == upstream_sha
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", checkout)
    real_git_run = update_cmd._git_run
    stop_reasons = []
    monkeypatch.setattr(update_cmd, "_record_stop", lambda reason: stop_reasons.append(reason))

    def fail_abort(git_cmd, args, *rest, **kwargs):
        if args == ["merge", "--abort"]:
            if abort_failure == "exception":
                raise OSError("fixture abort refusal")
            if abort_failure == "returncode":
                return subprocess.CompletedProcess(git_cmd + args, 1, stdout="", stderr="fixture abort refusal")
            # Probe-failure fixtures leave the real conflict state untouched while
            # reporting a successful abort, so the subsequent state probe is the
            # only thing deciding whether recovery was proven.
            return subprocess.CompletedProcess(git_cmd + args, 0, stdout="", stderr="")
        if abort_failure == "probe_exception" and args == ["rev-parse", "-q", "--verify", "MERGE_HEAD"]:
            raise OSError("fixture merge-state refusal")
        if abort_failure == "probe_returncode" and args == ["rev-parse", "-q", "--verify", "MERGE_HEAD"]:
            return subprocess.CompletedProcess(
                git_cmd + args, 128, stdout="", stderr="fixture merge-state refusal")
        if abort_failure == "probe_path_exception" and args == ["rev-parse", "--git-path", "MERGE_HEAD"]:
            raise RuntimeError("fixture git-path refusal")
        if abort_failure == "probe_malformed_marker" and args == ["merge", "--abort"]:
            marker = Path(_git(checkout, "rev-parse", "--git-path", "MERGE_HEAD").stdout.strip())
            if not marker.is_absolute():
                marker = checkout / marker
            marker.write_text("not-a-valid-commit\n", encoding="utf-8")
            return subprocess.CompletedProcess(git_cmd + args, 0, stdout="", stderr="")
        return real_git_run(git_cmd, args, *rest, **kwargs)

    monkeypatch.setattr(update_cmd, "_git_run", fail_abort)
    with pytest.raises(SystemExit) as exc:
        update_cmd._pull_updates(
            GIT, "main", None, prompt_for_restore=False, gw_input_fn=None,
            discard_local_changes=False, keep_stash=False)

    assert exc.value.code == 1
    assert _git(checkout, "rev-parse", "HEAD").stdout.strip() == local_sha
    assert (checkout / ".git" / "MERGE_HEAD").exists()
    assert any(sha == local_sha for sha in _rescue_refs(checkout).values())
    out = capsys.readouterr().out
    assert "could not be safely aborted" in out
    assert "merge_abort_failed" not in out  # stop token is recorded, not printed
    assert stop_reasons == ["merge_abort_failed"]


def test_hermes_update_protects_current_rescue_ref_during_retention_burst(tmp_path):
    """The just-created ref survives same-second lexical pruning, even above the keep count."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
         "commit", "--allow-empty", "-qm", "base")
    _git(tmp_path, "branch", "-M", "main")
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = "refs/hermes-update-backups/orphan-main-"
    candidates = [prefix + stamp + f"-{index:03d}" for index in range(11)]
    protected = prefix + stamp + "-current"
    for ref in [*candidates, protected]:
        _git(tmp_path, "update-ref", ref, "HEAD")

    update_cmd._prune_orphan_rescue_refs(
        ["git"], tmp_path, "main", keep=1, protect_ref=protected)
    remaining = set(_git(
        tmp_path, "for-each-ref", "--format=%(refname)", "refs/hermes-update-backups/"
    ).stdout.splitlines())
    assert remaining == {candidates[-1], protected}


@pytest.fixture()
def fast_forward_checkout(tmp_path):
    """A checkout whose HEAD is strictly behind origin/main."""
    upstream = tmp_path / "linear-upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q", "-b", "main")
    first = _commit(upstream, "shared.txt", "shared\n")
    _commit(upstream, "upstream-only.txt", "upstream\n")

    checkout = tmp_path / "linear-checkout"
    _git(tmp_path, "clone", "-q", str(upstream), str(checkout))
    _git(checkout, "reset", "-q", "--hard", first)
    return checkout, first


def test_live_index_lock_is_reported_without_false_divergence(
        fast_forward_checkout, monkeypatch, capsys):
    checkout, before = fast_forward_checkout
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", checkout)
    (checkout / ".git" / "index.lock").touch()

    with pytest.raises(SystemExit) as exc:
        update_cmd._pull_updates(
            GIT, "main", None, prompt_for_restore=False, gw_input_fn=None,
            discard_local_changes=False, keep_stash=False)

    assert exc.value.code == 1
    assert _git(checkout, "rev-parse", "HEAD").stdout.strip() == before
    assert _rescue_refs(checkout) == {}
    out = capsys.readouterr().out
    assert "index.lock" in out
    assert "HEAD is still an ancestor of origin/main" in out
    assert "Local history has diverged" not in out
    assert "Fast-forward not possible (history diverged)" not in out
    assert "git reset --hard" not in out


def test_operational_ff_failure_preserves_git_error_without_reset(
        fast_forward_checkout, monkeypatch, capsys):
    checkout, before = fast_forward_checkout
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", checkout)
    real_git_run = update_cmd._git_run

    def fail_merge(git_cmd, args, *rest, **kwargs):
        if args[:2] == ["merge", "--ff-only"]:
            return subprocess.CompletedProcess(
                git_cmd + args, 128, stdout="",
                stderr="fatal: unable to read tree: RPC failed; transfer closed")
        return real_git_run(git_cmd, args, *rest, **kwargs)

    monkeypatch.setattr(update_cmd, "_git_run", fail_merge)

    with pytest.raises(SystemExit) as exc:
        update_cmd._pull_updates(
            GIT, "main", None, prompt_for_restore=False, gw_input_fn=None,
            discard_local_changes=False, keep_stash=False)

    assert exc.value.code == 1
    assert _git(checkout, "rev-parse", "HEAD").stdout.strip() == before
    assert _rescue_refs(checkout) == {}
    out = capsys.readouterr().out
    assert "RPC failed; transfer closed" in out
    assert "HEAD is still an ancestor of origin/main" in out
    assert "Local history has diverged" not in out
    assert "git reset --hard" not in out
