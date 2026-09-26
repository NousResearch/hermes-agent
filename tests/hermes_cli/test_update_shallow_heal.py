"""Heal-before-fetch for stale shallow installer checkouts (#123254).

A depth-1 install far behind ``origin/main`` cannot finish ``git fetch origin
main`` inside the updater's 300s network cap: the server must send the full
ancestry of every side branch merged past the shallow boundary, every attempt
is killed mid-transfer, and the unshallow step that would make later fetches
incremental only ran *after* a successful update — so a stale shallow install
could never update at all. The updater now heals the checkout before the
bounded fetch, non-fatally, and the timeout wording no longer blames a dead
remote for a wall-clock kill on a healthy transfer.

These run against real local git repositories; only the timing caps are
irrelevant here (the fetches are local and instant).
"""

from __future__ import annotations

import subprocess

import hermes_cli.gitlock as gitlock
import hermes_cli.update_cmd as update_cmd


def _git(cwd, *args, check=True):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def _origin_with_history(tmp_path, *, allow_filter=True):
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "t@example.com")
    _git(origin, "config", "user.name", "t")
    _git(origin, "commit", "--allow-empty", "-qm", "c0")
    _git(origin, "commit", "--allow-empty", "-qm", "c1")
    _git(origin, "tag", "v0.21.2")
    if allow_filter:
        # file:// transport requires opt-in; real hosting does this by default.
        _git(origin, "config", "uploadpack.allowFilter", "true")
    return origin


def _shallow_clone(tmp_path, origin):
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", f"file://{origin}", str(clone)],
        check=True,
        capture_output=True,
        text=True,
    )
    return clone


class TestHealShallowHistory:
    def test_heals_stale_shallow_clone_and_returns_true(self, tmp_path):
        origin = _origin_with_history(tmp_path)
        clone = _shallow_clone(tmp_path, origin)
        assert (clone / ".git" / "shallow").exists()

        assert gitlock.heal_shallow_history(clone, "main") is True

        assert not (clone / ".git" / "shallow").exists()
        # The history behind the boundary is present, so the bounded fetch
        # that follows is incremental instead of re-downloading the graph.
        assert _git(clone, "rev-list", "--count", "--all").stdout.strip() == "2"

    def test_full_clone_is_a_noop_returning_false(self, tmp_path):
        origin = _origin_with_history(tmp_path)
        clone = tmp_path / "clone"
        subprocess.run(
            ["git", "clone", "-q", f"file://{origin}", str(clone)],
            check=True,
            capture_output=True,
            text=True,
        )

        assert gitlock.heal_shallow_history(clone, "main") is False
        assert _git(clone, "rev-list", "--count", "--all").stdout.strip() == "2"

    def test_origin_without_tags_still_unshallows(self, tmp_path):
        # The tag refspec alone matches nothing on a tag-less origin and git
        # fails the whole fetch; the branch refspec must ride along.
        origin = _origin_with_history(tmp_path)
        _git(origin, "tag", "-d", "v0.21.2")
        clone = _shallow_clone(tmp_path, origin)

        assert gitlock.heal_shallow_history(clone, "main") is True
        assert not (clone / ".git" / "shallow").exists()


class TestUpdateHealIsNonFatal:
    def test_heal_failure_keeps_the_update_alive(self, tmp_path, monkeypatch, capsys):
        origin = _origin_with_history(tmp_path)
        clone = _shallow_clone(tmp_path, origin)

        def broken_heal(*a, **kw):
            raise subprocess.CalledProcessError(
                128, "git", stderr="fatal: filter not supported"
            )

        monkeypatch.setattr(gitlock, "heal_shallow_history", broken_heal)
        # Must not raise: the update proceeds with the old fetch behaviour.
        update_cmd._heal_stale_shallow_checkout(clone, "main")
        out = capsys.readouterr().out
        assert "Could not heal the shallow checkout" in out
        assert "filter not supported" in out

    def test_healed_checkout_prints_one_line(self, tmp_path, capsys):
        origin = _origin_with_history(tmp_path)
        clone = _shallow_clone(tmp_path, origin)

        update_cmd._heal_stale_shallow_checkout(clone, "main")

        assert "shallow checkout healed" in capsys.readouterr().out
        assert not (clone / ".git" / "shallow").exists()

    def test_full_checkout_stays_silent(self, tmp_path, capsys):
        origin = _origin_with_history(tmp_path)
        clone = tmp_path / "clone"
        subprocess.run(
            ["git", "clone", "-q", f"file://{origin}", str(clone)],
            check=True,
            capture_output=True,
            text=True,
        )

        update_cmd._heal_stale_shallow_checkout(clone, "main")

        assert capsys.readouterr().out == ""


class TestTimeoutWording:
    def test_wall_clock_kill_does_not_blame_a_dead_remote(self, monkeypatch):
        # A wall-clock kill says nothing about the transport: the fetch may
        # have been actively transferring a multi-hundred-MB pack (#123254).
        def never_finishes(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, update_cmd.NETWORK_GIT_TIMEOUT_SECONDS)

        monkeypatch.setattr(update_cmd.subprocess, "run", never_finishes)
        result = update_cmd._git_run(
            ["git"], ["fetch", "origin", "main"], cwd="/tmp", network=True
        )
        assert result.returncode == 124
        assert "no response from the remote" not in result.stderr
        assert "network limit" in result.stderr
        assert "manual `git fetch`" in result.stderr
