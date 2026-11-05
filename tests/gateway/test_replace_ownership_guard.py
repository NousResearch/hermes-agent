"""Tests for issue #89315 — ``--replace`` must never signal a gateway it
cannot prove belongs to this HERMES_HOME.

Design contract (v3, after andrexibiza's second review): ownership is decided
by the persisted identity record ALONE — exact ``_same_hermes_home`` equality
bound to the live target by exact PID + start-time. A readable live argv
carries no HERMES_HOME, so it can never prove home ownership; it only feeds a
token-exact CONSISTENCY check that refuses explicit contradictions.

Pinned surfaces:

* record authority — valid+bound same-home allows; missing/legacy/unbound/
  foreign records refuse;
* argv consistency — token-exact: ``--profile timothy`` must NOT read as
  ``tim`` (the substring heuristic's false-allow), while an exact different
  profile flag contradicts and refuses;
* signal boundary — ``start_gateway(replace=True)`` on unprovable ownership
  refuses without calling ``terminate_pid`` or writing a takeover marker; the
  legitimate bound same-home target still reaches the replace flow;
* retained stamp (#135392) — the unprovable-ownership refusal stamps
  ``gateway_state: startup_failed`` (with an ``exit_reason`` code) so the
  stale PID record stops being trusted (``get_runtime_status_running_pid()``
  returns None for startup_failed) and /api/status shows reality, while the
  exit stays 1 so a supervisor's next attempt can win.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME mirroring tests/hermes_cli/test_profiles.py."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_home if (tmp_home := default_home) else default_home


def _record(pid=424242, start=111222333, home=None, argv=None):
    return {
        "pid": pid,
        "kind": "hermes-gateway",
        "argv": argv or ["python", "-m", "hermes_cli.main", "gateway", "run"],
        "start_time": start,
        "hermes_home": home,
    }


class TestRecordAuthority:
    def test_valid_bound_same_home_record_allows(self, profile_env):
        from gateway.run import _replace_target_belongs_to_other_profile

        with (
            patch(
                "gateway.status._read_pid_record",
                return_value=_record(home=str(profile_env / ".hermes")),
            ),
            patch(
                "gateway.status._get_pid_path",
                return_value=profile_env / ".hermes" / "gateway.pid",
            ),
            patch(
                "gateway.status._get_process_start_time",
                return_value=111222333,
            ),
            patch("gateway.status._read_process_cmdline", return_value=None),
            patch(
                "gateway.status._get_process_hermes_home",
                return_value=profile_env / ".hermes",
            ),
        ):
            assert _replace_target_belongs_to_other_profile(424242) is False

    def test_foreign_home_record_refuses(self, profile_env):
        """Exact-home equality: another root/profile in the record refuses,
        even with a bare argv that substring matching would have passed."""
        from gateway.run import _replace_target_belongs_to_other_profile

        with (
            patch(
                "gateway.status._read_pid_record",
                return_value=_record(
                    home="/home/other/.hermes/profiles/timothy"
                ),
            ),
            patch(
                "gateway.status._get_pid_path",
                return_value=profile_env / ".hermes" / "gateway.pid",
            ),
            patch(
                "gateway.status._get_process_start_time",
                return_value=111222333,
            ),
            patch("gateway.status._read_process_cmdline", return_value=None),
            patch(
                "gateway.status._get_process_hermes_home",
                return_value=profile_env / ".hermes" / "profiles" / "tim",
            ),
        ):
            assert _replace_target_belongs_to_other_profile(424242) is True

    def test_missing_record_refuses(self, profile_env):
        """No valid record → ownership unprovable → refuse."""
        from gateway.run import _replace_target_belongs_to_other_profile

        with (
            patch("gateway.status._read_pid_record", return_value=None),
            patch(
                "gateway.status._get_pid_path",
                return_value=profile_env / ".hermes" / "gateway.pid",
            ),
            patch(
                "gateway.status._get_process_hermes_home",
                return_value=profile_env / ".hermes",
            ),
        ):
            assert _replace_target_belongs_to_other_profile(424242) is True

    def test_legacy_record_without_home_refuses(self, profile_env):
        """A pre-hermes_home-stamping record cannot prove ownership."""
        from gateway.run import _replace_target_belongs_to_other_profile

        legacy = _record(home=None)
        legacy.pop("hermes_home")

        with (
            patch(
                "gateway.status._read_pid_record",
                return_value=legacy,
            ),
            patch(
                "gateway.status._get_pid_path",
                return_value=profile_env / ".hermes" / "gateway.pid",
            ),
            patch(
                "gateway.status._get_process_start_time",
                return_value=111222333,
            ),
            patch(
                "gateway.status._get_process_hermes_home",
                return_value=profile_env / ".hermes",
            ),
        ):
            assert _replace_target_belongs_to_other_profile(424242) is True

    def test_unbound_record_wrong_pid_refuses(self, profile_env):
        """A record naming a DIFFERENT pid proves nothing (poisoned shape)."""
        from gateway.run import _replace_target_belongs_to_other_profile

        with (
            patch(
                "gateway.status._read_pid_record",
                return_value=_record(pid=999999, home=str(profile_env / ".hermes")),
            ),
            patch(
                "gateway.status._get_pid_path",
                return_value=profile_env / ".hermes" / "gateway.pid",
            ),
            patch(
                "gateway.status._get_process_start_time",
                return_value=111222333,
            ),
            patch(
                "gateway.status._get_process_hermes_home",
                return_value=profile_env / ".hermes",
            ),
        ):
            assert _replace_target_belongs_to_other_profile(424242) is True

    def test_unbound_record_stale_start_time_refuses(self, profile_env):
        """PID reused since the record was written (start_time drift) → the
        record no longer describes the live process → refuse."""
        from gateway.run import _replace_target_belongs_to_other_profile

        with (
            patch(
                "gateway.status._read_pid_record",
                return_value=_record(start=1, home=str(profile_env / ".hermes")),
            ),
            patch(
                "gateway.status._get_pid_path",
                return_value=profile_env / ".hermes" / "gateway.pid",
            ),
            patch(
                "gateway.status._get_process_start_time",
                return_value=42,
            ),
            patch(
                "gateway.status._get_process_hermes_home",
                return_value=profile_env / ".hermes",
            ),
        ):
            assert _replace_target_belongs_to_other_profile(424242) is True

    def test_probe_exception_fails_closed(self, profile_env):
        from gateway.run import _replace_target_belongs_to_other_profile

        with patch(
            "gateway.status._read_pid_record",
            side_effect=RuntimeError("probe exploded"),
        ):
            assert _replace_target_belongs_to_other_profile(424242) is True


class TestArgvConsistencyCheck:
    """Readable argv is a consistency check ONLY — never authority."""

    def test_prefix_collision_is_not_a_conflict(self, profile_env):
        """``--profile timothy`` must NOT read as our profile ``tim``:
        substring matching would have false-allowed the foreign gateway."""
        from gateway.run import (
            _looks_like_profile_conflict_from_cmdline as conflict,
        )

        tim_home = Path("/home/x/.hermes/profiles/tim")
        # Foreign target advertising timothy — NOT ours.
        assert (
            conflict("python -m hermes_cli.main --profile timothy gateway run", tim_home)
            is True
        )
        # Our own exact name stays consistent.
        assert (
            conflict("python -m hermes_cli.main --profile tim gateway run", tim_home)
            is False
        )
        assert (
            conflict("python -m hermes_cli.main -p tim gateway run", tim_home)
            is False
        )

    def test_explicit_home_flag_exact_compare(self, profile_env):
        """HERMES_HOME= on the argv compares path-exactly, not by prefix."""
        from gateway.run import (
            _looks_like_profile_conflict_from_cmdline as conflict,
        )

        tim_home = Path("/home/x/.hermes/profiles/tim")
        assert (
            conflict(
                "python -m hermes_cli.main HERMES_HOME=/home/x/.hermes/profiles/timothy gateway run",
                tim_home,
            )
            is True
        )
        assert (
            conflict(
                "python -m hermes_cli.main --hermes-home /home/x/.hermes/profiles/tim/ gateway run",
                tim_home,
            )
            is False  # trailing slash normalizes away
        )

    def test_default_home_refuses_any_named_profile_flag(self):
        from gateway.run import (
            _looks_like_profile_conflict_from_cmdline as conflict,
        )

        root = Path("/home/x/.hermes")
        assert conflict("python -m x --profile sam run", root) is True
        assert conflict("python -m x -p sam run", root) is True
        assert conflict("python -m x run", root) is False

    def test_consistency_contradiction_refuses_even_with_agreeing_record(
        self, profile_env
    ):
        """Record says same-home but the argv explicitly advertises another
        profile → refuse (argv contradiction wins the conservative call)."""
        from gateway.run import _replace_target_belongs_to_other_profile

        with (
            patch(
                "gateway.status._read_pid_record",
                return_value=_record(home=str(profile_env / ".hermes")),
            ),
            patch(
                "gateway.status._get_pid_path",
                return_value=profile_env / ".hermes" / "gateway.pid",
            ),
            patch(
                "gateway.status._get_process_start_time",
                return_value=111222333,
            ),
            patch(
                "gateway.status._read_process_cmdline",
                return_value="python -m hermes_cli.main --profile other-profile gateway run",
            ),
            patch(
                "gateway.status._get_process_hermes_home",
                return_value=profile_env / ".hermes",
            ),
        ):
            assert _replace_target_belongs_to_other_profile(424242) is True


class TestSignalBoundary:
    """Integration witness at the destructive boundary (#89315 review req)."""

    def _run_replace(self, agent_patches):
        from gateway import run as gateway_run

        calls = {"terminate": 0, "marker": 0, "status": []}

        def _fake_terminate(pid, force=False):
            calls["terminate"] += 1

        def _fake_marker(pid):
            calls["marker"] += 1

        def _fake_status(**fields):
            calls["status"].append(fields)

        base = [
            patch("gateway.status.get_running_pid", return_value=424242),
            patch.object(gateway_run, "_replace_target_belongs_to_other_profile"),
            patch("gateway.status.write_runtime_status", side_effect=_fake_status),
            patch("gateway.status.terminate_pid", side_effect=_fake_terminate),
            patch("gateway.status.write_takeover_marker", side_effect=_fake_marker),
        ]
        import contextlib

        with contextlib.ExitStack() as stack:
            for p in base:
                stack.enter_context(p)
            # caller configures the guard mock
            agent_patches(stack)
            try:
                result = asyncio_run(gateway_run.start_gateway(replace=True))
            except Exception:
                result = "raised"
        return result, calls

    def test_unprovable_ownership_never_signals(self, profile_env):
        """Unprovable ownership → start_gateway returns False WITHOUT calling
        terminate_pid or writing a takeover marker."""

        def configure(stack):
            guard = stack.enter_context(
                patch(
                    "gateway.run._replace_target_belongs_to_other_profile",
                    return_value=True,
                )
            )
            return guard

        result, calls = self._run_replace(lambda s: configure(s))

        assert result is False, (
            "the refusal must stay exit-1-shaped so a supervisor's next attempt "
            "can succeed once the stale record stops being trusted"
        )
        assert calls["terminate"] == 0, (
            "--replace must not signal a target whose ownership is unproven"
        )
        assert calls["marker"] == 0, (
            "no takeover marker may be written for a refused target"
        )

    def test_unprovable_ownership_stamps_startup_failed(self, profile_env):
        """The refusal publishes the retained fatal state + an exit reason, so
        /api/status reports startup_failed instead of a stale heartbeat and the
        next start no longer trusts the stale PID record."""
        def configure(stack):
            stack.enter_context(
                patch(
                    "gateway.run._replace_target_belongs_to_other_profile",
                    return_value=True,
                )
            )

        _result, calls = self._run_replace(configure)

        assert calls["status"], "the ownership refusal must stamp runtime status"
        stamp = calls["status"][-1]
        assert stamp.get("gateway_state") == "startup_failed"
        assert stamp.get("exit_reason") == "gateway_replace_ownership_unproven"

    def test_provable_same_home_reaches_replace_flow(self, profile_env):
        """Counterpart: bound same-home target still enters the replace flow
        (terminate attempted) — the fail-closed gate must not disable legit
        Windows-style replaces."""
        def configure(stack):
            stack.enter_context(
                patch(
                    "gateway.run._replace_target_belongs_to_other_profile",
                    return_value=False,
                )
            )
            stack.enter_context(
                patch(
                    "gateway.status.get_process_start_time",
                    return_value=111222333,
                )
            )
            stack.enter_context(patch("gateway.run.time.sleep"))

        _result, calls = self._run_replace(configure)

        assert calls["terminate"] == 1, (
            "a provably same-home target must still be replaceable"
        )


def asyncio_run(coro):
    import asyncio

    return asyncio.new_event_loop().run_until_complete(_swallow(coro))


async def _swallow(coro):
    """Run the coroutine; later machinery (runtime locks etc.) may raise in
    unit context — callers inspect side-effect counters, not the outcome."""
    try:
        return await coro
    except Exception:
        return "raised"


_CHILD_SCRIPT = '''
import asyncio, sys, time
sys.path.insert(0, r"@@REPO@@")

import gateway.run as gateway_run
import gateway.status as status

# A stale record nothing can prove belongs to this home.
status.get_running_pid = lambda *args, **kwargs: 424242
gateway_run._replace_target_belongs_to_other_profile = lambda pid: True

# A stalled disk: the background writer takes five seconds to persist, so an
# unflushed stamp dies with the process (the CLI exits through os._exit).
def _slow_write(path, payload):
    time.sleep(5.0)
    status._write_json_file(path, payload)

status._get_runtime_status_writer = lambda: status._RuntimeStatusWriter(write_fn=_slow_write)

result = None
try:
    result = asyncio.run(gateway_run.start_gateway(replace=True))
except SystemExit as exc:
    result = exc.code
print(f"start_gateway -> {result!r}", file=sys.stderr)
# The real CLI exit path (gateway/run.py main()): False -> exit 1 -> os._exit.
gateway_run._exit_after_graceful_shutdown(0 if result is True else (1 if result is False else 2))
'''


class TestStampDurability:
    """The refusal stamp must be durable BEFORE the CLI's os._exit hard-exit path.

    _write_runtime_status_quiet only queues the snapshot to the background
    writer; the caller then hard-exits without flushing it, so the next
    supervised start could read the same stale record and refuse again — the
    exact loop this branch exists to break. Regression crosses the process
    boundary: a child enters the refusal branch and exits through the real
    hard-exit path; the parent verifies what actually reached disk.
    """

    def test_refusal_stamp_reaches_disk_before_the_hard_exit(self, tmp_path):
        home = tmp_path / "scratch-home"
        home.mkdir()
        (home / "gateway_state.json").write_text(
            json.dumps({"gateway_state": "running", "pid": 424242}), encoding="utf-8")
        repo_root = str(Path(__file__).resolve().parents[2])
        child = tmp_path / "refusal_child.py"
        child.write_text(_CHILD_SCRIPT.replace("@@REPO@@", repo_root), encoding="utf-8")
        env = {**os.environ, "HERMES_HOME": str(home)}

        proc = subprocess.run(
            [sys.executable, str(child)],
            env=env, capture_output=True, text=True, timeout=120, check=False,
        )

        assert proc.returncode == 1, f"child stderr: {proc.stderr[-2000:]}"
        stamped = json.loads((home / "gateway_state.json").read_text(encoding="utf-8"))
        assert stamped.get("gateway_state") == "startup_failed"
        assert stamped.get("exit_reason") == "gateway_replace_ownership_unproven"

        verify = subprocess.run(
            [sys.executable, "-c",
             "import gateway.status as s; print(s.get_runtime_status_running_pid())"],
            env=env, capture_output=True, text=True, timeout=60, cwd=repo_root, check=False,
        )
        assert verify.stdout.strip() == "None", (
            "a fresh process must no longer trust the stale PID after the refusal"
        )
