"""
Tests for gateway.hermes_home_puller.

Covers:
- Disabled via env var
- Disabled via config
- Skips when not on main branch
- Skips when working tree is dirty
- Skips when already up-to-date (0 commits behind)
- Pulls when on main, clean, and behind
- Pull failure is non-fatal
- start_hermes_home_puller returns None when disabled
- start_hermes_home_puller returns None when hermes_home has no .git
- HermesHomePuller.stop() joins the thread cleanly
"""

from __future__ import annotations

import subprocess
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from gateway.hermes_home_puller import (
    HermesHomePuller,
    _commits_behind,
    _current_branch,
    _fast_forward,
    _is_clean,
    _is_disabled_by_config,
    _is_disabled_by_env,
    _pull_once,
    start_hermes_home_puller,
)


class TestDisabledByEnv(unittest.TestCase):
    def test_disabled_when_set_to_1(self):
        with patch.dict("os.environ", {"HERMES_GATEWAY_NO_AUTO_PULL": "1"}):
            self.assertTrue(_is_disabled_by_env())

    def test_disabled_when_set_to_true(self):
        with patch.dict("os.environ", {"HERMES_GATEWAY_NO_AUTO_PULL": "true"}):
            self.assertTrue(_is_disabled_by_env())

    def test_enabled_when_not_set(self):
        env = {k: v for k, v in __import__("os").environ.items()
               if k != "HERMES_GATEWAY_NO_AUTO_PULL"}
        with patch.dict("os.environ", env, clear=True):
            self.assertFalse(_is_disabled_by_env())

    def test_enabled_when_set_to_0(self):
        with patch.dict("os.environ", {"HERMES_GATEWAY_NO_AUTO_PULL": "0"}):
            self.assertFalse(_is_disabled_by_env())

    def test_enabled_when_set_to_false(self):
        with patch.dict("os.environ", {"HERMES_GATEWAY_NO_AUTO_PULL": "false"}):
            self.assertFalse(_is_disabled_by_env())


class TestDisabledByConfig(unittest.TestCase):
    def _make_cfg_get(self, return_value):
        """Patch cfg_get to return a fixed value for hermes_home_auto_pull."""
        def fake_cfg_get(cfg, *keys, default=None):
            if keys == ("gateway", "hermes_home_auto_pull"):
                return return_value
            return default
        return fake_cfg_get

    def test_disabled_when_config_false(self):
        with patch(
            "hermes_cli.config.cfg_get",
            side_effect=self._make_cfg_get(False),
        ):
            self.assertTrue(_is_disabled_by_config({"gateway": {}}))

    def test_enabled_when_config_true(self):
        with patch(
            "hermes_cli.config.cfg_get",
            side_effect=self._make_cfg_get(True),
        ):
            self.assertFalse(_is_disabled_by_config({"gateway": {}}))

    def test_enabled_when_cfg_is_none(self):
        self.assertFalse(_is_disabled_by_config(None))

    def test_enabled_when_cfg_get_raises(self):
        with patch(
            "hermes_cli.config.cfg_get",
            side_effect=ImportError("no config"),
        ):
            self.assertFalse(_is_disabled_by_config({"gateway": {}}))


class TestCurrentBranch(unittest.TestCase):
    def _make_completed_process(self, returncode, stdout="", stderr=""):
        cp = MagicMock(spec=subprocess.CompletedProcess)
        cp.returncode = returncode
        cp.stdout = stdout
        cp.stderr = stderr
        return cp

    def test_returns_branch_name(self):
        with patch(
            "gateway.hermes_home_puller._run_git",
            return_value=self._make_completed_process(0, "main\n"),
        ):
            self.assertEqual(_current_branch(Path("/fake")), "main")

    def test_returns_empty_on_failure(self):
        with patch(
            "gateway.hermes_home_puller._run_git",
            return_value=self._make_completed_process(1, ""),
        ):
            self.assertEqual(_current_branch(Path("/fake")), "")


class TestIsClean(unittest.TestCase):
    def _cp(self, returncode, stdout=""):
        cp = MagicMock(spec=subprocess.CompletedProcess)
        cp.returncode = returncode
        cp.stdout = stdout
        return cp

    def test_clean_tree(self):
        with patch("gateway.hermes_home_puller._run_git", return_value=self._cp(0, "")):
            self.assertTrue(_is_clean(Path("/fake")))

    def test_dirty_tree(self):
        with patch(
            "gateway.hermes_home_puller._run_git",
            return_value=self._cp(0, " M config.yaml\n"),
        ):
            self.assertFalse(_is_clean(Path("/fake")))

    def test_git_failure(self):
        with patch("gateway.hermes_home_puller._run_git", return_value=self._cp(128)):
            self.assertFalse(_is_clean(Path("/fake")))


class TestCommitsBehind(unittest.TestCase):
    def _cp(self, returncode, stdout="", stderr=""):
        cp = MagicMock(spec=subprocess.CompletedProcess)
        cp.returncode = returncode
        cp.stdout = stdout
        cp.stderr = stderr
        return cp

    def test_returns_count_behind(self):
        with patch(
            "gateway.hermes_home_puller._run_git",
            side_effect=[self._cp(0, ""), self._cp(0, "3\n")],  # fetch, rev-list
        ):
            self.assertEqual(_commits_behind(Path("/fake")), 3)

    def test_returns_zero_when_fetch_fails(self):
        with patch(
            "gateway.hermes_home_puller._run_git",
            return_value=self._cp(1, "", "fatal: not a git repo"),
        ):
            self.assertEqual(_commits_behind(Path("/fake")), 0)

    def test_returns_zero_when_up_to_date(self):
        with patch(
            "gateway.hermes_home_puller._run_git",
            side_effect=[self._cp(0, ""), self._cp(0, "0\n")],
        ):
            self.assertEqual(_commits_behind(Path("/fake")), 0)


class TestFastForward(unittest.TestCase):
    def _cp(self, returncode, stderr=""):
        cp = MagicMock(spec=subprocess.CompletedProcess)
        cp.returncode = returncode
        cp.stdout = ""
        cp.stderr = stderr
        return cp

    def test_returns_true_on_success(self):
        with patch(
            "gateway.hermes_home_puller._run_git", return_value=self._cp(0)
        ):
            self.assertTrue(_fast_forward(Path("/fake")))

    def test_returns_false_on_failure(self):
        with patch(
            "gateway.hermes_home_puller._run_git",
            return_value=self._cp(1, "Not possible to fast-forward"),
        ):
            self.assertFalse(_fast_forward(Path("/fake")))


class TestPullOnce(unittest.TestCase):
    """Integration-level tests for _pull_once — stubs _current_branch,
    _is_clean, _commits_behind, _fast_forward individually."""

    def test_skips_when_not_on_main(self):
        with (
            patch("gateway.hermes_home_puller._current_branch", return_value="feature"),
            patch("gateway.hermes_home_puller._commits_behind") as mock_behind,
        ):
            _pull_once(Path("/fake"))
            mock_behind.assert_not_called()

    def test_skips_when_up_to_date(self):
        # behind is now counted before the clean check, so up-to-date short
        # circuits without ever touching _is_clean or _fast_forward.
        with (
            patch("gateway.hermes_home_puller._current_branch", return_value="main"),
            patch("gateway.hermes_home_puller._commits_behind", return_value=0),
            patch("gateway.hermes_home_puller._is_clean") as mock_clean,
            patch("gateway.hermes_home_puller._fast_forward") as mock_ff,
        ):
            _pull_once(Path("/fake"))
            mock_clean.assert_not_called()
            mock_ff.assert_not_called()

    def test_pulls_when_clean_and_behind(self):
        with (
            patch("gateway.hermes_home_puller._current_branch", return_value="main"),
            patch("gateway.hermes_home_puller._commits_behind", return_value=2),
            patch("gateway.hermes_home_puller._is_clean", return_value=True),
            patch("gateway.hermes_home_puller._fast_forward") as mock_ff,
        ):
            _pull_once(Path("/fake"))
            mock_ff.assert_called_once_with(Path("/fake"))

    def test_dirty_and_behind_is_wedged_and_notifies(self):
        # main + behind>0 + dirty = WEDGED: no fast-forward, notify fires.
        notify = MagicMock()
        with (
            patch("gateway.hermes_home_puller._current_branch", return_value="main"),
            patch("gateway.hermes_home_puller._commits_behind", return_value=5),
            patch("gateway.hermes_home_puller._is_clean", return_value=False),
            patch("gateway.hermes_home_puller._fast_forward") as mock_ff,
        ):
            _pull_once(Path("/fake"), notify=notify)
            mock_ff.assert_not_called()
            notify.assert_called_once_with(5)

    def test_dirty_and_up_to_date_does_not_notify(self):
        # Dirty but behind==0 is NOT wedged — short-circuits at the behind
        # check before clean/notify are ever reached.
        notify = MagicMock()
        with (
            patch("gateway.hermes_home_puller._current_branch", return_value="main"),
            patch("gateway.hermes_home_puller._commits_behind", return_value=0),
            patch("gateway.hermes_home_puller._is_clean") as mock_clean,
            patch("gateway.hermes_home_puller._fast_forward") as mock_ff,
        ):
            _pull_once(Path("/fake"), notify=notify)
            notify.assert_not_called()
            mock_ff.assert_not_called()
            mock_clean.assert_not_called()

    def test_wedged_without_notify_callback_is_non_fatal(self):
        # No notify callback supplied — wedged state just logs, no crash.
        with (
            patch("gateway.hermes_home_puller._current_branch", return_value="main"),
            patch("gateway.hermes_home_puller._commits_behind", return_value=3),
            patch("gateway.hermes_home_puller._is_clean", return_value=False),
        ):
            _pull_once(Path("/fake"))  # should not raise

    def test_notify_callback_exception_is_non_fatal(self):
        notify = MagicMock(side_effect=RuntimeError("boom"))
        with (
            patch("gateway.hermes_home_puller._current_branch", return_value="main"),
            patch("gateway.hermes_home_puller._commits_behind", return_value=3),
            patch("gateway.hermes_home_puller._is_clean", return_value=False),
        ):
            _pull_once(Path("/fake"), notify=notify)  # should not raise
            notify.assert_called_once_with(3)

    def test_pull_failure_is_non_fatal(self):
        with (
            patch("gateway.hermes_home_puller._current_branch", return_value="main"),
            patch("gateway.hermes_home_puller._commits_behind", return_value=1),
            patch("gateway.hermes_home_puller._is_clean", return_value=True),
            patch("gateway.hermes_home_puller._fast_forward", return_value=False),
        ):
            # Should not raise
            _pull_once(Path("/fake"))

    def test_timeout_is_non_fatal(self):
        with (
            patch(
                "gateway.hermes_home_puller._current_branch",
                side_effect=subprocess.TimeoutExpired(cmd="git", timeout=30),
            ),
        ):
            # Should not raise
            _pull_once(Path("/fake"))


class TestStartHermesHomePuller(unittest.TestCase):
    def test_returns_none_when_disabled_by_env(self):
        with patch.dict("os.environ", {"HERMES_GATEWAY_NO_AUTO_PULL": "1"}):
            result = start_hermes_home_puller(hermes_home=Path("/fake"))
        self.assertIsNone(result)

    def test_returns_none_when_disabled_by_config(self):
        env = {k: v for k, v in __import__("os").environ.items()
               if k != "HERMES_GATEWAY_NO_AUTO_PULL"}
        with (
            patch.dict("os.environ", env, clear=True),
            patch("gateway.hermes_home_puller._is_disabled_by_config", return_value=True),
        ):
            result = start_hermes_home_puller(
                cfg={}, hermes_home=Path("/fake")
            )
        self.assertIsNone(result)

    def test_returns_none_when_no_git_dir(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {k: v for k, v in __import__("os").environ.items()
                   if k != "HERMES_GATEWAY_NO_AUTO_PULL"}
            with patch.dict("os.environ", env, clear=True):
                result = start_hermes_home_puller(hermes_home=Path(tmpdir))
        self.assertIsNone(result)

    def test_returns_puller_when_git_dir_present(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            git_dir = Path(tmpdir) / ".git"
            git_dir.mkdir()
            env = {k: v for k, v in __import__("os").environ.items()
                   if k != "HERMES_GATEWAY_NO_AUTO_PULL"}
            with (
                patch.dict("os.environ", env, clear=True),
                patch.object(HermesHomePuller, "start"),  # don't actually start the thread
            ):
                result = start_hermes_home_puller(hermes_home=Path(tmpdir))
        self.assertIsInstance(result, HermesHomePuller)


class TestWedgedAlertThrottle(unittest.TestCase):
    """Exercises HermesHomePuller._on_wedged throttle semantics directly,
    using a fake monotonic clock so we don't sleep through a 6h window."""

    def _make_puller(self, notifier, throttle=3600.0):
        return HermesHomePuller(
            hermes_home=Path("/fake"),
            interval=9999,
            notifier=notifier,
            alert_throttle_secs=throttle,
        )

    def test_first_wedged_alert_fires(self):
        notifier = MagicMock()
        puller = self._make_puller(notifier)
        puller._now = lambda: 1000.0
        puller._on_wedged(4)
        notifier.assert_called_once()
        # Message carries the behind count and actionable guidance.
        (msg,), _ = notifier.call_args
        self.assertIn("4", msg)
        self.assertIn("WEDGED", msg)

    def test_second_alert_within_window_is_suppressed(self):
        notifier = MagicMock()
        puller = self._make_puller(notifier, throttle=3600.0)
        clock = {"t": 1000.0}
        puller._now = lambda: clock["t"]

        puller._on_wedged(4)
        self.assertEqual(notifier.call_count, 1)

        # 30 min later — inside the 1h window → suppressed.
        clock["t"] = 1000.0 + 1800.0
        puller._on_wedged(5)
        self.assertEqual(notifier.call_count, 1)

    def test_alert_fires_again_after_window_elapses(self):
        notifier = MagicMock()
        puller = self._make_puller(notifier, throttle=3600.0)
        clock = {"t": 1000.0}
        puller._now = lambda: clock["t"]

        puller._on_wedged(4)
        self.assertEqual(notifier.call_count, 1)

        # 2h later — past the 1h window → fires again.
        clock["t"] = 1000.0 + 7200.0
        puller._on_wedged(6)
        self.assertEqual(notifier.call_count, 2)

    def test_no_notifier_is_noop(self):
        puller = self._make_puller(notifier=None)
        puller._now = lambda: 1000.0
        # Should not raise and should not advance the throttle clock.
        puller._on_wedged(4)
        self.assertIsNone(puller._last_alert_ts)

    def test_notifier_exception_does_not_advance_throttle(self):
        # If the notifier raises, the alert wasn't delivered — the throttle
        # clock must NOT advance, so the next tick retries.
        notifier = MagicMock(side_effect=RuntimeError("send failed"))
        puller = self._make_puller(notifier)
        puller._now = lambda: 1000.0
        puller._on_wedged(4)
        self.assertIsNone(puller._last_alert_ts)
        self.assertEqual(notifier.call_count, 1)


class TestStartPullerThreadsNotifier(unittest.TestCase):
    def test_notifier_is_passed_to_puller(self):
        import tempfile
        notifier = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            git_dir = Path(tmpdir) / ".git"
            git_dir.mkdir()
            env = {k: v for k, v in __import__("os").environ.items()
                   if k != "HERMES_GATEWAY_NO_AUTO_PULL"}
            with (
                patch.dict("os.environ", env, clear=True),
                patch.object(HermesHomePuller, "start"),
            ):
                puller = start_hermes_home_puller(
                    hermes_home=Path(tmpdir), notifier=notifier
                )
        self.assertIsInstance(puller, HermesHomePuller)
        self.assertIs(puller._notifier, notifier)


class TestHermesHomePullerLifecycle(unittest.TestCase):
    def test_stop_joins_thread(self):
        hermes_home = Path("/fake")
        puller = HermesHomePuller(hermes_home=hermes_home, interval=9999)

        # Replace _run with a quick no-op to avoid git calls
        started = threading.Event()
        stopped = threading.Event()

        def quick_run():
            started.set()
            puller._stop_event.wait()
            stopped.set()

        puller._run = quick_run
        puller.start()
        started.wait(timeout=2)
        puller.stop(timeout=2)
        self.assertTrue(stopped.is_set(), "thread should have exited after stop()")


if __name__ == "__main__":
    unittest.main()
