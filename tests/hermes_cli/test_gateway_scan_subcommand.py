"""Regression tests for gateway-only PID matching in _scan_gateway_pids().

A same-profile ``dashboard`` (or any non-gateway subcommand) process carries
the same ``--profile`` / ``-p`` flag as a real gateway. The historical matcher
keyed on that flag alone, so it misidentified those processes as a running
gateway: ``gateway status``/``start`` reported the dashboard PID, and
``gateway stop``/``restart`` force-killed it and its process tree.

See: NousResearch/hermes-agent#47120
"""

import os
from unittest.mock import MagicMock, patch

import hermes_cli.gateway as gateway_mod


# ---------------------------------------------------------------------------
# _is_gateway_command — the pure classifier
# ---------------------------------------------------------------------------


class TestIsGatewayCommand:
    """Only real ``gateway`` invocations classify as gateway commands."""

    def test_module_gateway_run(self):
        assert gateway_mod._is_gateway_command(
            "python -m hermes_cli.main gateway run"
        )

    def test_module_profiled_gateway_run(self):
        assert gateway_mod._is_gateway_command(
            "python -m hermes_cli.main --profile gibson gateway run"
        )

    def test_dedicated_entrypoints(self):
        assert gateway_mod._is_gateway_command("hermes-gateway.exe")
        assert gateway_mod._is_gateway_command("python /opt/x/gateway/run.py")
        assert gateway_mod._is_gateway_command("hermes.exe gateway run")

    def test_dashboard_with_profile_is_not_gateway(self):
        # The core #47120 regression: a same-profile dashboard must NOT match.
        assert not gateway_mod._is_gateway_command(
            "python -m hermes_cli.main --profile gibson dashboard --no-open --port 0"
        )
        assert not gateway_mod._is_gateway_command(
            "python.exe -m hermes_cli.main --profile gibson dashboard --host 127.0.0.1 --port 0"
        )

    def test_other_subcommands_with_profile_are_not_gateway(self):
        assert not gateway_mod._is_gateway_command(
            "python -m hermes_cli.main -p gibson chat"
        )
        assert not gateway_mod._is_gateway_command(
            "python -m hermes_cli.main -p gibson dashboard"
        )

    def test_unrelated_process(self):
        assert not gateway_mod._is_gateway_command("python -m some_other_thing")

    def test_profile_literally_named_gateway_is_not_subcommand(self):
        # ``gateway`` as a --profile value is not the subcommand.
        assert not gateway_mod._is_gateway_command(
            "python -m hermes_cli.main -p gateway dashboard"
        )

    def test_profile_named_gateway_with_real_gateway_subcommand(self):
        assert gateway_mod._is_gateway_command(
            "python -m hermes_cli.main -p gateway gateway run"
        )


# ---------------------------------------------------------------------------
# _scan_gateway_pids — end-to-end via /proc
# ---------------------------------------------------------------------------


def _fake_proc_dir(entries: dict):
    def _isdir(path):
        return str(path) == "/proc"

    def _listdir(path):
        if str(path) == "/proc":
            return [str(pid) for pid in entries] + ["self", "version"]
        raise FileNotFoundError(path)

    def _open(path, mode="r", **kwargs):
        path_str = str(path)
        if "/cmdline" in path_str:
            pid = int(path_str.split("/proc/")[1].split("/")[0])
            raw = entries.get(pid, "").encode("utf-8").replace(b" ", b"\x00")
            m = MagicMock()
            m.read.return_value = raw
            m.__enter__ = lambda s: s
            m.__exit__ = MagicMock(return_value=False)
            return m
        raise FileNotFoundError(path)

    return _isdir, _listdir, _open


class TestScanExcludesDashboard:
    """A same-profile dashboard PID is never returned as a gateway."""

    def test_dashboard_pid_not_matched(self):
        my_pid = os.getpid()
        gateway_pid = 12345
        dashboard_pid = 23456
        entries = {
            my_pid: "python -m hermes_cli.main",
            gateway_pid: "python -m hermes_cli.main --profile gibson gateway run",
            dashboard_pid: (
                "python -m hermes_cli.main --profile gibson dashboard "
                "--no-open --host 127.0.0.1 --port 0"
            ),
        }
        _isdir, _listdir, _open = _fake_proc_dir(entries)

        with (
            patch("hermes_cli.gateway.is_windows", return_value=False),
            patch("os.path.isdir", side_effect=_isdir),
            patch("os.listdir", side_effect=_listdir),
            patch("builtins.open", side_effect=_open),
            patch("hermes_cli.gateway._get_ancestor_pids", return_value=set()),
            patch("subprocess.run"),
        ):
            pids = gateway_mod._scan_gateway_pids(set(), all_profiles=True)

        assert gateway_pid in pids
        assert dashboard_pid not in pids
