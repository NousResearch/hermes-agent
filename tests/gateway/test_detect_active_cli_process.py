"""Tests for gateway.status.detect_active_cli_process.

Builds a fake /proc tree (real filesystem, no mocking of Path/os) so the
scan logic runs exactly as it would against real /proc/*/cmdline files.
"""

from gateway.status import detect_active_cli_process


def _write_cmdline(proc_dir, pid: int, argv: list[str]) -> None:
    d = proc_dir / str(pid)
    d.mkdir(parents=True)
    (d / "cmdline").write_bytes(("\x00".join(argv) + "\x00").encode("utf-8"))


class TestDetectActiveCliProcess:
    def test_no_proc_dir_returns_false(self, tmp_path):
        assert detect_active_cli_process(proc_dir=tmp_path / "does-not-exist") is False

    def test_empty_proc_returns_false(self, tmp_path):
        assert detect_active_cli_process(proc_dir=tmp_path) is False

    def test_bare_hermes_invocation_detected(self, tmp_path):
        _write_cmdline(
            tmp_path, 111,
            ["/home/user/.hermes/hermes-agent/.venv/bin/python3",
             "/home/user/.hermes/hermes-agent/.venv/bin/hermes"],
        )
        assert detect_active_cli_process(proc_dir=tmp_path) is True

    def test_explicit_chat_subcommand_detected(self, tmp_path):
        _write_cmdline(
            tmp_path, 112,
            ["/usr/bin/python3", "/home/user/.local/bin/hermes", "chat", "--resume"],
        )
        assert detect_active_cli_process(proc_dir=tmp_path) is True

    def test_positional_chat_message_detected(self, tmp_path):
        # `hermes "summarize this"` -- no recognized subcommand token, so the
        # positional is treated as a chat message, not a real subcommand.
        _write_cmdline(
            tmp_path, 113,
            ["/usr/bin/python3", "/home/user/.local/bin/hermes", "summarize this"],
        )
        assert detect_active_cli_process(proc_dir=tmp_path) is True

    def test_gateway_subcommand_not_detected(self, tmp_path):
        _write_cmdline(
            tmp_path, 114,
            ["/usr/bin/python3", "/home/user/.local/bin/hermes", "gateway", "run"],
        )
        assert detect_active_cli_process(proc_dir=tmp_path) is False

    def test_dashboard_subcommand_not_detected(self, tmp_path):
        _write_cmdline(
            tmp_path, 115,
            ["/usr/bin/python3", "/home/user/.local/bin/hermes", "dashboard", "--port", "9119"],
        )
        assert detect_active_cli_process(proc_dir=tmp_path) is False

    def test_setup_subcommand_not_detected(self, tmp_path):
        _write_cmdline(
            tmp_path, 116,
            ["/usr/bin/python3", "/home/user/.local/bin/hermes", "setup"],
        )
        assert detect_active_cli_process(proc_dir=tmp_path) is False

    def test_module_invocation_without_hermes_script_not_detected(self, tmp_path):
        # This is exactly how the gateway/dashboard services are actually
        # started (systemd ExecStart): `python -m hermes_cli.main <cmd>` --
        # no ".../bin/hermes" token anywhere in argv, so it must never match.
        _write_cmdline(
            tmp_path, 117,
            ["/home/user/.hermes/hermes-agent/.venv/bin/python", "-m",
             "hermes_cli.main", "gateway", "run"],
        )
        assert detect_active_cli_process(proc_dir=tmp_path) is False

    def test_module_invocation_dashboard_not_detected(self, tmp_path):
        _write_cmdline(
            tmp_path, 118,
            ["/home/user/.hermes/hermes-agent/.venv/bin/python", "-m",
             "hermes_cli.main", "dashboard", "--host", "0.0.0.0"],
        )
        assert detect_active_cli_process(proc_dir=tmp_path) is False

    def test_hermes_agent_binary_not_confused_with_hermes(self, tmp_path):
        # `hermes-agent` is a DIFFERENT entry point (run_agent:main) -- must
        # not match just because its basename contains "hermes".
        _write_cmdline(tmp_path, 119, ["/usr/bin/python3", "/home/user/.local/bin/hermes-agent"])
        assert detect_active_cli_process(proc_dir=tmp_path) is False

    def test_hermes_acp_binary_not_confused_with_hermes(self, tmp_path):
        _write_cmdline(tmp_path, 120, ["/usr/bin/python3", "/home/user/.local/bin/hermes-acp"])
        assert detect_active_cli_process(proc_dir=tmp_path) is False

    def test_non_numeric_proc_entries_skipped(self, tmp_path):
        (tmp_path / "self").mkdir()
        (tmp_path / "cpuinfo").write_text("fake")
        assert detect_active_cli_process(proc_dir=tmp_path) is False

    def test_unreadable_cmdline_skipped_not_fatal(self, tmp_path):
        d = tmp_path / "121"
        d.mkdir()
        # No cmdline file at all (process could have exited mid-scan) --
        # must not raise.
        assert detect_active_cli_process(proc_dir=tmp_path) is False

    def test_mix_of_service_and_cli_processes_detects_the_cli_one(self, tmp_path):
        _write_cmdline(
            tmp_path, 200,
            ["/home/user/.hermes/hermes-agent/.venv/bin/python", "-m",
             "hermes_cli.main", "gateway", "run"],
        )
        _write_cmdline(
            tmp_path, 201,
            ["/home/user/.hermes/hermes-agent/.venv/bin/python", "-m",
             "hermes_cli.main", "dashboard"],
        )
        _write_cmdline(
            tmp_path, 202,
            ["/usr/bin/python3", "/home/user/.local/bin/hermes"],
        )
        assert detect_active_cli_process(proc_dir=tmp_path) is True
