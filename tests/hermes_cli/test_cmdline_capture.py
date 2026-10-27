"""Tests for hermes_cli.cmdline_capture: structural argv reconstruction from flattened
``ps -o command=`` output, and refusal when that cannot be done without guessing.

Regression context: ``shlex.split`` on a line holding an inline ``-c <bootstrap>``
payload shatters the bootstrap into bare tokens; respawning that argv dies with
``SyntaxError`` on ``-c import``. See ``test_update_stale_dashboard.py ::
TestCmdlineCapture::test_flattened_ps_output_with_inline_c_payload_is_not_shattered``.
"""

import struct
import subprocess
import sys

import pytest

from hermes_cli import cmdline_capture as cc


def _procargs2_buffer(exec_path: str, argv: list[str]) -> bytes:
    """KERN_PROCARGS2-shaped buffer: int32 argc, exec path, NUL padding, argv strings."""
    body = struct.pack("=i", len(argv)) + exec_path.encode() + b"\x00" + b"\x00" * 4
    return body + b"\x00".join(a.encode() for a in argv) + b"\x00"


FLATTENED_C_LINE = (
    "/Users/sgt.snips/.hermes/tools/python3 -I -c "
    "import sys, runpy; sys.path.insert(0, '/repo'); "
    "sys.argv = ['/repo/.venv/bin/hermes', 'dashboard', '--no-open']; "
    "runpy.run_path('/repo/.venv/bin/hermes', run_name='__main__') "
    "dashboard --no-open --host 127.0.0.1 --port 9119 --skip-build"
)


class TestPlainCommandRoundTrip:
    def test_plain_argv_round_trips_through_shlex(self):
        assert cc.reconstruct_argv(1, "hermes serve --port 9119") == [
            "hermes", "serve", "--port", "9119"]

    def test_quoted_elements_split_by_shlex(self):
        assert cc.reconstruct_argv(1, "/usr/bin/python3 -m hermes_cli.main serve") == [
            "/usr/bin/python3", "-m", "hermes_cli.main", "serve"]

    def test_empty_command_refuses(self):
        assert cc.reconstruct_argv(1, "   ") is None

    def test_unbalanced_quote_falls_back_to_whitespace_split(self):
        # Never round-tripped quoted elements; refusal here would regress plain argvs.
        assert cc.reconstruct_argv(1, 'echo "hello world') == ["echo", '"hello', "world"]


class TestInlineCPayload:
    def test_non_macos_refuses_without_guessing(self, monkeypatch):
        monkeypatch.setattr(cc.sys, "platform", "linux")
        monkeypatch.setattr(cc, "_macos_procargs_bytes",
                            lambda pid: (_ for _ in ()).throw(AssertionError("sysctl called")))
        assert cc.reconstruct_argv(42192, FLATTENED_C_LINE) is None

    @pytest.mark.skipif(sys.platform != "darwin", reason="KERN_PROCARGS2 is macOS-only")
    def test_bootstrap_recovered_intact_from_kernel_record(self, monkeypatch):
        argv = ["/Users/sgt.snips/.hermes/tools/python3", "-I", "-c",
                "import sys, runpy; sys.path.insert(0, '/repo')", "dashboard", "--no-open"]
        monkeypatch.setattr(cc, "_macos_procargs_bytes",
                            lambda pid: _procargs2_buffer(argv[0], argv))
        assert cc.reconstruct_argv(42192, FLATTENED_C_LINE) == argv

    @pytest.mark.skipif(sys.platform != "darwin", reason="KERN_PROCARGS2 is macOS-only")
    def test_kernel_record_disagreeing_with_ps_line_refuses(self, monkeypatch):
        # Signature of a recycled pid or a ps/kernel race: never answer with the kernel's
        # argv for a different process.
        argv = ["/usr/bin/python4", "-I", "-c", "import sys", "dashboard"]
        monkeypatch.setattr(cc, "_macos_procargs_bytes",
                            lambda pid: _procargs2_buffer(argv[0], argv))
        assert cc.reconstruct_argv(42192, FLATTENED_C_LINE) is None

    @pytest.mark.skipif(sys.platform != "darwin", reason="KERN_PROCARGS2 is macOS-only")
    def test_kernel_record_without_c_refuses(self, monkeypatch):
        argv = ["/Users/sgt.snips/.hermes/tools/python3", "dashboard", "--port", "9119"]
        monkeypatch.setattr(cc, "_macos_procargs_bytes",
                            lambda pid: _procargs2_buffer(argv[0], argv))
        assert cc.reconstruct_argv(42192, FLATTENED_C_LINE) is None

    @pytest.mark.skipif(sys.platform != "darwin", reason="KERN_PROCARGS2 is macOS-only")
    def test_unreadable_kernel_record_refuses(self, monkeypatch):
        monkeypatch.setattr(cc, "_macos_procargs_bytes", lambda pid: None)
        assert cc.reconstruct_argv(42192, FLATTENED_C_LINE) is None


class TestProcargs2Decode:
    def test_wellformed_buffer_decodes_all_elements(self):
        argv = ["/bin/sh", "-c", "echo hi", "arg after empty"]
        assert cc._decode_procargs2(_procargs2_buffer(argv[0], argv)) == argv

    def test_truncated_buffer_is_none(self):
        assert cc._decode_procargs2(b"") is None
        assert cc._decode_procargs2(b"\x01") is None
        assert cc._decode_procargs2(struct.pack("=i", 2) + b"/bin/sh\x00") is None

    def test_zero_argc_is_none(self):
        assert cc._decode_procargs2(
            struct.pack("=i", 0) + b"/bin/sh\x00\x00\x00\x00\x00\x00") is None

    def test_argc_exceeding_strings_is_none(self):
        assert cc._decode_procargs2(
            struct.pack("=i", 3) + b"/bin/sh\x00\x00\x00\x00\x00only\x00") is None

    def test_empty_argv_element_refuses(self):
        # An empty argv element is unreadable-as-recorded for our purposes: refuse.
        assert cc._decode_procargs2(
            struct.pack("=i", 2) + b"/bin/sh\x00\x00\x00\x00\x00\x00-c\x00") is None


@pytest.mark.skipif(sys.platform != "darwin", reason="live KERN_PROCARGS2 proof")
class TestLiveKernel:
    def test_procargs2_matches_spawned_c_process(self):
        """Real kernel proof: a spawned ``python -I -c`` child's argv is reconstructed
        exactly, and its ps line is indeed flattened (the shape that used to shatter)."""
        payload = "import time; time.sleep(10)"
        proc = subprocess.Popen(
            [sys.executable, "-I", "-c", payload],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            ps = subprocess.run(["ps", "-p", str(proc.pid), "-o", "command="],
                                capture_output=True, text=True, timeout=10,
                                check=False)
            flattened = ps.stdout.strip()
            assert "-c" in flattened.split()
            assert "import time; time.sleep(10)" not in flattened.split(), (
                "ps did not flatten as expected; this test's premise changed")

            argv = cc.reconstruct_argv(proc.pid, flattened)
            assert argv == [sys.executable, "-I", "-c", payload]
        finally:
            proc.terminate()
            proc.wait(timeout=10)
