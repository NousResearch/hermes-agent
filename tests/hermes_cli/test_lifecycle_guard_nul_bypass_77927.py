"""Tests for #77927: NUL-padded (or NUL-spliced) *text* scripts must be
scanned, never skipped as "binary".

bash executes NUL-bearing text just fine, so a single pad byte must not
silence the guard. The pre-fix classifier treated *any* NUL in the head
as a compiled binary ("nothing to scan"); the fix classifies binaries by
magic number only and strips NULs from text before scanning (fail-closed:
stripping can only splice tokens together, never apart).
"""

import pytest

from cron.lifecycle_guard import (
    GatewayLifecycleBlocked,
    _MAX_REFERENCED_SCRIPT_BYTES,
    _read_referenced_script,
    check_gateway_lifecycle,
    contains_gateway_lifecycle_command_or_referenced_script,
)


class TestNulPaddedScriptBypass77927:
    @staticmethod
    def _write(tmp_path, name, content: bytes):
        path = tmp_path / name
        path.write_bytes(content)
        return path

    def test_nul_padded_text_script_is_blocked(self, tmp_path):
        # The exact issue shape: a text script with a NUL pad byte on a
        # later line — `bash padded.sh` runs it, the guard must block it.
        script = self._write(
            tmp_path, "padded.sh", b"# ok\n# pad\x00\nhermes gateway restart\n"
        )
        with pytest.raises(GatewayLifecycleBlocked):
            check_gateway_lifecycle("daily ops", str(script))

    def test_nul_padded_shebang_script_is_blocked(self, tmp_path):
        script = self._write(
            tmp_path, "padded2.sh", b"#!/bin/bash\x00\nhermes gateway stop\n"
        )
        with pytest.raises(GatewayLifecycleBlocked):
            check_gateway_lifecycle("daily ops", str(script))

    def test_nul_spliced_keyword_in_script_is_blocked(self, tmp_path):
        # Issue side benefit: `hermes gateway rest\x00art` inside a file
        # stops evading the matcher once NULs are stripped.
        script = self._write(
            tmp_path,
            "spliced.sh",
            b"#!/bin/bash\nhermes gateway rest\x00art\n",
        )
        with pytest.raises(GatewayLifecycleBlocked):
            check_gateway_lifecycle("daily ops", str(script))

    def test_terminal_command_referencing_nul_padded_script_is_blocked(self, tmp_path):
        # The terminal-tool entry point: `bash padded.sh` must be blocked
        # when the referenced file is NUL-padded text.
        script = self._write(
            tmp_path, "padded3.sh", b"# pad\x00\nhermes gateway restart\n"
        )
        assert (
            contains_gateway_lifecycle_command_or_referenced_script(
                f"bash {script}", cwd=str(tmp_path)
            )
            is True
        )

    def test_nul_padded_prompt_is_blocked(self):
        # The prompt channel is text too: NUL padding must not silence the
        # direct regex scan.
        with pytest.raises(GatewayLifecycleBlocked):
            check_gateway_lifecycle("run hermes gateway rest\x00art now", None)

    def test_nul_padded_remote_callback_text_is_scanned(self):
        # Remote reads follow the same contract: NUL-bearing *text* is
        # scanned with NULs stripped, not skipped as binary.
        result = contains_gateway_lifecycle_command_or_referenced_script(
            "bash /nonexistent/dir/helper.sh",
            cwd="/tmp",
            read_remote_script=lambda _p: (
                "#!/bin/sh\n# pad\x00\nhermes gateway restart\n"
            ),
        )
        assert result is True

    def test_launchctl_submit_with_spliced_nul_is_blocked(self, tmp_path):
        script = self._write(
            tmp_path,
            "submit.sh",
            b"#!/bin/bash\nlaunchctl sub\x00mit -l ai.hermes.gateway-loop -- /bin/sh x.sh\n",
        )
        with pytest.raises(GatewayLifecycleBlocked):
            check_gateway_lifecycle("daily ops", str(script))

    def test_magic_prefixed_binary_with_nuls_still_skipped(self, tmp_path):
        # The binary side of the contract is unchanged: magic numbers win,
        # even when the file also carries NULs and lifecycle-looking text.
        for name, magic in [
            ("elf", b"\x7fELF"),
            ("pe", b"MZ"),
            ("macho", b"\xcf\xfa\xed\xfe"),
            ("ar", b"!<arch>\n"),
            ("gzip", b"\x1f\x8b"),
            ("zip", b"PK\x03\x04"),
        ]:
            binary = self._write(
                tmp_path, name, magic + b"\x00hermes gateway restart\x00"
            )
            assert (
                contains_gateway_lifecycle_command_or_referenced_script(
                    f"bash {binary}", cwd=str(tmp_path)
                )
                is False
            ), name

    def test_oversized_nul_padded_file_still_fails_closed(self, tmp_path):
        # #77927 ordering detail: the size check must run BEFORE the NUL
        # strip, or a >1 MiB NUL-padded file shrinks under the threshold
        # and skips the fail-closed branch.
        script = tmp_path / "bigpadded.sh"
        script.write_bytes(b"\x00" * (_MAX_REFERENCED_SCRIPT_BYTES + 1))
        with pytest.raises(GatewayLifecycleBlocked):
            check_gateway_lifecycle("daily ops", str(script))

    def test_read_referenced_script_strips_nuls_from_text(self, tmp_path):
        script = self._write(
            tmp_path, "pad.sh", b"# ok\n# pad\x00\nhermes gateway restart\n"
        )
        text, unsafe = _read_referenced_script(script)
        assert unsafe is False
        assert text is not None
        assert "\x00" not in text
        assert "hermes gateway restart" in text
