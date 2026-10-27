"""Process substitutions are not command boundaries for the referenced-script walk (#145145).

``_split_segments`` split every segment at ``)`` (a punctuation-char control token), so in
``diff <(echo a) /dev/null`` the ``/dev/null`` operand started a fresh segment whose first token was
scanned as an executed command: a character device fails the regular-file check and the guard
blocks the read-only diff (``lifecycle guard cannot scan referenced script /dev/null``), and a
large data file in the same position fails closed on the size cap. A ``<(``/``>(`` group executes
its interior but the tokens after the closing ``)`` stay operands of the outer command.
"""

import pytest

from cron.lifecycle_guard import (
    _iter_referenced_shell_scripts,
    _shlex_tokens,
    _split_segments,
    scan_gateway_lifecycle,
)


def _segments(command: str) -> list[list[str]]:
    return list(_split_segments(_shlex_tokens(command)))


class TestProcessSubstitutionIsNotACommandBoundary:
    """The operand after ``<(...)`` / ``>(...)`` stays with the outer command."""

    @pytest.mark.parametrize("command", [
        "diff <(echo a 2>/dev/null | head -5) /dev/null | head -8",
        "diff <(echo a) /dev/null",
        "diff >(wc -l) /dev/null",
        "diff <(echo $((1+1))) /dev/null",  # arithmetic: "((...)))" balances
        "diff <(head -5 <(echo nested)) /dev/null",  # nested substitutions
        "tee >(gzip > out.gz) file.txt",
        "diff <(echo a) <(echo b)",
        "while read l; do echo $l; done < <(grep x f.txt)",
    ])
    def test_operand_after_the_group_is_not_a_script_reference(self, command):
        assert list(_iter_referenced_shell_scripts(command)) == []

    def test_issue_reproduction_is_allowed(self):
        """The reported command: blocked before the fix (device read as a script), allowed now."""
        unsafe, refusal = scan_gateway_lifecycle(
            "diff <(echo a 2>/dev/null | head -5) /dev/null | head -8")
        assert (unsafe, refusal) == (False, None)

    def test_data_file_operand_is_not_walked_as_a_script(self, tmp_path):
        """A regular file after the group was also read as a script; a >1MiB one failed closed."""
        big = tmp_path / "big.dat"
        big.write_bytes(b"A" * (1024 * 1024 + 1))
        command = f"diff <(echo a) {big}"
        assert list(_iter_referenced_shell_scripts(command, cwd=str(tmp_path))) == []
        assert scan_gateway_lifecycle(command) == (False, None)

    def test_first_token_of_the_outer_segment_survives(self):
        """/dev/null must be an operand of diff, not the executable of a phantom segment."""
        segments = _segments("diff <(echo a) /dev/null")
        assert ["diff", "/dev/null"] in segments
        assert ["/dev/null"] not in segments


class TestSubstitutionInteriorIsStillScanned:
    """Fixing the boundary must not stop the walk from seeing what the substitution runs."""

    def test_interior_command_is_its_own_segment(self):
        segments = _segments("diff <(echo a 2>/dev/null | head -5) /dev/null")
        assert ["echo", "a", "2>/dev/null"] in segments
        assert ["head", "-5"] in segments

    def test_interior_lifecycle_command_still_blocks(self):
        unsafe, _refusal = scan_gateway_lifecycle("diff <(echo x && hermes gateway restart) /dev/null")
        assert unsafe is True

    def test_interior_script_is_still_walked_and_blocks(self, tmp_path):
        script = tmp_path / "inner.sh"
        script.write_text("#!/bin/bash\nhermes gateway stop\n")
        command = f"diff <(bash {script}) /dev/null"
        assert script in list(_iter_referenced_shell_scripts(command, cwd=str(tmp_path)))
        unsafe, _ = scan_gateway_lifecycle(command, cwd=str(tmp_path))
        assert unsafe is True


class TestOrdinaryRedirectionsAndSubshellsUnchanged:
    """Plain redirections, subshells and rebuild mode keep their pre-fix behavior."""

    def test_plain_redirection_operand_stays_scannable(self):
        # "<" without "(": the target merges into the run; no script reference either way.
        assert list(_iter_referenced_shell_scripts("grep < file.txt")) == []

    def test_plain_subshell_still_splits(self):
        assert _segments("wc; (cd /tmp; ls)") == [["wc"], ["cd", "/tmp"], ["ls"]]

    def test_rebuild_mode_preserves_every_token(self):
        """keep_controls=True rebuilds the line text exactly (the data-sink masker depends on it)."""
        tokens = _shlex_tokens("tee >(gzip > out.gz) f.txt; echo hi")
        rebuilt = [t for segment in _split_segments(tokens, keep_controls=True) for t in segment]
        assert rebuilt == tokens
