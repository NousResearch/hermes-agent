"""A `<(...)` / `>(...)` process substitution does not end the outer command (#145145).

The tokenizer splits `(` and `)` as command boundaries, so the operand after the closing `)` used to
start a new segment and was read as an executed script: `diff <(cmd) /dev/null` failed closed on the
character device. The substituted commands themselves still run, so they are still scanned.
"""

import pytest

from cron.lifecycle_guard import _iter_referenced_shell_scripts, scan_gateway_lifecycle


@pytest.mark.parametrize(
    "command",
    [
        "diff <(echo a 2>/dev/null | head -5) /dev/null | head -8",
        "diff <(echo a) ./other.txt",
        "tee >(gzip > out.gz) ./copy.txt",
        "diff <(sort <(cat a)) ./b.txt",
        "diff <( (cd x; ls) ) ./b.txt",
    ],
)
def test_operand_after_process_substitution_is_not_a_script(tmp_path, command):
    assert list(_iter_referenced_shell_scripts(command, cwd=str(tmp_path))) == []
    assert scan_gateway_lifecycle(command, cwd=str(tmp_path)) == (False, None)


@pytest.mark.parametrize(
    "command",
    [
        "diff <(./run.sh) /dev/null",
        "diff <(sort <(./run.sh)) /dev/null",
        "./run.sh <(echo a) /dev/null",
        "./run.sh <(echo a",
    ],
)
def test_executed_scripts_around_process_substitution_are_still_scanned(tmp_path, command):
    assert list(_iter_referenced_shell_scripts(command, cwd=str(tmp_path))) == [tmp_path / "run.sh"]
