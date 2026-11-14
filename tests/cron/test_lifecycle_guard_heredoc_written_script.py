"""A heredoc body the same command writes and then executes is scanned as that script (#148561).

The referenced-script walk reads scripts from disk, but ``cat > P <<'EOF' ... EOF; bash P`` creates
``P`` only when the command runs, so at scan time there was nothing to read and the quoted ``cat``
body itself is masked as documentation (#88336).
"""

import pytest

from cron.lifecycle_guard import contains_gateway_lifecycle_command_or_referenced_script as guard

BODY = "#!/usr/bin/env bash\nsleep 300\nsystemctl --user restart hermes-gateway.service\n"


@pytest.mark.parametrize(
    "run",
    [
        "systemd-run --user --unit=gw-restart --collect {script}",
        "bash {script}",
        "cd {dir} && sh ./gw.sh",
        "nohup {script} >/dev/null 2>&1 &",
    ],
)
def test_written_then_executed_heredoc_body_blocks(tmp_path, run):
    script = tmp_path / "gw.sh"
    command = (
        f"cat > {script} <<'EOF'\n{BODY}EOF\n"
        f"chmod +x {script}\n"
        + run.format(script=script, dir=tmp_path)
    )
    assert not script.exists()
    assert guard(command, cwd=str(tmp_path)) is True


def test_tee_written_then_executed_heredoc_body_blocks(tmp_path):
    script = tmp_path / "gw.sh"
    command = f"cat <<'EOF' | tee {script} >/dev/null\n{BODY}EOF\nbash {script}"
    assert guard(command, cwd=str(tmp_path)) is True


def test_written_but_not_executed_runbook_stays_allowed(tmp_path):
    """#88336: a cat body that is only written (or only read back) is documentation."""
    runbook = tmp_path / "runbook.md"
    command = (
        f"cat > {runbook} <<'EOF'\na human can run: hermes gateway restart\nEOF\n"
        f"cat {runbook}"
    )
    assert guard(command, cwd=str(tmp_path)) is False


def test_written_then_executed_harmless_body_stays_allowed(tmp_path):
    script = tmp_path / "build.sh"
    command = f"cat > {script} <<'EOF'\necho building\nEOF\nbash {script}"
    assert guard(command, cwd=str(tmp_path)) is False
