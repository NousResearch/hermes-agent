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


def _write(script, body, op=">"):
    return f"cat {op} {script} <<'EOF'\n{body}EOF\n"


@pytest.mark.parametrize(
    "shape",
    [
        "{w_life}bash {script}\n{w_ok}",
        "{w_life}nohup bash {script} &\n{w_ok}",
        "{a_life}{a_ok}bash {script}",
        "{a_ok}{a_life}bash {script}",
    ],
)
def test_every_body_written_to_a_path_is_scanned(tmp_path, shape):
    """A later write to the same path must not hide an earlier lifecycle body: the script may run
    between the writes, and ``>>`` keeps every appended body."""
    script = tmp_path / "gw.sh"
    command = shape.format(
        script=script,
        w_life=_write(script, BODY), w_ok=_write(script, "echo ok\n"),
        a_life=_write(script, BODY, ">>"), a_ok=_write(script, "echo ok\n", ">>"),
    )
    assert guard(command, cwd=str(tmp_path)) is True


@pytest.mark.parametrize(
    "run",
    [
        "systemd-run --user --on-active=300 /bin/bash -c '{script}'",
        "nohup sh -c 'sleep 300; {script}' &",
        "sh -c 'bash {script}'",
        "su -c 'bash {script}'",
        "env -S 'bash {script}'",
        "bash -c \"sh -c 'bash {script}'\"",
    ],
)
def test_written_script_run_from_nested_shell_payload_blocks(tmp_path, run):
    script = tmp_path / "gw.sh"
    command = _write(script, BODY) + f"chmod +x {script}\n" + run.format(script=script)
    assert guard(command, cwd=str(tmp_path)) is True


def test_nested_payload_running_harmless_written_script_stays_allowed(tmp_path):
    script = tmp_path / "build.sh"
    command = _write(script, "echo building\n") + f"sh -c 'bash {script}'"
    assert guard(command, cwd=str(tmp_path)) is False


def test_large_write_then_run_is_not_charged_twice(tmp_path):
    """The body is a substring of the already-charged command: scanning it as the script must not
    exhaust the line budget a second time."""
    script = tmp_path / "big.sh"
    body = "echo line\n" * 10_000
    assert guard(_write(script, body) + f"bash {script}", cwd=str(tmp_path)) is False
