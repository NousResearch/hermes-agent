"""Process-level regression coverage for the standalone doctor command."""

from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def test_doctor_command_exits_with_a_stuck_non_daemon_worker():
    """Completed diagnostics must not hang during interpreter shutdown (#100792)."""
    program = textwrap.dedent(
        """
        import sys
        import threading

        import hermes_cli.doctor as doctor
        from hermes_cli.main import cmd_doctor

        blocker = threading.Event()
        def diagnostics(_args):
            threading.Thread(
                target=blocker.wait, name="stuck-doctor-dependency", daemon=False
            ).start()
            print("diagnostics complete")
            print("diagnostic warning", file=sys.stderr)

        doctor.run_doctor = diagnostics

        cmd_doctor(object())
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == b"diagnostics complete\n"
    assert b"diagnostic warning\n" in result.stderr


@pytest.mark.parametrize("error", [RuntimeError("diagnostics failed"), SystemExit(2)])
def test_doctor_command_preserves_diagnostic_errors(monkeypatch, error):
    import hermes_cli.doctor as doctor
    from hermes_cli.main import cmd_doctor

    def fail(_args):
        raise error

    monkeypatch.setattr(doctor, "run_doctor", fail)

    with pytest.raises(type(error)) as caught:
        cmd_doctor(object())

    assert caught.value is error
