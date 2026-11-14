"""Vault registry calls must survive finalizers that log through redaction."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("operation", ["register", "clear", "redact"])
def test_vault_scope_finalizer_can_log_without_deadlocking(operation):
    # A child bounds the deadlock on unfixed code without leaving a blocked
    # thread (or a held logging lock) in the pytest worker.
    script = textwrap.dedent(
        """
        import gc
        import io
        import logging
        import sys

        import hermes_constants
        from agent import redact

        secret = "vault-finalizer-test-value"
        redact.register_vault_redaction_value(secret)
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        handler.setFormatter(redact.RedactingFormatter("%(message)s"))
        logger = logging.getLogger("vault-finalizer-regression")
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        logger.propagate = False

        class LogOnCollect:
            def __del__(self):
                logger.warning("finalizer echoed %s", secret)

        original_home = hermes_constants.get_hermes_home
        armed = True

        def resolve_home():
            global armed
            if armed:
                armed = False
                cycle = LogOnCollect()
                cycle.cycle = cycle
                del cycle
                gc.collect()
            return original_home()

        hermes_constants.get_hermes_home = resolve_home
        operation = sys.argv[1]
        if operation == "register":
            redact.register_vault_redaction_value("another-vault-value")
            assert "another-vault-value" not in redact.redact_registered_vault_values(
                "another-vault-value"
            )
        elif operation == "clear":
            redact.clear_vault_redaction_values()
            assert redact.redact_registered_vault_values(secret) == secret
        else:
            assert secret not in redact.redact_registered_vault_values(secret)

        assert "finalizer echoed" in output.getvalue()
        assert secret not in output.getvalue(), output.getvalue()
        print("completed")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, operation],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "completed"
