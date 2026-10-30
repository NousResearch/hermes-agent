"""Environment-shaped output is scrubbed independently of its producer."""

import pytest

from agent.redact import redact_terminal_output


@pytest.mark.parametrize("command", ["python dump_env.py", "jq -r '.[]'", None])
def test_environment_output_masks_opaque_token(command):
    secret = "opaque-service-value-1234567890"
    output = f"HOME=/tmp/test\nMY_SERVICE_TOKEN={secret}\nLANG=C.UTF-8\n"
    redacted = redact_terminal_output(output, command, force=True)
    assert secret not in redacted
    assert "HOME=/tmp/test" in redacted


def test_source_output_preserves_fixture_assignments():
    output = 'MY_SERVICE_TOKEN="fixture-value-1234567890"\nprint(MY_SERVICE_TOKEN)\n'
    assert redact_terminal_output(output, "python source.py", force=True) == output


def test_code_execution_uses_the_same_environment_output_policy(monkeypatch):
    from tools.code_execution_tool import _clean_output

    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)
    secret = "opaque-execution-value-1234567890"
    output = f"HOME=/tmp/test\nMY_SERVICE_TOKEN={secret}\nLANG=C.UTF-8\n"
    redacted, _ = _clean_output(output)
    assert secret not in redacted
