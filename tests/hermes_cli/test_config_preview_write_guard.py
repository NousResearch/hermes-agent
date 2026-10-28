"""``save_env_value`` is the write chokepoint every writer shares (``hermes config set``, setup
wizards, dashboard / messaging / MCP routes, the agent's own tools): it refuses display-only
credential previews, so a copied preview can never replace the real secret (#35519)."""

import os
from unittest.mock import patch

import pytest

from agent.redact import REDACTED_CREDENTIAL_WRITE_DETAIL
from hermes_cli.config import load_env, save_env_value


@pytest.mark.parametrize("preview", [
    "***", "***#abcdef", "«redacted:***#abcdef»", "«redacted-secret»", "abcd...efgh",
    # Non-ASCII stripping must not turn a value INTO a preview after the check ran.
    "***#abc\u200bdef", "***\u00a0",
])
def test_preview_shapes_raise_and_leave_env_untouched(tmp_path, preview):
    env_path = tmp_path / ".env"
    env_path.write_text("EXA_API_KEY=real-secret-value\n")
    before = env_path.read_bytes()
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        with pytest.raises(ValueError, match="redacted credential preview"):
            save_env_value("EXA_API_KEY", preview)
        assert load_env()["EXA_API_KEY"] == "real-secret-value"
    assert env_path.read_bytes() == before
    assert "redacted credential preview" in REDACTED_CREDENTIAL_WRITE_DETAIL


def test_real_value_and_clearing_still_save(tmp_path):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        save_env_value("EXA_API_KEY", "new-real-secret")
        assert load_env()["EXA_API_KEY"] == "new-real-secret"
        save_env_value("EXA_API_KEY", "")
        assert load_env().get("EXA_API_KEY", "") == ""
