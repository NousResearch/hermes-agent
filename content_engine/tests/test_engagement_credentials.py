"""Tests for engagement credential configuration."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ce_dir = Path(__file__).resolve().parent.parent
if str(ce_dir) not in sys.path:
    sys.path.insert(0, str(ce_dir))

XURL_ENV = {
    "XURL_CLIENT_ID": "client-id-test",
    "XURL_CLIENT_SECRET": "client-secret-test",
    "XURL_CONSUMER_KEY": "consumer-key-test",
    "XURL_CONSUMER_SECRET": "consumer-secret-test",
}


def _set_xurl_env(monkeypatch):
    for name, value in XURL_ENV.items():
        monkeypatch.setenv(name, value)


def _mock_postiz_row(monkeypatch):
    """Mock the Postiz docker psql query so configure_xurl_from_postiz
    reaches the credential-resolution stage without a live DB."""
    import subprocess

    mock_result = MagicMock(returncode=0, stderr="")
    mock_result.stdout = "1|test-name|access-token:access-secret|rtok|Sahil_Saghir\n"
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: mock_result)


def test_configure_xurl_success(monkeypatch, tmp_path):
    """Writes config only from the four required environment variables."""
    _set_xurl_env(monkeypatch)
    xurl_path = tmp_path / ".xurl"
    _mock_postiz_row(monkeypatch)
    monkeypatch.setattr("os.path.expanduser", lambda path: str(xurl_path))

    from engagement_suggester import configure_xurl_from_postiz

    assert configure_xurl_from_postiz() is True
    config_text = xurl_path.read_text()
    for name, value in XURL_ENV.items():
        yaml_key = name.removeprefix("XURL_").lower()
        assert f"{yaml_key}: {value}" in config_text


@pytest.mark.parametrize("missing", tuple(XURL_ENV))
def test_configure_xurl_raises_when_any_required_credential_is_missing(monkeypatch, missing):
    """A missing credential must raise an explicit configuration error."""
    _set_xurl_env(monkeypatch)
    monkeypatch.delenv(missing)
    _mock_postiz_row(monkeypatch)

    from engagement_suggester import XUrlConfigurationError, configure_xurl_from_postiz

    with pytest.raises(XUrlConfigurationError):
        configure_xurl_from_postiz()


def test_configure_xurl_error_message_lists_required_vars(monkeypatch):
    """The raised error names every required variable, never a value."""
    monkeypatch.delenv("XURL_CONSUMER_SECRET", raising=False)
    _mock_postiz_row(monkeypatch)

    from engagement_suggester import XUrlConfigurationError, configure_xurl_from_postiz

    with pytest.raises(XUrlConfigurationError) as exc_info:
        configure_xurl_from_postiz()
    message = str(exc_info.value)
    for var in ("XURL_CLIENT_ID", "XURL_CLIENT_SECRET", "XURL_CONSUMER_KEY", "XURL_CONSUMER_SECRET"):
        assert var in message
    for secret_value in XURL_ENV.values():
        assert secret_value not in message


def test_no_credential_values_leak_into_stdout_when_config_present(monkeypatch, tmp_path, capsys):
    """Running configure-xurl with valid config must not print any credential value."""
    _set_xurl_env(monkeypatch)
    xurl_path = tmp_path / ".xurl"
    _mock_postiz_row(monkeypatch)
    monkeypatch.setattr("os.path.expanduser", lambda path: str(xurl_path))

    from engagement_suggester import configure_xurl_from_postiz

    assert configure_xurl_from_postiz() is True
    captured = capsys.readouterr().out
    for secret_value in XURL_ENV.values():
        assert secret_value not in captured
    assert "access-token:access-secret" not in captured
    assert "access-token" not in captured
    assert "access-secret" not in captured


def test_no_committed_xurl_credential_literals():
    """The xurl configuration path must not assign credential literals in source."""
    import engagement_suggester as module

    source = Path(module.__file__).read_text()
    assignment_literal = re.compile(
        r'(?m)^(?!\s*#).*\b(?:client_id|client_secret|consumer_key|consumer_secret)'
        r'\s*=\s*["\'][^"\']{6,}["\']'
    )
    rendered_config_literal = re.compile(
        r'(?m)^\s*(?:client_id|client_secret|consumer_key|consumer_secret):'
        r'\s*[A-Za-z0-9_-]{6,}\s*$'
    )
    # Also catch token-shaped literals that look like X OAuth pairs
    # ('<id>-<token>:<secret>') anywhere in the source, including comments.
    oauth_pair_literal = re.compile(
        r"\d{6,}-[A-Za-z0-9]{10,}:[A-Za-z0-9]{10,}"
    )
    assert assignment_literal.search(source) is None
    assert rendered_config_literal.search(source) is None
    assert oauth_pair_literal.search(source) is None
