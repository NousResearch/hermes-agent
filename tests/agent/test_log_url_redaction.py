"""URL credentials must be masked on disk without changing tool-facing URLs."""

import logging
from contextlib import closing

import pytest

from agent import redact


@pytest.mark.parametrize(
    ("url", "secret", "public_part"),
    [
        ("https://user:LOG_PASS@example.invalid/v1", "LOG_PASS", "example.invalid/v1"),
        ("https://example.invalid/v1?token=LOG_TOKEN&model=demo", "LOG_TOKEN", "model=demo"),
        ("/v1?client%255Fsecret=LOG_ENCODED&model=demo", "LOG_ENCODED", "model=demo"),
        ("/v1?token=LOG_RELATIVE;model=demo", "LOG_RELATIVE", "model=demo"),
        ("//user:LOG_NETWORK@example.invalid/v1", "LOG_NETWORK", "example.invalid/v1"),
    ],
)
def test_file_log_masks_url_credentials_but_preserves_tool_urls(
    tmp_path, monkeypatch, url, secret, public_part
):
    monkeypatch.setattr(redact, "_REDACT_ENABLED", True)
    log_path = tmp_path / "agent.log"
    logger = logging.Logger("url-redaction-test", logging.DEBUG)
    with closing(logging.FileHandler(log_path, encoding="utf-8")) as handler:
        handler.setFormatter(redact.RedactingFormatter("%(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.warning("Endpoint probe failed: %s", url)
        handler.flush()

    logged = log_path.read_text(encoding="utf-8")
    assert secret not in logged
    assert public_part in logged
    assert "WARNING Endpoint probe failed:" in logged
    assert redact.redact_sensitive_text(url) == url


def test_exception_log_masks_credentials_without_losing_failure_context(tmp_path, monkeypatch):
    monkeypatch.setattr(redact, "_REDACT_ENABLED", True)
    url = "https://user:TRACE_PASS@example.invalid/v1?access_token=TRACE_TOKEN&model=demo"
    log_path = tmp_path / "errors.log"
    logger = logging.Logger("url-exception-test", logging.DEBUG)
    with closing(logging.FileHandler(log_path, encoding="utf-8")) as handler:
        handler.setFormatter(redact.RedactingFormatter("%(levelname)s %(message)s"))
        logger.addHandler(handler)
        try:
            raise RuntimeError("HTTP 401 from " + url)
        except RuntimeError:
            logger.exception("Provider metadata request failed")
        handler.flush()

    logged = log_path.read_text(encoding="utf-8")
    assert "TRACE_PASS" not in logged
    assert "TRACE_TOKEN" not in logged
    assert "RuntimeError: HTTP 401" in logged
    assert "example.invalid/v1" in logged
    assert "model=demo" in logged
    assert "Provider metadata request failed" in logged
