"""DB connection URLs that carry the password as a query parameter."""

import pytest

from agent.redact import redact_sensitive_text


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


@pytest.mark.parametrize("text, expected", [
    ("postgresql://db:5432/app?user=app&password=Zq7x9LmP2kk",
     "postgresql://db:5432/app?user=app&password=***"),
    ("jdbc:postgresql://db/app?password=Zq7x9LmP2kk&sslmode=require",
     "jdbc:postgresql://db/app?password=***&sslmode=require"),
    ("jdbc:mysql://db:3306/app?user=app&password=Zq7x9LmP2kk&useSSL=true",
     "jdbc:mysql://db:3306/app?user=app&password=***&useSSL=true"),
    ("url = 'jdbc:mariadb://db/app?user=app&PWD=Zq7x9LmP2kk'",
     "url = 'jdbc:mariadb://db/app?user=app&PWD=***'"),
    ("postgresql+psycopg2://db/app?user=app&password=Zq7x9LmP2kk",
     "postgresql+psycopg2://db/app?user=app&password=***"),
    ("mongodb+srv://cluster.example.net/app?password=Zq7x9LmP2kk",
     "mongodb+srv://cluster.example.net/app?password=***"),
    ("mysql://db/app?password=first1&password=second2",
     "mysql://db/app?password=***&password=***"),
])
def test_query_password_is_masked(text, expected):
    assert redact_sensitive_text(text) == expected


@pytest.mark.parametrize("text", [
    "postgresql://db/app?user=app&password=${DB_PASSWORD}",
    "postgresql://db/app?user=app&password=",
    "postgresql://db/app?user=app&passwordless=1",
    "postgresql://db/app?sslmode=require",
    # Web-URL query params stay untouched (magic links, OAuth callbacks).
    "https://example.com/login?user=app&password=Zq7x9LmP2kk",
])
def test_non_secrets_pass_through(text):
    assert redact_sensitive_text(text) == text


def test_repeated_db_schemes_without_query_scan_in_linear_time():
    import time

    text = "postgresql://h/" * (100_000 // 15)
    t0 = time.perf_counter()
    redact_sensitive_text(text, force=True)
    assert time.perf_counter() - t0 < 0.5
