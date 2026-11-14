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
    ("mysql://db/app?user=app&passwd=Zq7x9LmP2kk",
     "mysql://db/app?user=app&passwd=***"),
    ("postgresql://db/app?sslmode=verify-full&sslpassword=Zq7x9LmP2kk",
     "postgresql://db/app?sslmode=verify-full&sslpassword=***"),
    ("mysql://db/app?password=first1&password=second2",
     "mysql://db/app?password=***&password=***"),
    # ``'`` is legal in a query: an earlier value holding one must not end the scan.
    ("postgresql://db/app?application_name=O'Reilly&password=Zq7x9LmP2kk",
     "postgresql://db/app?application_name=O'Reilly&password=***"),
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
    # ...also when a DB URL with a query sits right before them in compact JSON or autolinks.
    '{"db":"postgresql://h/app?sslmode=require","login":"https://example.com/login?password=Zq7x9LmP2kk"}',
    "<postgresql://h/app?sslmode=require><https://example.com/login?password=Zq7x9LmP2kk>",
    # Templates a raw space or quote cuts short (Helm, GitHub Actions, f-strings).
    "postgresql://db/app?password={{ .Values.db.password }}",
    "postgres://localhost/test?password=${{ secrets.PG_PASSWORD }}",
    "f\"postgresql://h/app?password={cfg['pw']}\"",
])
def test_non_secrets_pass_through(text):
    assert redact_sensitive_text(text) == text


def test_repeated_db_schemes_without_query_scan_in_linear_time():
    import time

    # 300k chars: about 10 s if the run stops crossing ``://``, so a 2 s bound still catches it.
    text = "postgresql://h/" * (300_000 // 15)
    t0 = time.perf_counter()
    redact_sensitive_text(text, force=True)
    assert time.perf_counter() - t0 < 2.0
