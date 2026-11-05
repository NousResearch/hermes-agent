"""Credential shapes that leaked past agent.redact (#53741, #125680, #134475, #141283, #147240).

Each class pins one family: the leak is masked in both the default (head/tail) mask and the
non-reusable ``file_read`` + ``secret_file`` sentinel, and the look-alike identifiers or docs
samples next to it stay byte-identical.
"""

import pytest

from agent.redact import redact_sensitive_text

V = "q7Zr4Lm9Xc2Vb5Nw8Kd3Hf6J"
SK = "sk-" + "q7Zr4Lm9Xc2Vb5Nw8Kd3Hf6Jt1Pa0Sg7Ye4Uo2"
GH = "ghp_" + "Ab3dE5gH7jK9mN1pQ3sT5vX7zB9dF1hJ3kL5"


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


def _both_modes(text: str) -> list[str]:
    return [redact_sensitive_text(text, force=True),
            redact_sensitive_text(text, force=True, file_read=True, secret_file=True)]


def _assert_masked(text: str, secret: str) -> None:
    for out in _both_modes(text):
        assert secret not in out, out


def _assert_unchanged(text: str) -> None:
    assert _both_modes(text) == [text, text]


class TestQuotedYamlScalars:
    """#53741 / #147240 family 1: ``api_key: "v"`` skipped the YAML pass on its quote."""

    @pytest.mark.parametrize("text", [f'api_key: "{V}"', f"api_key: '{V}'", 'password: "hunter2"',
                                      f'  5|  secret: "{V}"'])
    def test_quoted_value_masked(self, text):
        _assert_masked(text, V if V in text else "hunter2")

    @pytest.mark.parametrize("text", ['api_key: "<your key>"', 'api_key: "${API_KEY}"', 'password: ""',
                                      'Secretary: "J. Smith"', "api_key: 'process.env.OPENAI_API_KEY'"])
    def test_placeholders_and_prose_kept(self, text):
        _assert_unchanged(text)


class TestConcatenatedPasswordKeys:
    """#134475: ``dbpassword`` / ``rootpasswd`` never put the keyword on a word start."""

    @pytest.mark.parametrize("text, secret", [("dbpassword: hunter2xyz", "hunter2xyz"),
                                              ("rootpasswd: s3cr3tzz", "s3cr3tzz"),
                                              ("app.dbpassword=hunter2xyz", "hunter2xyz")])
    def test_masked(self, text, secret):
        _assert_masked(text, secret)

    @pytest.mark.parametrize("text", ["passwordless: true", "passwordless=true"])
    def test_passwordless_kept(self, text):
        _assert_unchanged(text)


class TestDatabaseClientCredentials:
    """#141283: keyword/value connection strings and glued client env names."""

    @pytest.mark.parametrize("text, secret", [
        ("Host=db;Username=app;Password=s3cretpw;Database=x", "s3cretpw"),
        ("Server=db;UID=app;PWD={Zq7x;LmP2};Database=core", "Zq7x;LmP2"),
        ('psql "host=db user=app password=s3cretpw dbname=x"', "s3cretpw"),
        ("PGPASSWORD=s3cretpw psql -h db", "s3cretpw"),
        ("MYSQL_PWD=s3cretpw", "s3cretpw"),
        ("SSHPASS=s3cretpw sshpass -e ssh host", "s3cretpw"),
    ])
    def test_masked(self, text, secret):
        _assert_masked(text, secret)

    @pytest.mark.parametrize("text", ["PWD=/home/app/project", "PGPASSFILE=/home/app/.pgpass",
                                      "the password= field", "host=db port=5432 dbname=x",
                                      "Data Source=db;Initial Catalog=x;Integrated Security=true"])
    def test_non_credentials_kept(self, text):
        _assert_unchanged(text)


class TestUserinfoAndBasicPairs:
    """#125680: ``scheme://user:pass@`` userinfo, and ``NAME=user:pass`` at force boundaries."""

    @pytest.mark.parametrize("text, expected", [
        ("https://user:s3cretpw@host/path", "https://user:***@host/path"),
        ("git remote set-url origin https://me:hunter2pass@github.com/o/r.git",
         "git remote set-url origin https://me:***@github.com/o/r.git"),
        ("MY_PROXY_BASIC=user:s3cretpw", "MY_PROXY_BASIC=user:***"),
    ])
    def test_masked(self, text, expected):
        assert redact_sensitive_text(text, force=True) == expected
        assert "s3cretpw" not in redact_sensitive_text(text, force=True, file_read=True, secret_file=True)

    @pytest.mark.parametrize("text", [
        "https://example.com/path", "git@github.com:org/repo.git", "ssh user@host", "user@example.com",
        "postgres://user@host/db", "https://user:${TOKEN}@host/x", "https://api.x.com/cb?code=abc123&s=1",
        "IMAGE=ubuntu:24.04-noble", "TAG=myrepo:latest-release", "SRC=C:\\Users\\me",
        "CHECKSUM=sha256:" + "ab12" * 16, "TS=2026-01-01T10:00:00", "HOST=localhost:5432",
    ])
    def test_non_credentials_kept(self, text):
        _assert_unchanged(text)


class TestAffixedJsonKeys:
    """#147240 family 2: JSON keys containing a secret word outside the closed key list."""

    @pytest.mark.parametrize("key", ["API_TOKEN", "DB_PASSWORD", "client_secret", "my_api_key_value"])
    def test_masked(self, key):
        _assert_masked(f'{{"{key}": "{V}"}}', V)

    @pytest.mark.parametrize("text", ['{"token_count": 12}', '{"password_policy": "strict"}',
                                      '{"secret_name": "prod-db-creds"}', '{"API_TOKEN": "<your token>"}',
                                      '{"secretary": "Jane Smith"}', '{"tokenizer": "cl100k_base"}'])
    def test_metadata_kept(self, text):
        _assert_unchanged(text)


class TestPrefixedKeyAfterSeparator:
    """#147240 family 3: a known-prefix key glued to an identifier by ``-`` / ``_``."""

    @pytest.mark.parametrize("text, secret", [(f"backup-{SK}.txt", SK[3:]), (f"backup_{SK}.txt", SK[3:]),
                                              (f"x-{GH}", GH[4:])])
    def test_masked(self, text, secret):
        _assert_masked(text, secret)

    @pytest.mark.parametrize("text", ["backup-sk-learn.txt", "scikit-learn", "x-request-id",
                                      "deploy-sk-prod-deployment-01", "build_hf_transformers_cache"])
    def test_identifiers_kept(self, text):
        _assert_unchanged(text)
