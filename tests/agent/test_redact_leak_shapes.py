"""Credential shapes that leaked past hermes_agent.agent.redact (#53741, #125680, #134475, #141283, #147240).

Each class pins one family: the leak is masked in both the default (head/tail) mask and the
non-reusable ``file_read`` + ``secret_file`` sentinel, and the look-alike identifiers or docs
samples next to it stay byte-identical.
"""

import pytest

from hermes_agent.agent.redact import redact_sensitive_text, redact_terminal_output

V = "q7Zr4Lm9Xc2Vb5Nw8Kd3Hf6J"
SK = "sk-" + "q7Zr4Lm9Xc2Vb5Nw8Kd3Hf6Jt1Pa0Sg7Ye4Uo2"
GH = "ghp_" + "Ab3dE5gH7jK9mN1pQ3sT5vX7zB9dF1hJ3kL5"


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("hermes_agent.agent.redact._REDACT_ENABLED", True)


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

    def test_braced_odbc_password_masked_in_a_file_read(self):
        # ``cat`` output is read as code, where ``{pw}`` is a template; an ODBC brace is a literal.
        out = redact_terminal_output("Server=db;UID=app;PWD={Zq7x;LmP2};Database=core",
                                     "cat appsettings.json", force=True)
        assert "Zq7x" not in out
        # A brace literal that opens like an identifier still carries password characters.
        assert "ssw0rd" not in redact_terminal_output("Server=db;UID=app;PWD={P@ssw0rd!};Database=core",
                                                      "cat appsettings.json", force=True)
        for template in ("{0}", "{cfg['pw']}", "{pw!r}", "{ get_pw() }"):
            line = f"Server=db;PWD={template};Database=x"
            assert redact_terminal_output(line, "cat app.py", force=True) == line

    @pytest.mark.parametrize("text", ["PWD=/home/app/project", "PGPASSFILE=/home/app/.pgpass",
                                      "the password= field", "the pwd= flag and host= flag",
                                      "host=db port=5432 dbname=x",
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
        ('env = ["MY_PROXY_BASIC=user:s3cretpw", "X=1"]', 'env = ["MY_PROXY_BASIC=user:***", "X=1"]'),
    ])
    def test_masked(self, text, expected):
        assert redact_sensitive_text(text, force=True) == expected
        assert "s3cretpw" not in redact_sensitive_text(text, force=True, file_read=True, secret_file=True)

    @pytest.mark.parametrize("text", [
        "https://example.com/path", "git@github.com:org/repo.git", "ssh user@host", "user@example.com",
        "postgres://user@host/db", "https://user:${TOKEN}@host/x", "https://api.x.com/cb?code=abc123&s=1",
        "IMAGE=ubuntu:24.04-noble", "TAG=myrepo:latest-release", "SRC=C:\\Users\\me",
        "CHECKSUM=sha256:" + "ab12" * 16, "TS=2026-01-01T10:00:00", "HOST=localhost:5432",
        # Colon values under names that never hold a Basic pair, and userinfo docs samples.
        "IMAGE=redis:7-alpine", "BRANCH=origin:feature-login2", "Server=tcp:db,1433;Initial Catalog=x",
        'url = f"https://{user}:{pw}@host"', "The format is https://user:password@host",
        "PROXY_DEST=buildhost:Release2026/artifact.tar",
    ])
    def test_non_credentials_kept(self, text):
        _assert_unchanged(text)


class TestAffixedJsonKeys:
    """#147240 family 2: JSON keys containing a secret word outside the closed key list."""

    @pytest.mark.parametrize("key", ["API_TOKEN", "DB_PASSWORD", "client_secret", "my_api_key_value"])
    def test_masked(self, key):
        _assert_masked(f'{{"{key}": "{V}"}}', V)

    @pytest.mark.parametrize("text", [f'{{"output": "{{\\"DB_PASSWORD\\": \\"{V}\\"}}"}}',
                                      f'{{"DB_PASSWORD": "{V[:8]}\\\\{V[8:]}"}}'])
    def test_escaped_json_masked(self, text):
        # A tool result wrapping a config dump escapes its quotes; a value may carry escapes too.
        _assert_masked(text, V[8:])

    @pytest.mark.parametrize("text", ['{"token_count": 12}', '{"password_policy": "strict-min12"}',
                                      '{"passwordChanged": "2015-09-03T16:41:46.000Z"}', '{"has_password": "true"}',
                                      '{"tokenUrl": "https://login.example.com/oauth2/token"}',
                                      '{"secret_id": "3f2a9c1e-1b2c-4d5e-8f90-123456789abc"}',
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


class TestUserinfoStopsAtQuote:
    """A ``"`` ends a JSON string, so ``"https://a.com","u:pw@y"`` holds no ``user:pass@`` userinfo."""

    @pytest.mark.parametrize("text", ['["https://a.com","u:longpassword1@y"]',
                                      '"https://example.com","x@y.com"'])
    def test_adjacent_json_strings_stay_intact(self, text):
        _assert_unchanged(text)
        # Strict mode (memory context) masks every userinfo, and must stop at the quote too.
        assert redact_sensitive_text(text, redact_url_credentials=True) == text
        assert redact_sensitive_text('{"@context":"https://schema.org","@type":"Thing"}',
                                     redact_url_credentials=True) == '{"@context":"https://schema.org","@type":"Thing"}'

    def test_userinfo_password_still_masked(self):
        _assert_masked("https://user:FakeSecret9876@host/path", "FakeSecret9876")

class TestNetworkPathUserinfo:
    """#125664 residual: scheme-less ``//user:pass@host/share`` (SMB/CIFS mounts) under force."""

    @pytest.mark.parametrize("text", ["//user:FakeSecret9876@host/share",
                                      "mount -t cifs //user:FakeSecret9876@host/share /mnt",
                                      'src="//user:FakeSecret9876@host/share"', "`//user:FakeSecret9876@host`",
                                      "[//user:FakeSecret9876@host/a, //user:FakeSecret9876@host/b]"])
    def test_forced_network_path_password_masked(self, text):
        _assert_masked(text, "FakeSecret9876")
        assert "//user:***@host" in redact_sensitive_text(text, force=True)

    @pytest.mark.parametrize("text", ["// TODO: user:note@x", "//user:note@x", "//cdn.example.com/lib.js",
                                      '<a href="//example.com">', "a//user:FakeSecret9876@h",
                                      "//user:${SMB_PASSWORD}@host/share", "//user:%SMB_PASSWORD%@host/share"])
    def test_comments_relative_urls_and_placeholders_unchanged(self, text):
        _assert_unchanged(text)


class TestGoogleOAuthCredentials:
    """#64351: a pasted ``client_secret.json`` / OAuth access token logged as a chat preview leaked."""

    SECRET = "GOCSPX-" + "k1fQ7z_P2mX9vLk4n-Rt8wYb1cDf6"  # real bodies are base64url: carry "-" and "_"
    ACCESS = "ya29." + "a0AfH6SMBq7z_P2mX9vLk4n-Rt8wYb1cDf6gHj3sA0"
    SA_ACCESS = "ya29.c." + "c0ASRK0GaPq7z_P2mX9vLk4n-Rt8wYb1cDf6gHj3sA0"  # service account / GCE metadata
    STS_ACCESS = "ya29.d." + "c0AZ4bNpq7z_P2mX9vLk4n-Rt8wYb1cDf6gHj3sA0"  # workload identity federation (STS)

    @pytest.mark.parametrize("template", [
        "msg='client_secret\":\"{}\",\"redirect_uris\":[\"http://localhost\"]}}}}'",  # cut mid-JSON
        "here is my secret {} thanks", "[\"{}\"]", "Authorization: {}",
    ])
    def test_masked_without_a_secret_key_name(self, template):
        for secret in (self.SECRET, self.ACCESS, self.SA_ACCESS, self.STS_ACCESS):
            _assert_masked(template.format(secret), secret)

    def test_client_secret_fragment_cut_by_the_preview_is_masked(self):
        fragment = self.SECRET[:17]  # 80-char preview can end mid-body
        _assert_masked("msg='client_secret\":\"" + fragment, fragment)

    @pytest.mark.parametrize("separator", ["-", "_"])
    def test_client_secret_glued_to_identifier_is_masked(self, separator):
        # Full base64url secrets need masking even if no body run reaches 16 alphanumerics.
        _assert_masked(f"msg='archive{separator}{self.SECRET}'", self.SECRET)

    @pytest.mark.parametrize("separator", ["-", "_"])
    def test_access_token_glued_to_identifier_is_masked(self, separator):
        # Same shape for ya29.: separators every few chars leave no 16-alphanumeric run.
        token = "ya29.a0AfH6SM_q7z-P2mX9vL_k4n-Rt8wYb1c-Df6gHj3s_A0x"
        _assert_masked(f"msg='log{separator}{token}'", token)

    @pytest.mark.parametrize("text", ["GOCSPX-short", "ya29.short", "ya29.c.short", "notGOCSPX-" + "A" * 24, "mya29." + "A" * 24])
    def test_short_or_embedded_lookalikes_kept(self, text):
        _assert_unchanged(text)
