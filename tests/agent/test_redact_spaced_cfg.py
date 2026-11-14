"""Spaced ``key = value`` config assignments (#130827): config secrets masked, source code kept."""

import pytest

from agent.redact import redact_for_egress, redact_sensitive_text, redact_terminal_output


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


@pytest.mark.parametrize("text, secret", [
    ('password = "hunter2"\n', "hunter2"),                       # the issue repro
    ("[database]\npassword = 'x9Kp2LmQ'\n", "x9Kp2LmQ"),           # INI, single-quoted
    ('[server]\ndb.password = "S3cretVal"  # prod\n', "S3cretVal"),  # TOML dotted + comment
    ('  5|api_token = "abc123def456"\r\n', "abc123def456"),       # read_file gutter, CRLF
    ("client_secret = a1b2c3d4e5f6a7b8c9d0e1f2\n", "a1b2c3d4e5f6a7b8c9d0e1f2"),  # unquoted opaque
])
def test_spaced_config_secret_masked(text, secret):
    assert secret not in redact_for_egress(text)
    assert secret not in redact_terminal_output(text, "cat ~/.hermes/.env")


def test_spaced_quoted_value_with_space_masked_whole():
    # The adjacent form's value stops at whitespace; a quoted spaced value must not leak its tail.
    out = redact_sensitive_text('password = "hunter 2 horse"\n')
    assert "hunter" not in out and "horse" not in out


@pytest.mark.parametrize("line", [
    'password = getpass.getpass("Password: ")',
    "auth = HTTPBasicAuth(user, password)",
    "credentials = session.get_credentials()",
    "self.password = password",
    "this.secret = secret;",
    "  auth = None",
    '  password = ""',
    "  password = input()",
    "token = wintypes.HANDLE()",
    "auth_store = _load_auth_store()",
    "api.token = self._access_token",
    "password = os.environ",
    'password = os.environ["DB_PASSWORD"]',
    "secret = null",
    "token = tokens[0]",
    "auth = a or b",
    'auth_type = "oauth"',
    'auth = "header"',
    'no_token_msg = "no Codex OAuth token found (run: hermes model)"',
    "password => value",
    "password =~ /re/",
])
def test_spaced_source_code_unchanged(line):
    text = f"```python\n{line}\n```\n"
    assert redact_sensitive_text(text) == text
    assert redact_for_egress(text) == text
