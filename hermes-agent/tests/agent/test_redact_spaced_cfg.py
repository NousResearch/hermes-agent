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
    ('  5|password = "abc123def456"\r\n', "abc123def456"),        # read_file gutter, CRLF
    ("secret = a1b2c3d4e5f6a7b8c9d0e1f2\n", "a1b2c3d4e5f6a7b8c9d0e1f2"),  # unquoted opaque
    ("[openai]\nAPI Key = 9f8e7d6c5b4a39281706f5e4d3c2b1a0\n", "9f8e7d6c5b4a39281706f5e4d3c2b1a0"),  # spaced keyword
    ("service.API Key = a1b2c3d4e5f6a7b8c9d0e1f2\n", "a1b2c3d4e5f6a7b8c9d0e1f2"),  # spaced keyword, dotted
    ('api db.password = "hunter2"\n', "hunter2"),  # a dotted key may still start after a plain ``api ``
])
def test_spaced_config_secret_masked(text, secret):
    assert secret not in redact_for_egress(text)
    assert secret not in redact_terminal_output(text, "cat ~/.hermes/.env")


@pytest.mark.parametrize("key", ["password", "db_password", "DB_PASSWORD", "client_secret"])
def test_spaced_quoted_value_with_space_masked_whole(key):
    # The adjacent form's value stops at whitespace; a quoted spaced value must not leak its tail.
    out = redact_sensitive_text(f'{key} = "hunter 2 horse"\n')
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
    'password = "prefix-" + suffix',
    'secret = "${SECRET}"',
])
def test_spaced_source_code_unchanged(line):
    text = f"```python\n{line}\n```\n"
    assert redact_sensitive_text(text) == text
    assert redact_for_egress(text) == text


def test_spaced_separator_masked_in_dotted_quoted_and_export_forms():
    """INI/TOML files align values with spaces around ``=``; only the adjacent
    spelling was masked (the #16413 gap extended to ``password = x``)."""
    text = (
        "db.password = Sup3rS3cret!\n"
        "password = \"An0therS3cret!\"\n"
        "export password = Th1rdS3cret!\n"
    )
    result = redact_sensitive_text(text)
    assert "Sup3rS3cret!" not in result
    assert "An0therS3cret!" not in result
    assert "Th1rdS3cret!" not in result
    assert "db.password" in result  # keys preserved


def test_prose_mid_sentence_spaced_equals_unchanged():
    # Not line-anchored, not dotted: conversational text, leave alone (#4367).
    text = "I have password = foo and other things"
    assert redact_sensitive_text(text) == text


def test_spaced_equality_comparison_unchanged():
    # ``==`` is a comparison, not an assignment; spacing tolerance must not mask it.
    text = "password == expected\nassert cfg.password == other\n"
    assert redact_sensitive_text(text) == text


@pytest.mark.parametrize("text", [
    "password" * 1500 + "." * 1500 + " = ",  # `[..]*NAME[..]*` keys backtracked per start offset
    "a." + "password" * 6000 + " = ",
    "  5|" + "password" * 6000 + " = ",
    "api key." * 10000 + " = ",  # a key starting after every joined ``api `` rescanned the run
    "password" * 12000 + '. = "hunter2"',  # a valid value reaches the bare-key keyword check
], ids=["dotted-run", "dotted-prefix", "anchored-gutter", "dotted-api-key-joins", "bare-key-check"])
def test_spaced_pass_linear_on_long_keyword_runs(text):
    import time

    from agent.redact import _mask_token
    from agent.redact_spaced_cfg import redact_spaced_cfg_assignments

    start = time.perf_counter()
    redact_spaced_cfg_assignments(text, _mask_token)
    assert time.perf_counter() - start < 2.0
