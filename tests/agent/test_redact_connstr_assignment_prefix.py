"""A keyword/value connection string assigned to a variable is still a connection string.

``printenv``, a systemd ``Environment=CONN=…`` line and a ``docker run -e CONN=…`` argv dump all
put the connection string on the right of an ``=``. The companion-key boundary in
:mod:`agent.redact_connstr` only accepted start-of-line / whitespace / ``;`` / a quote, so the
``Server=`` key was not recognised and the ``Password=`` value went out verbatim.

The boundary is widened for the companion key only, and only on a ``;``-delimited line: a
``password=`` key behind ``=`` is usually prose, and an assigned ``host=`` in a space-separated
line is not a connection string.
"""

import pytest

from agent.redact import redact_sensitive_text, redact_terminal_output

PW = "s3cretpw7Qm2"


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


def _assert_masked(text: str) -> None:
    outputs = [redact_sensitive_text(text, force=True),
               redact_sensitive_text(text, force=True, file_read=True, secret_file=True),
               redact_terminal_output(text)]
    for out in outputs:
        assert PW not in out, out


class TestAssignedConnectionStrings:
    """The companion key sits behind ``NAME=``, not at a word boundary."""

    @pytest.mark.parametrize("text", [
        f"DB_CONN=Server=db;Database=app;User Id=app;Password={PW};",
        f"CONN_STR=Host=db;Username=app;Pwd={PW};Database=x",
        f"Environment=CONN=Server=db;Password={PW}",
        f"CONN=Host=db;Password={PW};",                       # printenv output
        f"docker run -e CONN=Host=db;Password={PW}; app",
        f"root 42 0.1 docker run -e CONN=Host=db;Password={PW}; app",
    ])
    def test_masked(self, text):
        _assert_masked(text)

    @pytest.mark.parametrize("text", [
        f"X=host=db;Encrypt=true;Password={PW};",
        f"X=host=db;CustomOption=value;Password={PW};",
        f"X=host=db=primary;Password={PW};",
        f'X=host="db;primary";Password={PW};',
        f"X=host={{db;primary}};Password={PW};",
        f"X=host=db;Password={PW}==;",
        f'X=host=db;Password="{PW};inside";',
        f"X=host=db;Password={{{PW};inside}};",
        f"X=host=db;Encrypt=true;Password={PW};\r\n",
        f"X=host=db;Encrypt=true;Password={PW}",
        f"X=hOsT=db;EnCrYpT=true;pWd={PW};",
        f"X=host={'d' * 100_000};Encrypt=true;Password={PW};",
    ])
    def test_arbitrary_fields_and_value_grammars(self, text):
        _assert_masked(text)

    @pytest.mark.parametrize("text", [
        "X=host=db",                      # companion key, no password field
        "MODE=password=off",              # a password key with no companion: not a conninfo
        "PWD=/home/app/project",
        "the password= field",
        "the pwd= flag and host= flag",   # prose, still kept
    ])
    def test_non_credentials_kept(self, text):
        assert redact_sensitive_text(text, force=True) == text

    @pytest.mark.parametrize("text", [
        # Assigned companion key, but no ``;`` between it and the password key: not one
        # connection string, so the shell's PWD must not be masked as a password.
        "env X=host=db PWD=/home/app/project myapp",
        # A password key behind ``=`` is prose; the ``;port=`` field alone must not turn the
        # sentence into a conninfo value.
        "X=host=db note=password=field and all subsequent ordinary prose;port=reference",
        # Another variable's value carries the ``;``-delimited field; it must not pair with the
        # assigned ``host=`` and swallow PWD.
        "env X=host=db PWD=/home/app/project OPTIONS=';sslmode=require' myapp",
        # Same, with the ``;``-carrying value before PWD.
        "env X=host=db OPTIONS=';sslmode=require' PWD=/home/app/project myapp",
        # Semicolon-prefixed prose next to an assigned companion: the space after ``=`` still
        # says "field", not "empty value followed by the secret".
        "X=host=db; password= field",
        "the conn string X=host=db; pwd= flag",
    ])
    def test_assigned_forms_do_not_widen_prose(self, text):
        assert redact_sensitive_text(text, force=True) == text
        assert redact_terminal_output(text, command="env") == text

    @pytest.mark.parametrize("text", [
        # A ``;`` inside another variable's quoted value is not this connection string's field
        # separator, however the quote is spelled.
        'env X=host=db NOTE=";password=off"',
        "env X=host=db NOTE=';password=off'",
        'env X=host=db NOTE=";Password=open";other',
    ])
    def test_assigned_forms_do_not_cross_quoting_contexts(self, text):
        assert redact_sensitive_text(text, force=True) == text
        assert redact_terminal_output(text, command="env") == text

    @pytest.mark.parametrize("text", [
        # The whole conninfo is quoted, so companion and password share one quoting context.
        f"export CONN='Host=db;Password={PW};'",
        f'export CONN="Host=db;Password={PW};"',
        f"export CONN='Server=db;User Id=app;Password={PW}'",
    ])
    def test_quoted_assignment_is_still_masked(self, text):
        _assert_masked(text)
