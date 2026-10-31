"""Database-client credentials the word-boundary key test could not see.

``PGPASSWORD`` / ``MYSQL_PWD`` / ``SSHPASS`` glue the keyword onto a prefix with no
separator, so ``export PGPASSWORD=…``, ``env`` dumps, ``docker run -e`` and systemd
``Environment=`` lines leaked verbatim. libpq's space-delimited conninfo
(``psql "host=… password=…"``) had no ``;`` and skipped the connection-string pass.
"""

import pytest

from agent.redact import redact_sensitive_text, redact_terminal_output

SECRET = "Zq7xLmP2vR9kTw4s"
SHORT = "hunter2x"  # below the opaque-value length bar: only the key can flag it


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


@pytest.mark.parametrize("secret", [SECRET, SHORT])
@pytest.mark.parametrize("template", [
    "PGPASSWORD={s} psql -h 10.0.0.5 -U app",
    "export PGPASSWORD='{s}'",
    'PGPASSWORD="{s}"',
    "PGHOST=10.0.0.5\nPGPASSWORD={s}\nPGUSER=app",
    "docker run -e PGPASSWORD={s} postgres:17",
    "Environment=PGPASSWORD={s}",
    "$env:PGPASSWORD = '{s}'",
    "PGPASSWORD: {s}",
    '{{"PGPASSWORD": "{s}"}}',
    "MYSQL_PWD={s} mysql -u app",
    "SSHPASS={s} sshpass -e ssh host",
])
def test_glued_env_names_masked(template, secret):
    out = redact_sensitive_text(template.format(s=secret))
    assert secret not in out, out


@pytest.mark.parametrize("text", [
    "PWD=/home/app/project",
    "OLDPWD=/home/app",
    "PGPASSFILE=/home/app/.pgpass",
    "PGPASSWORD=$DB_PASSWORD psql",
    "PGPASSWORD=${DB_PASSWORD} psql",
])
def test_paths_and_references_untouched(text):
    assert redact_sensitive_text(text) == text


@pytest.mark.parametrize("text", [
    f"psql 'host=10.0.0.5 user=app password={SECRET} dbname=core'",
    f'psql "host=10.0.0.5 port=5432 dbname=core user=app password={SECRET}"',
    f"host=10.0.0.5 user=app password={SECRET}",
    f"psql 'hostaddr=10.0.0.5 dbname=core password=\"{SECRET}\"'",
])
def test_libpq_conninfo_masked(text):
    out = redact_sensitive_text(text)
    assert SECRET not in out, out


def test_libpq_conninfo_keeps_other_fields():
    text = f"psql 'host=10.0.0.5 user=app password={SECRET} dbname=core sslmode=require'"
    assert redact_sensitive_text(text) == (
        "psql 'host=10.0.0.5 user=app password=*** dbname=core sslmode=require'")


@pytest.mark.parametrize("text", [
    "set password= in the settings form",
    "the password=field is required",
])
def test_prose_without_companion_key_untouched(text):
    assert redact_sensitive_text(text) == text


@pytest.mark.parametrize("command", ["env | grep PG", "printenv", "cat .env"])
def test_terminal_output_env_dump(command):
    out = redact_terminal_output(f"PGHOST=10.0.0.5\nPGPASSWORD={SHORT}\nPGUSER=app\n", command)
    assert SHORT not in out, out


def test_terminal_output_libpq_conninfo_any_command():
    # The connection-string pass runs on all terminal output, not only env dumps.
    out = redact_terminal_output(
        f"connecting with host=10.0.0.5 user=app password={SECRET} dbname=core\n", "python app.py")
    assert SECRET not in out, out


def test_idempotent():
    once = redact_sensitive_text(f"export PGPASSWORD={SECRET}")
    assert redact_sensitive_text(once) == once


@pytest.mark.parametrize("text, expected", [
    # libpq has no ``;`` delimiter: it belongs to the value.
    (f"host=db password={SECRET[:6]};{SECRET[6:]} dbname=core", "host=db password=*** dbname=core"),
    # Backslash escapes a space inside an unquoted value.
    (f"host=db password={SECRET[:6]}\\ {SECRET[6:]} dbname=core", "host=db password=*** dbname=core"),
    (f"host=db password='{SECRET[:6]}\\' {SECRET[6:]}' dbname=core", "host=db password='***' dbname=core"),
])
def test_libpq_value_grammar_leaves_no_tail(text, expected):
    assert redact_sensitive_text(text) == expected
