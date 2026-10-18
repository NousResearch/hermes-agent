"""Keyword/value connection strings (ADO.NET, Npgsql, ODBC, JDBC SQL Server).

The URL pass only knows ``scheme://user:pass@``; ``Host=…;Password=…`` leaked
verbatim through configmap dumps, ``cat appsettings.json`` and env listings.
"""

import pytest

from agent.redact import redact_sensitive_text, redact_terminal_output

SECRET = "Zq7xLmP2vR9kTw4s"


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


@pytest.mark.parametrize("text", [
    f"Host=10.0.0.5;Port=5432;Username=app;Password={SECRET};Database=core",
    f"Server=db;User Id=sa;Pwd={SECRET};",
    f"Data Source=db;Initial Catalog=x;password={SECRET}",
    f"jdbc:sqlserver://db:1433;user=sa;password={SECRET};encrypt=true",
    f'"Default": "Host=db;Username=app;Password={SECRET};Database=core"',
    f"ConnectionStrings__Default=Host=db;Username=app;Password = {SECRET} ;Pooling=true",
    f"  READ_DB: Host=db;Username=ro;PASSWORD={SECRET}",
])
@pytest.mark.parametrize("code_file", [False, True])
def test_password_value_masked(text, code_file):
    out = redact_sensitive_text(text, code_file=code_file)
    assert SECRET not in out, out
    assert "***" in out


def test_quoted_value_with_semicolon_leaves_no_tail():
    text = 'Host=db;Username=app;Password="Zq7x;LmP2""vR9";Database=core'
    assert redact_sensitive_text(text) == 'Host=db;Username=app;Password="***";Database=core'


def test_other_fields_untouched():
    text = f"Host=10.0.0.5;Port=5432;Username=app;Password={SECRET};Database=core"
    assert redact_sensitive_text(text) == "Host=10.0.0.5;Port=5432;Username=app;Password=***;Database=core"


def test_only_the_connection_string_line_is_touched():
    prose = "the password= field is required; see docs"
    out = redact_sensitive_text(f"line one\nHost=db;Username=app;Password={SECRET}\n{prose}")
    assert SECRET not in out
    assert out.endswith(prose)


@pytest.mark.parametrize("text", [
    "the password= field is required; see docs",
    "a=1;b=2;c=3",
    "Password=hunter2",                # no companion key, no ';'
    "mysql --password=x; echo done",   # CLI flag, no companion key
])
def test_non_connection_strings_untouched(text):
    assert redact_sensitive_text(text, code_file=True) == text


@pytest.mark.parametrize("tmpl", ["{0}", "{password}", "${DB_PASSWORD}"])
def test_code_file_templates_preserved(tmpl):
    text = 'var cs = $"Host={host};Username=app;Password=' + tmpl + ';Database=core";'
    assert redact_sensitive_text(text, code_file=True) == text


@pytest.mark.parametrize("cmd", ["cat k8s/configmap.yaml", "env", "grep -r Host= ."])
def test_terminal_output_path(cmd):
    text = f"data:\n  ConnectionStrings__Default: Host=db;Username=app;Password={SECRET};Database=core\n"
    assert SECRET not in redact_terminal_output(text, cmd)


def test_idempotent():
    once = redact_sensitive_text(f"Host=db;Username=app;Password={SECRET}")
    assert redact_sensitive_text(once) == once
