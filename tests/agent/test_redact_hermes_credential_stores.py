"""Hermes-home credential stores are secret-file args (#157915).

`cat ~/.hermes/auth.json` must classify like `cat ~/.hermes/config.yaml` —
the stores live in agent/redact_hermes_home.is_hermes_credential_store, over
the same inventory the file tools hard-deny, anchored to the Hermes home so a
project's own auth.json never matches.
"""

from agent.redact import _command_reads_secret_file, _is_secret_file_arg
from agent.redact import redact_terminal_output


def test_hermes_credential_stores_are_secret_file_args():
    # Anchored: under ~/.hermes or $HERMES_HOME -> credential dump.
    assert _is_secret_file_arg("~/.hermes/auth.json")
    assert _is_secret_file_arg("~/.hermes/.anthropic_oauth.json")
    assert _is_secret_file_arg("~/.hermes/auth/google_oauth.json")
    assert _is_secret_file_arg("$HERMES_HOME/auth.json")
    assert _is_secret_file_arg('"$HERMES_HOME/auth.json"')  # quotes stripped like cat sees them
    assert _command_reads_secret_file("cat ~/.hermes/auth.json")
    assert _command_reads_secret_file("grep access_token ~/.hermes/.anthropic_oauth.json")

    # A project's own auth.json is NOT a Hermes credential store — anchored, never bare.
    assert not _is_secret_file_arg("/home/user/myapp/auth.json")
    assert not _command_reads_secret_file("cat /home/user/myapp/auth.json")


def test_cat_of_hermes_auth_json_output_is_redacted():
    out = 'agent said: {"provider": "openrouter", "api_key": "sk-or-v1-abc123def456ghi789jkl"}\n'
    redacted = redact_terminal_output(out, "cat ~/.hermes/auth.json")
    assert "sk-or-v1-abc123def456ghi789jkl" not in redacted
