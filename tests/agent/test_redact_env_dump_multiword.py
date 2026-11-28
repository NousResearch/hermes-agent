"""env-dump multi-word secret values are masked to end of line."""
import pytest


class TestEnvDumpMultiWord:
    # ── env-dump multi-word values ──

    @pytest.mark.parametrize("line", [
        'N8N_MCP_AUTHORIZATION="Bearer live-token-should-not-leak"',
        "N8N_MCP_AUTHORIZATION=Bearer live-token-should-not-leak",
        'export N8N_MCP_AUTHORIZATION="Bearer live-token-should-not-leak"',
        "N8N_MCP_AUTHORIZATION='Bearer live-token-should-not-leak'",
    ])
    def test_env_dump_masks_multi_word_secret_value_to_eol(self, line):
        from agent.redact import redact_terminal_output
        red = redact_terminal_output(f"{line}\nPATH=/usr/bin\nHOME=/home/u", "env", force=True)
        assert "live-token-should-not-leak" not in red
        assert "N8N_MCP_AUTHORIZATION=***" in red
        assert "PATH=/usr/bin" in red and "HOME=/home/u" in red

    def test_env_file_read_masks_multi_word_secret_value(self):
        from agent.redact import redact_terminal_output
        red = redact_terminal_output('N8N_MCP_AUTHORIZATION="Bearer live-token-should-not-leak"\n', "cat .env", force=True)
        assert "live-token-should-not-leak" not in red

    def test_env_dump_keeps_rc_references_readable(self):
        from agent.redact import redact_terminal_output
        out = ("SSH_AUTH_SOCK=/run/user/1000/keyring/ssh\n"
               "export SSH_AUTH_SOCK=$(gpgconf --list-dirs agent-ssh-socket)\n"
               "XAUTHORITY=/run/user/1000/gdm/Xauthority\n"
               "GNOME_KEYRING_CONTROL=/run/user/1000/keyring\n"
               "MAX_TOKENS=100\n")
        assert redact_terminal_output(out, "env", force=True) == out

    @pytest.mark.parametrize("line", [
        # env/printenv print values literally: an escaped quote or a ``#`` does not end the value.
        'N8N_MCP_AUTHORIZATION="Bearer abc\\" live-token-should-not-leak"',
        "N8N_MCP_AUTHORIZATION=Bearer abc # live-token-should-not-leak",
        # A reference followed by more words is judged as the whole value, not by its first word.
        "API_TOKEN=$HOME/x live-token-should-not-leak",
        "N8N_MCP_AUTHORIZATION=$(cat /run/token) live-token-should-not-leak",
        "N8N_MCP_AUTHORIZATION=$(cat /run/token) $(echo live-token-should-not-leak)",
    ])
    def test_env_dump_masks_whole_value_past_quote_comment_and_reference_heads(self, line):
        from agent.redact import redact_terminal_output
        red = redact_terminal_output(f"{line}\nPATH=/usr/bin", "printenv", force=True)
        assert "live-token-should-not-leak" not in red
        assert "PATH=/usr/bin" in red

    @pytest.mark.parametrize("line", [
        'N8N_MCP_AUTHORIZATION="Bearer abc\\" live-token-should-not-leak"',  # escaped quote is not the end
        'N8N_MCP_AUTHORIZATION="Bearer abc" live-token-should-not-leak',      # non-comment tail: ambiguous
        'N8N_MCP_AUTHORIZATION="Bearer live-token-should-not-leak',           # unterminated quote
    ])
    def test_dotenv_read_masks_ambiguous_quoting_to_eol(self, line):
        from agent.redact import redact_terminal_output
        red = redact_terminal_output(f"{line}\nPATH=/usr/bin", "cat .env", force=True)
        assert "live-token-should-not-leak" not in red
        assert "PATH=/usr/bin" in red

    def test_non_env_command_leaves_multi_word_assignment_alone(self):
        from agent.redact import redact_terminal_output
        # The to-end-of-line pass is opt-in: source/prose output is not an env dump.
        text = 'DESCRIPTION_AUTH_NOTE="see the docs for more"'
        assert redact_terminal_output(text, "python app.py", force=True) == text



_KEY_BODY = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKj"


@pytest.mark.parametrize("command", ["env", "cat .env", "printenv"])
@pytest.mark.parametrize("quote", ['"', "'", ""])
@pytest.mark.parametrize("header", ["PRIVATE KEY", "RSA PRIVATE KEY", "OPENSSH PRIVATE KEY", "PGP PRIVATE KEY BLOCK"])
def test_multiline_private_key_in_env_dump_is_masked_whole(command, quote, header):
    # The env-line pass must not consume the BEGIN marker before the key-block pass runs,
    # or the body lines below it would leak.
    from agent.redact import redact_terminal_output
    text = (f"SSH_PRIVATE_KEY={quote}-----BEGIN {header}-----\n{_KEY_BODY}\n{_KEY_BODY[::-1]}\n"
            f"-----END {header}-----{quote}\nPATH=/usr/bin\n")
    out = redact_terminal_output(text, command, force=True)
    assert _KEY_BODY not in out and _KEY_BODY[::-1] not in out
    assert "PATH=/usr/bin" in out
