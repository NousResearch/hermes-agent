"""Bitwarden Secrets Manager access tokens: the shape-matched redaction pass.

Split from test_redact.py so neither file crosses the 2,000-line cap; the fixtures are
built by concatenation so no contiguous token-shaped literal exists in the tree.
"""
import pytest

from agent.redact import redact_sensitive_text

class TestBitwardenSecretsManagerTokens:
    """BWS access tokens (``0.<uuid>.<client-secret>:<base64-key>``) carry no vendor prefix,
    and in the reporting shape no ``KEY=value`` context either: an ``echo`` that rendered
    ``"${VAR:+set}${VAR:-unset}"`` concatenated the live value directly onto ``set``, so the
    token reached tool output (and the transcript) unmasked. The matcher is provider-shaped
    (version ``0.``, hyphenated UUID, opaque secret, colon key), which is what lets one pass
    cover the raw, JSON-field and exception renderings of the same token too.
    """

    # NOTE: every fixture is built by concatenation, so no contiguous token-shaped literal
    # exists in this file (push protection blocks realistic credential literals).
    _UUID = "6b1f0c3a-9d47-4e52-8a66-1c3b7f2d5e90"
    _SECRET = "Qn4Wd8Zk1Ry6Tp3Mc7Vg2Lx9Hb5Js0Fr"
    _KEY = "wGCdNOZhPpujFi/nM+P5Ig=="
    _TOKEN = "0." + _UUID + "." + _SECRET + ":" + _KEY
    _ECHO_CMD = ('echo "BWS_ACCESS_TOKEN in shell env: '
                 '${BWS_ACCESS_TOKEN:+set}${BWS_ACCESS_TOKEN:-unset}"')

    @pytest.mark.parametrize("text, context", [
        pytest.param(f"BWS_ACCESS_TOKEN in shell env: set{_TOKEN}\n",
                     "BWS_ACCESS_TOKEN in shell env: set", id="set-prefixed"),
        pytest.param(f"{_TOKEN}\n", "", id="raw-token"),
        pytest.param('{"BWS_ACCESS_TOKEN": "' + _TOKEN + '"}\n',
                     '{"BWS_ACCESS_TOKEN": "', id="json-field"),
        pytest.param(
            "Traceback (most recent call last):\n"
            "  File \"bws_cli.py\", line 7, in <module>\n"
            "    client.secrets().list()\n"
            "RuntimeError: bws rejected access token " + _TOKEN + "\n",
            "RuntimeError: bws rejected access token ", id="exception-text"),
    ])
    def test_bws_access_token_masked_on_every_surface(self, text, context):
        from agent.redact import redact_terminal_output

        # Incident surface: `echo` output is source-preserving (code_file=True), so the
        # ENV/JSON/YAML assignment passes never run — the token shape must mask on its own.
        term = redact_terminal_output(text, self._ECHO_CMD)
        assert self._SECRET not in term, term
        assert self._KEY not in term, term
        assert not context or context in term, term  # the env-var NAME is not secret

        # File-read surface: the mask must be non-reusable (no head/tail chars), so an agent
        # that reads it cannot write a truncated-looking dead token back (#35519).
        read = redact_sensitive_text(text, file_read=True)
        assert self._SECRET not in read, read
        assert self._KEY not in read, read
        assert "«redacted" in read, read

    def test_bws_matcher_leaves_benign_dotted_and_uuid_text_alone(self):
        """Provider-shaped, not a generic ``0.``-led dotted sweep: versions, floats and a
        bare UUID (hyphenated or 32-hex) stay byte-identical."""
        from agent.redact import redact_terminal_output

        benign = (
            "version 0.13.2, ratio 0.5, offset 0.0\n"
            "commit 6b1f0c3a9d474e528a661c3b7f2d5e90\n"
            "secret id 6b1f0c3a-9d47-4e52-8a66-1c3b7f2d5e90\n"
        )
        assert redact_terminal_output(benign, "cat report.txt") == benign
        assert redact_sensitive_text(benign) == benign
