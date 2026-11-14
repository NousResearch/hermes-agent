"""Discord bot token redaction (#117848)."""

import base64
import logging

import pytest

from agent.redact import RedactingFormatter, redact_sensitive_text


class TestDiscordTokens:
    """Discord bot tokens (#117848): the adapter keeps one in every install's .env, and
    only the ``KEY=value`` shape used to be masked."""

    TOKEN = "MTIzNDU2Nzg5MDEyMzQ1Njc4.GxAbCd.7Zq1rJ8mN0pQ2sT4uV6wX8yA0bC2dE4f"
    MFA_TOKEN = "mfa.VkO_2G4Qv3T--NO--lWetW_tjND-toY_XyGMTCsGrhkGRlEjJ1nHPvKIhQEB"

    def test_bare_token_is_masked(self):
        result = redact_sensitive_text(f"connecting with {self.TOKEN}")
        assert self.TOKEN not in result
        assert "7Zq1rJ8mN0pQ" not in result

    def test_json_value_is_masked(self):
        """The reported gap: masked as an assignment, verbatim as a JSON value."""
        result = redact_sensitive_text(f'{{"DISCORD_BOT_TOKEN": "{self.TOKEN}"}}')
        assert self.TOKEN not in result

    def test_mfa_form_is_masked(self):
        result = redact_sensitive_text(self.MFA_TOKEN)
        assert self.MFA_TOKEN not in result

    def test_shape_alone_does_not_match(self):
        """A long dotted CamelCase identifier fits the token's shape; only a first
        segment that decodes to a Discord snowflake counts, so code survives intact."""
        for benign in (
            "MyVeryLongModuleNameHere.Config.SomeExtremelyLongAttributeName_value",
            "org.apache.commons.lang3.StringUtils.isNotBlank(someVariableName)",
            "Mozilla.Firefox.Nightly.channel.release.update.manifest.checksum.value",
            "NetworkManagerConfiguration.Setting.SomeExtremelyLongAttributeValue",
            "OrganizationalUnitSettings.Values.AnotherLongAttributeNameHere_x",
        ):
            assert redact_sensitive_text(benign) == benign

    # The first character is the base64 of the id's leading digit: 0-3 -> M, 4-7 -> N,
    # 8-9 -> O. Ids 4xxx..9xxx (every bot created since ~2020) give N/O tokens.
    # Built from the ids rather than written out, as a real token encodes its bot's id.
    ID_TOKENS = {
        lead: ".".join((base64.b64encode(f"{lead}12345678901234567".encode()).decode().rstrip("="),
                        "GxAbCd", "7Zq1rJ8mN0pQ2sT4uV6wX8yA0bC2dE4fGh"))
        for lead in "12479"
    }

    @pytest.mark.parametrize("lead", ["1", "2", "4", "7", "9"])
    def test_every_id_prefix_is_masked_on_production_paths(self, lead):
        """terminal tool output, read_file content and log lines all mask the token,
        whichever of M/N/O its user id encodes to."""
        from agent.redact import redact_terminal_output

        token = self.ID_TOKENS[lead]
        term = redact_terminal_output(f'{{"token": "{token}"}}', "cat bot.json")
        read = redact_sensitive_text(f"token: {token}\n", file_read=True)
        record = logging.LogRecord("gateway", logging.INFO, "", 0, "connecting with %s", (token,), None)
        log = RedactingFormatter("%(message)s").format(record)

        for out in (term, read, log):
            assert token not in out
            assert "7Zq1rJ8mN0pQ" not in out
        assert "redacted-secret" in read  # file_read keeps the non-reusable sentinel

    def test_nearby_shapes_still_pass_through_terminal_output(self):
        """git shas, UUIDs and JWT-free dotted text starting N/O stay byte-identical."""
        from agent.redact import redact_terminal_output

        benign = (
            "commit 9fceb02d0ae598e95dc970b74767f19372d61af8\n"
            "id 550e8400-e29b-41d4-a716-446655440000\n"
            "NotADiscordTokenButLongEnough.Abcdef.ThisIsJustADottedIdentifierHere\n"
        )
        assert redact_terminal_output(benign, "git log") == benign

    @pytest.mark.parametrize("left", ["\\n", "\\t", "%0A", "\\u0020", "\x1b[31m", "\\x1b[0m"])
    def test_token_after_escape_spelling_is_masked(self, left):
        """JSON-dumped / URL-encoded / coloured output glues an escape's letter or digit to
        the token; it must still be masked."""
        result = redact_sensitive_text(f"line{left}{self.TOKEN}")
        assert "7Zq1rJ8mN0pQ" not in result
        assert result.startswith(f"line{left}")

    @pytest.mark.parametrize("tail", [".", "...", ". Next sentence.", ".\n", "\u2026"])
    def test_token_at_sentence_end_is_masked(self, tail):
        """A dot that ends the run is punctuation, not part of a dotted identifier."""
        for token in (self.TOKEN, self.MFA_TOKEN):
            result = redact_sensitive_text(f"Bot token: {token}{tail}")
            assert token.split(".")[-1][:12] not in result
            assert result.endswith(tail)

    def test_dotted_continuation_is_not_a_token(self):
        """``<shape>.more`` continues an identifier; neither form claims it."""
        for text in (f"{self.TOKEN}.attr", f"{self.MFA_TOKEN}.attr", f"x.{self.MFA_TOKEN}",
                     f"schema{self.MFA_TOKEN}"):
            assert redact_sensitive_text(text) == text

    @pytest.mark.parametrize("left", ["\\n", "%0A", "\\u0020", "\x1b[31m"])
    def test_mfa_after_escape_spelling_is_masked(self, left):
        result = redact_sensitive_text(f"line{left}{self.MFA_TOKEN}")
        assert self.MFA_TOKEN not in result
        assert result.startswith(f"line{left}")
