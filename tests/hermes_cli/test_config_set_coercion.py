"""Regression tests for `config set` value coercion + key validation (round-3 CFG).

Covers:
- CFG-02: negative / whitespace-padded numerics coerce to int/float on
  numeric-typed keys (old code used str.isdigit() and stored strings).
- CFG-05: null/none/~ coerce to None so a nullable field can be cleared.
- CFG-04: malformed dotted keys with empty segments are rejected.
- Guard: string-typed enum keys (approvals.mode) are NOT coerced.
"""

import pytest

from hermes_cli import config as cfg


def _read(tmp_path, *path):
    """Read a nested value straight from the on-disk config.yaml."""
    import hermes_yaml as yaml
    data = yaml.safe_load((tmp_path / "config.yaml").read_text()) or {}
    node = data
    for seg in path:
        node = node[seg]
    return node


class TestNumericCoercion:
    def test_negative_int(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg.set_config_value("agent.max_turns", "-5")
        v = _read(tmp_path, "agent", "max_turns")
        assert v == -5 and isinstance(v, int)

    def test_whitespace_padded_int(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg.set_config_value("agent.max_turns", " 42 ")
        v = _read(tmp_path, "agent", "max_turns")
        assert v == 42 and isinstance(v, int)

    def test_negative_float(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg.set_config_value("agent.max_turns", "-2.5")
        v = _read(tmp_path, "agent", "max_turns")
        assert v == -2.5 and isinstance(v, float)

    def test_lossy_decimal_identifier_stays_string(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        client_id = "123456789012.98765432109876"

        cfg.set_config_value("mcp_servers.example.oauth.client_id", client_id)

        saved = _read(
            tmp_path, "mcp_servers", "example", "oauth", "client_id"
        )
        assert saved == client_id
        assert isinstance(saved, str)


class TestNullCoercion:
    @pytest.mark.parametrize("token", ["null", "none", "None", "~"])
    def test_null_tokens_coerce_to_none(self, tmp_path, monkeypatch, token):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg.set_config_value("agent.run_budget_seconds", token)
        assert _read(tmp_path, "agent", "run_budget_seconds") is None


class TestMalformedKey:
    @pytest.mark.parametrize("bad", ["agent.", ".agent", "agent..max_turns", "  "])
    def test_empty_segment_rejected(self, tmp_path, monkeypatch, bad):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with pytest.raises(SystemExit) as exc:
            cfg.set_config_value(bad, "5")
        assert exc.value.code == 1


class TestStringTypedGuardPreserved:
    def test_enum_off_stays_string(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg.set_config_value("approvals.mode", "off")
        v = _read(tmp_path, "approvals", "mode")
        assert v == "off" and isinstance(v, str)  # not bool False


class TestQuotedStructuredLookingText:
    """The structured-value refusal hint says quoting stores a plain string (#134149):
    the quotes themselves must not be written, and newlines must survive — parsing the
    quoted literal with yaml would fold its newlines to spaces, so the outer quotes are
    stripped verbatim instead."""

    def test_quoted_multiline_colon_text_stored_verbatim(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        text = "This topic is X.\nStage: brainstorm and design"
        cfg.set_config_value("channel_prompts.1", f"'{text}'")
        assert _read(tmp_path, "channel_prompts", "1") == text

    def test_double_quoted_multiline_stored_verbatim(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        text = "This topic is X.\nStage: brainstorm"
        cfg.set_config_value("channel_prompts.2", f'"{text}"')
        assert _read(tmp_path, "channel_prompts", "2") == text

    def test_quoted_bracket_text_stored_without_quotes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg.set_config_value("channel_prompts.3", "'[text'")
        assert _read(tmp_path, "channel_prompts", "3") == "[text"

    def test_unquoted_structured_looking_text_still_refused(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with pytest.raises(SystemExit) as exc:
            cfg.set_config_value(
                "channel_prompts.4", "This topic is X.\nStage: brainstorm and design")
        assert exc.value.code == 1

    def test_bare_list_literal_still_parses(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg.set_config_value("custom_providers", "['alpha', 'beta']")
        saved = _read(tmp_path, "custom_providers")
        assert saved == ["alpha", "beta"]
