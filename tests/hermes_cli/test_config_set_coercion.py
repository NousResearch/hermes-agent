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
    import yaml
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
