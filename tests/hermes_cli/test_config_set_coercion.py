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


class TestConfigSetRoundTripSafety:
    def test_bracketed_list_index_segment_is_rejected_without_writing(self, tmp_path, monkeypatch):
        """Bracket syntax is not a config-set path syntax and must not create a literal key."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_path = tmp_path / "config.yaml"
        original = "custom_providers:\n  - name: existing # keep\n"
        config_path.write_text(original, encoding="utf-8")

        with pytest.raises(SystemExit) as exc:
            cfg.set_config_value("custom_providers[0].name", "changed")

        assert exc.value.code == 1
        assert config_path.read_text(encoding="utf-8") == original
        import yaml
        assert "custom_providers[0]" not in yaml.safe_load(original)

    def test_nested_update_preserves_trailing_comment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "model:\n  default: old-model # selected model\n  provider: openrouter # keep provider\n",
            encoding="utf-8",
        )

        cfg.set_config_value("model.default", "new-model")

        saved = config_path.read_text(encoding="utf-8")
        assert "default: new-model # selected model" in saved
        assert "provider: openrouter # keep provider" in saved

    def test_normal_nested_and_list_index_updates_still_succeed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "display:\n  verbose: false\n"
            "custom_providers:\n"
            "  - name: first\n    api_key: old\n"
            "  - name: second\n    api_key: keep\n",
            encoding="utf-8",
        )

        cfg.set_config_value("display.verbose", "true")
        cfg.set_config_value("custom_providers.0.api_key", "new")

        import yaml
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["display"]["verbose"] is True
        assert saved["custom_providers"] == [
            {"name": "first", "api_key": "new"},
            {"name": "second", "api_key": "keep"},
        ]

    def test_list_item_update_preserves_item_and_trailing_comments(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "fallback_providers:\n"
            "  - provider: nous  # keep first\n"
            "  - provider: other  # keep second\n"
            "# trailing\n",
            encoding="utf-8",
        )

        cfg.set_config_value("fallback_providers.1.provider", "changed")

        saved = config_path.read_text(encoding="utf-8")
        assert "provider: nous  # keep first" in saved
        assert "provider: changed" in saved
        assert "# keep second" in saved
        assert "# trailing" in saved
        import yaml
        assert yaml.safe_load(saved)["fallback_providers"] == [
            {"provider": "nous"},
            {"provider": "changed"},
        ]

    def test_unrelated_anchors_aliases_and_block_scalars_survive(self, tmp_path, monkeypatch):
        """A leaf update must not turn a hand-edited document into a fresh YAML dump."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "defaults: &defaults\n"
            "  retries: 3 # retain this\n"
            "custom_provider:\n"
            "  <<: *defaults\n"
            "  prompt: |\n"
            "    Keep this block scalar exactly as a block.\n"
            "display:\n"
            "  verbose: false\n",
            encoding="utf-8",
        )

        cfg.set_config_value("display.verbose", "true")

        saved = config_path.read_text(encoding="utf-8")
        assert "defaults: &defaults" in saved
        assert "<<: *defaults" in saved
        assert "prompt: |" in saved
        assert "    Keep this block scalar exactly as a block." in saved
        assert "retries: 3 # retain this" in saved
        assert _read(tmp_path, "display", "verbose") is True


class TestStringTypedGuardPreserved:
    def test_enum_off_stays_string(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg.set_config_value("approvals.mode", "off")
        v = _read(tmp_path, "approvals", "mode")
        assert v == "off" and isinstance(v, str)  # not bool False
