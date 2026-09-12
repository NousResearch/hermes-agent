"""PUT /api/config must refuse a defaults-built record overwriting a populated config.

A Desktop connection switch can write the settings record read from backend A to
backend B (#109357): every hand-tuned leaf on B then arrives holding A's factory
default, and the deep-merge write destroys B's live configuration wholesale. The
guard fails closed when explicitly-set leaves regress to factory defaults across
several top-level sections at once — a legitimate save never has that shape
(0 regressions; a per-tab form reset stays inside one section).
"""

import pytest


class TestDefaultsRegressedSectionsHelper:
    """Unit tests for the pure comparison helper."""

    def _sections(self, existing, incoming):
        from hermes_cli.web_routers.config_env import _defaults_regressed_sections
        return _defaults_regressed_sections(existing, incoming)

    def test_no_regressions_when_values_match_disk(self):
        existing = {"agent": {"max_turns": 50, "gateway_timeout": 1800}}
        incoming = {"agent": {"max_turns": 50, "gateway_timeout": 1800}}
        assert self._sections(existing, incoming) == set()

    def test_explicit_yaml_none_on_disk_is_not_a_user_choice(self):
        # ``gateway_timeout:`` (YAML empty) must not read as a deliberate
        # override of the 1800 default, else a normal defaulted record
        # would count it as a regression.
        existing = {"agent": {"gateway_timeout": None}}
        incoming = {"agent": {"gateway_timeout": 1800}}
        assert self._sections(existing, incoming) == set()

    def test_leaf_omitted_by_incoming_is_not_a_regression(self):
        # Deep-merge semantics: an omitted leaf keeps the disk value.
        existing = {"agent": {"max_turns": 50}, "session": {"terminal_continue": False}}
        incoming = {"agent": {"max_turns": 50}}
        assert self._sections(existing, incoming) == set()

    def test_scalar_default_vs_dict_disk_never_mismatches(self):
        # DEFAULT_CONFIG ships ``model`` as a string; a configured host stores a
        # dict. The paths never align, so the model section is simply not
        # comparable and must not crash or fabricate regressions.
        existing = {"model": {"default": "x", "provider": "openai"}}
        incoming = {"model": ""}
        assert self._sections(existing, incoming) == set()

    def test_sections_collected_per_top_level_key(self):
        existing = {
            "agent": {"max_turns": 50},
            "session": {"terminal_continue": False},
            "database": {"journal_mode": "delete"},
        }
        incoming = {
            "agent": {"max_turns": None},
            "session": {"terminal_continue": True},
            "database": {"journal_mode": "wal"},
        }
        assert self._sections(existing, incoming) == {"agent", "session", "database"}


class TestPutConfigDefaultsRegressionGuard:
    """API-level tests: the cross-machine defaults write is refused, everything
    a legitimate client does still saves."""

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")

        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    def _seed_configured_host(self):
        """Three sections of deliberate non-default values, like a hand-tuned host."""
        from hermes_cli.config import save_config

        save_config({
            "agent": {"max_turns": 50, "gateway_timeout": 900},
            "session": {"terminal_continue": False},
            "database": {"journal_mode": "delete"},
        })

    def _defaults_record_resetting(self, *leaf_edits):
        """GET the live record (defaults + seeded values) then reset the seeded
        leaves back to factory defaults — the exact payload a pristine install's
        record would carry for this host."""
        record = self.client.get("/api/config").json()
        for dotted, section in leaf_edits:
            keys = dotted.split(".")
            node = record
            for key in keys[:-1]:
                node = node[key]
            node[keys[-1]] = section
        assert record["agent"]["max_turns"] is None
        return record

    def test_cross_machine_defaults_record_rejected_and_disk_untouched(self):
        from hermes_cli.config import read_raw_config

        self._seed_configured_host()
        record = self._defaults_record_resetting(
            ("agent.max_turns", None),
            ("agent.gateway_timeout", 1800),
            ("session.terminal_continue", True),
            ("database.journal_mode", "wal"),
        )

        resp = self.client.put("/api/config", json={"config": record})

        assert resp.status_code == 400
        assert "factory defaults" in resp.json()["detail"]
        # Fail closed: nothing was merged or saved.
        disk = read_raw_config()
        assert disk["agent"]["max_turns"] == 50
        assert disk["session"]["terminal_continue"] is False
        assert disk["database"]["journal_mode"] == "delete"

    def test_two_section_regression_still_saves(self):
        """Below the section limit (a user resetting one tab plus one stray key,
        or any two-section deliberate change) must keep saving normally."""
        from hermes_cli.config import load_config

        self._seed_configured_host()
        record = self._defaults_record_resetting(
            ("agent.max_turns", None),
            ("session.terminal_continue", True),
        )

        resp = self.client.put("/api/config", json={"config": record})

        assert resp.status_code == 200, resp.text
        assert load_config()["agent"]["max_turns"] is None
        assert load_config()["session"]["terminal_continue"] is True

    def test_normal_settings_save_unaffected(self):
        from hermes_cli.config import load_config

        self._seed_configured_host()
        record = self.client.get("/api/config").json()
        record["agent"]["max_turns"] = 75

        resp = self.client.put("/api/config", json={"config": record})

        assert resp.status_code == 200, resp.text
        assert load_config()["agent"]["max_turns"] == 75
        # Untouched deliberate settings survive the save.
        assert load_config()["agent"]["gateway_timeout"] == 900

    def test_explicit_defaults_reset_allowed_with_flag(self):
        from hermes_cli.config import load_config

        self._seed_configured_host()
        record = self._defaults_record_resetting(
            ("agent.max_turns", None),
            ("agent.gateway_timeout", 1800),
            ("session.terminal_continue", True),
            ("database.journal_mode", "wal"),
        )

        resp = self.client.put(
            "/api/config",
            json={"config": record, "allow_defaults_regression": True},
        )

        assert resp.status_code == 200, resp.text
        assert load_config()["agent"]["gateway_timeout"] == 1800

    def test_partial_update_omitted_sections_not_regressions(self):
        from hermes_cli.config import load_config

        self._seed_configured_host()

        resp = self.client.put(
            "/api/config", json={"config": {"agent": {"max_turns": 75}}}
        )

        assert resp.status_code == 200, resp.text
        assert load_config()["agent"]["max_turns"] == 75
        assert load_config()["session"]["terminal_continue"] is False
