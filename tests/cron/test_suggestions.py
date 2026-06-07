"""Tests for the Suggested Cron Jobs feature.

Covers the store (add/dedup/cap/accept/dismiss/latch), catalog seeding, the
recipe->suggestion bridge, and the shared command handler. Uses an isolated
HERMES_HOME so the real suggestions.json is never touched.
"""

import importlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A cron.suggestions module bound to an isolated HERMES_HOME."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Reload so module-level CRON_DIR/SUGGESTIONS_FILE pick up the temp home.
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.suggestions as s
    importlib.reload(s)
    return s


def _add(store, key="k1", title="Test", source="catalog", schedule="0 9 * * *"):
    return store.add_suggestion(
        title=title,
        description="desc",
        source=source,
        job_spec={"prompt": "do it", "schedule": schedule, "name": title, "deliver": "origin"},
        dedup_key=key,
    )


class TestStore:
    def test_add_and_list_pending(self, store):
        rec = _add(store)
        assert rec is not None
        pending = store.list_pending()
        assert len(pending) == 1
        assert pending[0]["title"] == "Test"
        assert pending[0]["status"] == "pending"

    def test_dedup_blocks_duplicate_pending(self, store):
        assert _add(store, key="dup") is not None
        assert _add(store, key="dup") is None  # same key already pending
        assert len(store.list_pending()) == 1

    def test_dismiss_latches_against_redisplay(self, store):
        _add(store, key="latch")
        assert store.dismiss_suggestion("1") is True
        assert store.list_pending() == []
        # Re-adding the same key is refused (never re-offer a dismissed one).
        assert _add(store, key="latch") is None

    def test_unknown_source_rejected(self, store):
        with pytest.raises(ValueError):
            store.add_suggestion(title="x", description="d", source="bogus", job_spec={}, dedup_key="k")

    def test_pending_cap(self, store):
        for i in range(store.MAX_PENDING):
            assert _add(store, key=f"k{i}") is not None
        # One past the cap is dropped.
        assert _add(store, key="over") is None
        assert len(store.list_pending()) == store.MAX_PENDING

    def test_accept_creates_job_and_marks_accepted(self, store):
        _add(store, key="acc", title="My Job")
        created = {}

        def fake_create_job(**kwargs):
            created.update(kwargs)
            return {"id": "job123", "name": kwargs.get("name"), **kwargs}

        with patch("cron.jobs.create_job", fake_create_job):
            job = store.accept_suggestion("1", origin={"platform": "telegram", "chat_id": "5"})

        assert job is not None
        assert created["schedule"] == "0 9 * * *"
        assert created["origin"] == {"platform": "telegram", "chat_id": "5"}
        # No longer pending.
        assert store.list_pending() == []
        # And accepting again is a no-op (not pending anymore).
        assert store.accept_suggestion("acc") is None

    def test_get_by_id_and_index_and_title(self, store):
        rec = _add(store, key="byref", title="Findable")
        assert store.get_suggestion(rec["id"])["id"] == rec["id"]
        assert store.get_suggestion("1")["id"] == rec["id"]
        assert store.get_suggestion("findable")["id"] == rec["id"]
        assert store.get_suggestion("nope") is None

    def test_clear_resolved_drops_accepted_only(self, store):
        _add(store, key="a")
        _add(store, key="b")
        store.dismiss_suggestion("2")  # b dismissed (retained for latch)
        with patch("cron.jobs.create_job", lambda **k: {"id": "j"}):
            store.accept_suggestion("1")  # a accepted
        removed = store.clear_resolved()
        assert removed == 1  # only the accepted record pruned
        # Dismissed record retained so its dedup_key still latches.
        assert _add(store, key="b") is None


class TestCatalog:
    def test_seed_registers_all_entries(self, store):
        from cron.suggestion_catalog import CATALOG, seed_catalog_suggestions

        created = seed_catalog_suggestions(add_fn=store.add_suggestion)
        assert len(created) == len(CATALOG)
        assert len(store.list_pending()) == min(len(CATALOG), store.MAX_PENDING)

    def test_seed_is_idempotent(self, store):
        from cron.suggestion_catalog import seed_catalog_suggestions

        first = seed_catalog_suggestions(add_fn=store.add_suggestion)
        second = seed_catalog_suggestions(add_fn=store.add_suggestion)
        assert len(first) >= 1
        assert second == []  # already present -> nothing new

    def test_monitor_entry_references_classifier_script(self):
        from cron.suggestion_catalog import CATALOG, classify_items_script_path

        monitor = next(e for e in CATALOG if e.key == "catalog:important-mail-monitor")
        assert classify_items_script_path() in monitor.job_spec["prompt"]
        assert Path(classify_items_script_path()).name == "classify_items.py"


class TestRecipeBridge:
    def test_recipe_registers_suggestion(self, store):
        from tools.recipes import RecipeSpec, register_recipe_suggestion

        spec = RecipeSpec(skill_name="morning-brief", schedule="0 8 * * *", deliver="telegram")
        with patch("cron.suggestions.add_suggestion", store.add_suggestion):
            rec = register_recipe_suggestion(spec)
        assert rec is not None
        assert rec["source"] == "recipe"
        assert rec["job_spec"]["skills"] == ["morning-brief"]
        assert rec["job_spec"]["schedule"] == "0 8 * * *"

    def test_recipe_to_job_spec_matches_create_recipe_job(self):
        from tools.recipes import RecipeSpec, recipe_to_job_spec

        spec = RecipeSpec(skill_name="x", schedule="every 2h", deliver="origin", prompt="p")
        js = recipe_to_job_spec(spec)
        assert js["skills"] == ["x"]
        assert js["schedule"] == "every 2h"
        assert js["prompt"] == "p"


class TestCommandHandler:
    def test_bare_lists_pending(self, store):
        _add(store, key="c1", title="Daily thing")
        with patch("cron.suggestions.list_pending", store.list_pending):
            from hermes_cli.suggestions_cmd import handle_suggestions_command
            # Patch the module the handler imports.
            with patch.dict("sys.modules"):
                out = handle_suggestions_command("")
        assert "Daily thing" in out

    def test_accept_via_handler(self, store):
        _add(store, key="ha", title="Acceptable")
        from hermes_cli.suggestions_cmd import handle_suggestions_command

        with patch("cron.jobs.create_job", lambda **k: {"id": "j", "name": k.get("name"), "job_spec": k}):
            out = handle_suggestions_command("accept 1", origin={"platform": "cli", "chat_id": "1"})
        assert "Scheduled" in out
        assert store.list_pending() == []

    def test_dismiss_via_handler(self, store):
        _add(store, key="hd", title="Dismissable")
        from hermes_cli.suggestions_cmd import handle_suggestions_command

        out = handle_suggestions_command("dismiss 1")
        assert "Dismissed" in out
        assert store.list_pending() == []

    def test_empty_list_message(self, store):
        from hermes_cli.suggestions_cmd import handle_suggestions_command

        out = handle_suggestions_command("")
        assert "No suggested automations" in out

    def test_aux_monitor_config_default(self):
        from hermes_cli.config import DEFAULT_CONFIG

        assert "monitor" in DEFAULT_CONFIG["auxiliary"]
        assert DEFAULT_CONFIG["auxiliary"]["monitor"]["provider"] == "auto"
