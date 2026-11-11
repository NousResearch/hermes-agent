"""Config migrations that retire or rewrite obsolete keys (v43 / v44 / v51, stamp-independent steps)."""

import hermes_yaml as yaml


class TestRetiredMultiplexAllowlist:
    def test_v43_drops_multiplex_profile_allowlist_from_user_config(self, tmp_path, monkeypatch):
        """The multiplexer serves every profile; a stale allowlist must not linger in config.yaml."""
        from hermes_cli.config import DEFAULT_CONFIG
        from hermes_cli.config_migrations import run_migrations

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "_config_version": 42,
            "gateway": {"multiplex_profiles": True, "multiplex_profile_allowlist": ["worker"]},
        }), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        run_migrations(42, {"env_added": [], "config_added": [], "warnings": []}, quiet=True)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "multiplex_profile_allowlist" not in raw["gateway"]
        assert raw["gateway"]["multiplex_profiles"] is True
        assert "multiplex_profile_allowlist" not in DEFAULT_CONFIG["gateway"]


class TestCuratorFasterPrune:
    def test_v44_rewrites_old_curator_defaults_but_keeps_user_values(self, tmp_path, monkeypatch):
        """Old 30/90 defaults move to 14/30; an explicitly customized window is untouched."""
        from hermes_cli.config import DEFAULT_CONFIG
        from hermes_cli.config_migrations import run_migrations

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "_config_version": 43,
            "curator": {"stale_after_days": 30, "archive_after_days": 180},
        }), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        run_migrations(43, {"env_added": [], "config_added": [], "warnings": []}, quiet=True)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert raw["curator"]["stale_after_days"] == DEFAULT_CONFIG["curator"]["stale_after_days"]
        assert raw["curator"]["archive_after_days"] == 180

    def test_v44_lands_whole_or_not_at_all(self, tmp_path, monkeypatch):
        """A step that raises is retried whole on the next pass. v44 rewrites two keys: if the
        first rewrite landed and the second raised, the retry would rewrite the first again, over
        a value the user may have set back in between."""
        from hermes_cli import config_migrations
        from hermes_cli.config import DEFAULT_CONFIG, migrate_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "_config_version": 43, "curator": {"stale_after_days": 30, "archive_after_days": 90},
        }), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        real = config_migrations._persist_migration

        def full_disk_on_archive(config):
            if (config.get("curator") or {}).get("archive_after_days") == 30:
                raise OSError("No space left on device")
            real(config)

        monkeypatch.setattr(config_migrations, "_persist_migration", full_disk_on_archive)
        assert migrate_config(interactive=False, quiet=True)["failed_step"] == 44
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert raw["_config_version"] == 43
        assert raw["curator"] == {"stale_after_days": 30, "archive_after_days": 90}, "no half of v44 landed"

        monkeypatch.setattr(config_migrations, "_persist_migration", real)
        raw["curator"]["stale_after_days"] = 21  # the user's own choice before the retry
        config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        assert "failed_step" not in migrate_config(interactive=False, quiet=True)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert raw["curator"] == {"stale_after_days": 21, "archive_after_days": 30}
        assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]


def test_one_step_per_version():
    """The stamp records versions, and a failing step is retried whole: two steps under one
    version would be stamped and retried together, so a later one's failure re-runs the
    earlier one. Each version has exactly one step, in ascending order."""
    from hermes_cli.config_migrations import MIGRATIONS

    versions = [version for version, _step in MIGRATIONS]
    assert versions == sorted(set(versions))


_LEGACY_IMAGE = "nikolaik/python-nodejs:python3.11-nodejs20"
#: One config that triggers every registered step but :data:`_NOT_TRIGGERED_HERE`, each run alone at ``version - 1``.
_TRIGGER_EVERYTHING = {
    "custom_providers": [{"name": "a", "base_url": "http://h/v1"}, {"name": "b", "base_url": "http://i/v1"}],
    "stt": {"model": "base"},
    "display": {"tool_progress_overrides": {"telegram": "off"}, "personality": "pirate",
                "background_process_notifications": "all"},
    "compression": {"summary_model": "m", "threshold_tokens": 256000},
    "plugins": {"enabled": ["observability/nemo_relay"]},
    "memory": {"write_mode": "approve"}, "skills": {"write_mode": "off"}, "model_catalog": {"ttl_hours": 24},
    "agent": {"verify_on_stop": True, "clarify_timeout": 3600},
    "delegation": {"max_async_children": 7, "max_iterations": 50, "max_concurrent_children": 3},
    "platform_toolsets": {"cli": ["terminal", "bfl", "web"]}, "known_builtin_toolsets": {"cli": ["bfl"]},
    "cron": {"model_drift_guard": True, "bot_chat_delivery_timeout_seconds": 30},
    "gateway": {"multiplex_profile_allowlist": ["a"]},
    "curator": {"stale_after_days": 30, "archive_after_days": 90},
    "mcp_servers": {"s": {"command": "x", "disabled": True}, "t": {"command": "y", "disabled": True}},
    "terminal": {"docker_image": _LEGACY_IMAGE, "modal_image": _LEGACY_IMAGE, "daytona_image": _LEGACY_IMAGE,
                 "singularity_image": f"docker://{_LEGACY_IMAGE}", "vercel_runtime": "node24"},
    "security": {"tirith_enabled": True, "tirith_path": "x"},
    "browser": {"backend": "x", "extension_control": True},
}


def test_every_step_writes_config_yaml_at_most_once(tmp_path, monkeypatch):
    """A failing step is retried whole. A step that writes config.yaml twice can land its first
    write, raise on the second, and on the retry write the first part again over a value the
    user set back in between. So every step builds one document and saves it once."""
    import copy
    from hermes_cli import config_migrations

    real = config_migrations._persist_migration
    writes = {}
    for version, step in config_migrations.MIGRATIONS:
        home = tmp_path / f"v{version}"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        doc = {**copy.deepcopy(_TRIGGER_EVERYTHING), "_config_version": version - 1}
        (home / "config.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
        (home / ".env").write_text("LLM_MODEL=x\nOPENAI_MODEL=y\nTERMINAL_VERCEL_RUNTIME=node24\n", encoding="utf-8")
        count = [0]

        def counting(config, _count=count):
            _count[0] += 1
            real(config)

        monkeypatch.setattr(config_migrations, "_persist_migration", counting)
        step({"env_added": [], "config_added": [], "warnings": []}, True)
        monkeypatch.setattr(config_migrations, "_persist_migration", real)
        writes[version] = count[0]
    assert {v: n for v, n in writes.items() if n > 1} == {}
    assert {v for v, n in writes.items() if n == 0} == set(_NOT_TRIGGERED_HERE), \
        "a new step must be triggered by _TRIGGER_EVERYTHING (or listed with its reason)"


#: Steps the shared document does not make write config.yaml, and why.
_NOT_TRIGGERED_HERE = {
    13: "writes .env only",
    41: "writes SOUL.md files only",
    21: "skips when plugins.enabled exists, which v38's trigger needs",
    31: "matches verify_on_stop None/auto; v32's trigger is the literal true",
    40: "matches ttl_hours 1; v25's trigger is 24",
}


def test_v48_lands_whole_so_the_retry_keeps_a_pin_the_user_restored(tmp_path, monkeypatch):
    """v48 drops several old-default sandbox images. If it raised half way, the retry would drop
    again an image the user pinned back on purpose in between."""
    from hermes_cli import config_migrations
    from hermes_cli.config import DEFAULT_CONFIG, migrate_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "_config_version": 47, "terminal": {"docker_image": _LEGACY_IMAGE, "modal_image": _LEGACY_IMAGE},
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    real = config_migrations._persist_migration

    def full_disk_without_modal(config):
        if "modal_image" not in (config.get("terminal") or {}):
            raise OSError("No space left on device")
        real(config)

    monkeypatch.setattr(config_migrations, "_persist_migration", full_disk_without_modal)
    assert migrate_config(interactive=False, quiet=True)["failed_step"] == 48
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["_config_version"] == 47
    assert raw["terminal"] == {"docker_image": _LEGACY_IMAGE, "modal_image": _LEGACY_IMAGE}, "no half of v48 landed"

    monkeypatch.setattr(config_migrations, "_persist_migration", real)
    raw["terminal"]["docker_image"] = "mine:1"  # the user's own pin before the retry
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert "failed_step" not in migrate_config(interactive=False, quiet=True)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["terminal"] == {"docker_image": "mine:1"}
    assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]


class TestRetiredBotChatDeliveryTimeout:
    def test_v51_drops_bot_chat_delivery_timeout_with_a_note(self, tmp_path, monkeypatch):
        """The removed cron knob is dropped from existing configs with a one-time note;
        sibling cron settings and the rest of the file survive untouched."""
        from hermes_cli.config import DEFAULT_CONFIG
        from hermes_cli.config_migrations import run_migrations

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "_config_version": 50,
            "cron": {"bot_chat_delivery_timeout_seconds": 900, "max_parallel_jobs": 2},
        }), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        results = {"env_added": [], "config_added": [], "warnings": []}
        run_migrations(50, results, quiet=True)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "bot_chat_delivery_timeout_seconds" not in raw["cron"]
        assert raw["cron"]["max_parallel_jobs"] == 2
        assert any("bot_chat_delivery_timeout_seconds" in note for note in results["config_added"])
        assert "bot_chat_delivery_timeout_seconds" not in DEFAULT_CONFIG["cron"]


class TestRetiredKeysWhateverTheStamp:
    """Retired-key steps must not depend on a stamp that does not prove they ran: an unversioned
    file (seeded from an older template) and a home an unreleased branch build stamped with its
    own step numbers both still carry keys nothing reads."""

    def test_unversioned_config_drops_the_retired_bot_chat_timeout(self, tmp_path, monkeypatch):
        from hermes_cli.config import migrate_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "cron": {"bot_chat_delivery_timeout_seconds": 900, "max_parallel_jobs": 2},
        }), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        migrate_config(interactive=False, quiet=True)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert raw["cron"] == {"max_parallel_jobs": 2}

    def test_branch_stamped_home_still_gets_main_retired_key_steps(self, tmp_path, monkeypatch):
        """A home stamped 50 by a build whose v50 was the cron-key step skipped main's v46 (MCP
        `disabled: true`) and v50 (tirith keys). Both apply, and a second pass writes nothing."""
        from hermes_cli.config import migrate_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "_config_version": 50,
            "mcp_servers": {"old": {"command": "x", "enabled": True, "disabled": True}},
            "security": {"redact_secrets": True, "tirith_enabled": True},
        }), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        migrate_config(interactive=False, quiet=True)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert raw["mcp_servers"]["old"] == {"command": "x", "enabled": False}
        assert raw["security"] == {"redact_secrets": True}
        settled = config_path.read_bytes()
        assert migrate_config(interactive=False, quiet=True)["config_added"] == []
        assert config_path.read_bytes() == settled
        from hermes_cli.config_migrations import LEGACY_KEY_STEPS, STAMP_INDEPENDENT_STEPS
        assert STAMP_INDEPENDENT_STEPS <= LEGACY_KEY_STEPS


#: One input per stamp-independent step that its trigger key makes it change.
_STAMP_INDEPENDENT_TRIGGERS = {
    46: {"mcp_servers": {"s": {"command": "x", "disabled": True}}},
    50: {"security": {"redact_secrets": True, "tirith_enabled": True}},
}


class TestStampIndependentStepsAreIdempotent:
    """These steps run on EVERY pass, and a step that stopped the ladder is retried on every start,
    so a second run over a step's own output must change nothing."""

    def test_every_stamp_independent_step_has_a_trigger_here(self):
        from hermes_cli.config_migrations import STAMP_INDEPENDENT_STEPS
        assert set(_STAMP_INDEPENDENT_TRIGGERS) == STAMP_INDEPENDENT_STEPS

    def test_second_run_changes_nothing(self, tmp_path, monkeypatch):
        from hermes_cli.config_migrations import MIGRATIONS, STAMP_INDEPENDENT_STEPS

        config_path = tmp_path / "config.yaml"
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        for version, step in ((v, fn) for v, fn in MIGRATIONS if v in STAMP_INDEPENDENT_STEPS):
            config_path.write_text(yaml.safe_dump(
                {"_config_version": version - 1, **_STAMP_INDEPENDENT_TRIGGERS[version]}), encoding="utf-8")
            before = config_path.read_bytes()
            first = {"env_added": [], "config_added": [], "warnings": []}
            step(first, True)
            settled = config_path.read_bytes()
            assert settled != before and first["config_added"], f"v{version}'s trigger did not trigger it"
            second = {"env_added": [], "config_added": [], "warnings": []}
            step(second, True)
            assert config_path.read_bytes() == settled, f"v{version} changed its own output"
            assert second["config_added"] == [] and second["warnings"] == []


class TestNewerStamp:
    """A config stamped by a newer build (rollback, mixed desktop/CLI builds) is read as it is: no
    step runs on it, so this build does not rewrite what the newer one wrote, and saves keep it."""

    def _newer(self, tmp_path, monkeypatch):
        from hermes_cli.config import DEFAULT_CONFIG

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "_config_version": DEFAULT_CONFIG["_config_version"] + 7,
            "mcp_servers": {"s": {"command": "x", "disabled": True}},
            "security": {"tirith_enabled": True},
            "zz_newer_section": {"k": 1},
        }), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        return config_path

    def test_migrate_runs_no_step_on_it(self, tmp_path, monkeypatch):
        from hermes_cli.config import migrate_config

        config_path = self._newer(tmp_path, monkeypatch)
        before = config_path.read_bytes()
        results = migrate_config(interactive=False, quiet=True)
        assert config_path.read_bytes() == before
        assert "failed_step" not in results, "no step may even start on a newer file"
        assert any("newer Hermes" in w for w in results["warnings"])

    def test_a_legacy_disabled_server_stays_off_without_v46(self, tmp_path, monkeypatch):
        """v46 does not run on a newer file, so its `disabled: true` stays on disk; the runtime
        must still read it as off, or this build spawns a server the user turned off."""
        from hermes_cli.config import load_config, migrate_config
        import tools.mcp_tool as core
        from tools import mcp_tool_discovery as discovery
        from tools.mcp_tool_common import mcp_server_enabled

        config_path = self._newer(tmp_path, monkeypatch)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        raw["mcp_servers"]["t"] = {"command": "x"}
        config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        migrate_config(interactive=False, quiet=True)
        assert load_config()["mcp_servers"]["s"]["disabled"] is True
        assert mcp_server_enabled(load_config()["mcp_servers"]["s"]) is False

        spawned = []
        monkeypatch.setattr(core, "_ensure_mcp_sdk", lambda: True)
        monkeypatch.setattr(discovery._loop, "_ensure_mcp_loop", lambda: None)
        monkeypatch.setattr(discovery, "_run_discovery_pass", lambda servers: spawned.extend(sorted(servers)))
        monkeypatch.setattr(discovery, "_acquire_discovery_lock_with_retry", lambda: None)
        monkeypatch.setattr(discovery, "_register_lazy_from_cache", lambda s: (s, 0, 0))
        monkeypatch.setattr(core, "_servers", {})
        discovery.discover_mcp_tools()
        assert spawned == ["t"]

    def test_whole_file_save_keeps_what_the_newer_build_wrote(self, tmp_path, monkeypatch):
        """Saves still work on a newer file and keep its stamp, unknown keys and legacy keys:
        only the migration steps rewrote a newer build's settings."""
        from hermes_cli.config import load_config, save_config

        config_path = self._newer(tmp_path, monkeypatch)
        stamp = yaml.safe_load(config_path.read_text(encoding="utf-8"))["_config_version"]
        cfg = load_config()
        cfg.setdefault("display", {})["personality"] = "pirate"
        save_config(cfg)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert raw["display"]["personality"] == "pirate"
        assert raw["_config_version"] == stamp
        assert raw["zz_newer_section"] == {"k": 1}
        assert raw["mcp_servers"]["s"]["disabled"] is True
        assert raw["security"]["tirith_enabled"] is True


def test_raw_config_editor_maps_an_expected_refusal(tmp_path, monkeypatch):
    """PUT /api/config/raw says why an expected refusal saved nothing (409 here), not a 500."""
    from starlette.testclient import TestClient
    from hermes_cli import config as config_mod
    from hermes_cli.config_backend import ConfigWriteError
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def lost_race(*_a, **_kw):
        raise ConfigWriteError("config.yaml changed underneath this save", code="config_version_conflict")

    monkeypatch.setattr(config_mod, "save_config", lost_race)
    r = TestClient(app, raise_server_exceptions=False).put(
        "/api/config/raw", json={"yaml_text": "display:\n  skin: ares\n"},
        headers={_SESSION_HEADER_NAME: _SESSION_TOKEN})
    assert r.status_code == 409, (r.status_code, r.text)
    assert "changed underneath" in r.json()["detail"]
