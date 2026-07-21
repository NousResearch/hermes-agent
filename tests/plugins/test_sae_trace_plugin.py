"""Tests for the bundled observability/sae_trace plugin.

Covers:

  * Manifest + layout (plugin.yaml fields, hooks list, requires_env).
  * Opt-in discovery via ``PluginManager.discover_and_load``.
  * Runtime gate: without ``HERMES_SAE_TRACE_FILE`` the config is a cached
    miss and every hook is inert (fail-open, langfuse-style).
  * Sidecar tailing: offset persistence, partial trailing lines,
    malformed-line skipping, truncation/rotation recovery.
  * Correlation tiers: request_id > session_id > time_window, model
    gating, once-only claiming, and the per-session output line format
    (top-features summary and npz_path passthrough).
  * ``/sae`` slash command: status / last / help / unconfigured.
  * Standalone (user-dir) load path: module works when
    ``plugins.plugin_utils`` is not importable.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "sae_trace"

MOD_NAME = "plugins.observability.sae_trace"


def _fresh_plugin():
    """Import the plugin module fresh (clears cached config/tailer/stats)."""
    sys.modules.pop(MOD_NAME, None)
    return importlib.import_module(MOD_NAME)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Configured environment: sidecar file + out dir, fresh module."""
    sidecar = tmp_path / "sae_history.jsonl"
    sidecar.write_text("", encoding="utf-8")
    out_dir = tmp_path / "out"
    monkeypatch.setenv("HERMES_SAE_TRACE_FILE", str(sidecar))
    monkeypatch.setenv("HERMES_SAE_TRACE_OUT_DIR", str(out_dir))
    monkeypatch.delenv("HERMES_SAE_TRACE_SKEW", raising=False)
    plugin = _fresh_plugin()
    return plugin, sidecar, out_dir


def _append(sidecar: Path, record: dict) -> None:
    with open(sidecar, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _utc_iso(epoch: float) -> str:
    return (
        datetime.fromtimestamp(epoch, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


# ---------------------------------------------------------------------------
# Manifest + layout
# ---------------------------------------------------------------------------

class TestManifest:
    def test_plugin_directory_exists(self):
        assert PLUGIN_DIR.is_dir()
        assert (PLUGIN_DIR / "plugin.yaml").exists()
        assert (PLUGIN_DIR / "__init__.py").exists()
        assert (PLUGIN_DIR / "README.md").exists()

    def test_manifest_fields(self):
        data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8"))
        assert data["name"] == "sae_trace"
        assert data["version"]
        assert set(data["hooks"]) == {"pre_api_request", "post_api_request"}
        assert "HERMES_SAE_TRACE_FILE" in data["requires_env"]
        assert "SolshineCode" in data["author"]


# ---------------------------------------------------------------------------
# Plugin discovery: sae_trace is opt-in (not loaded unless enabled).
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_plugin_is_discovered_as_standalone_opt_in(self, tmp_path, monkeypatch):
        """Scanner should find the plugin but NOT load it by default."""
        from hermes_cli import plugins as plugins_mod

        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        manager = plugins_mod.PluginManager()
        manager.discover_and_load()

        loaded = manager._plugins.get("observability/sae_trace")
        assert loaded is not None, "plugin not discovered"
        assert loaded.enabled is False
        assert "not enabled" in (loaded.error or "").lower()


# ---------------------------------------------------------------------------
# Runtime gate: no HERMES_SAE_TRACE_FILE -> cached miss, inert hooks.
# ---------------------------------------------------------------------------

class TestRuntimeGate:
    def test_config_none_without_trace_file(self, monkeypatch):
        monkeypatch.delenv("HERMES_SAE_TRACE_FILE", raising=False)
        plugin = _fresh_plugin()
        assert plugin._get_config() is None

    def test_config_miss_is_cached(self, monkeypatch):
        """A miss must be cached — no env re-reads on later hook calls."""
        monkeypatch.delenv("HERMES_SAE_TRACE_FILE", raising=False)
        plugin = _fresh_plugin()
        assert plugin._get_config() is None

        import os

        called = {"n": 0}
        real_get = os.environ.get

        def tracking_get(key, default=None):
            if key == "HERMES_SAE_TRACE_FILE":
                called["n"] += 1
            return real_get(key, default)

        monkeypatch.setattr(os.environ, "get", tracking_get)
        for _ in range(20):
            assert plugin._get_config() is None
        assert called["n"] == 0

    def test_hooks_inert_without_config(self, monkeypatch):
        monkeypatch.delenv("HERMES_SAE_TRACE_FILE", raising=False)
        plugin = _fresh_plugin()
        # Neither hook may raise, and nothing is written anywhere.
        plugin.on_pre_api_request(model="m")
        plugin.on_post_api_request(
            session_id="s", model="m", started_at=time.time(), ended_at=time.time()
        )
        with plugin._STATS.lock:
            assert plugin._STATS.turns_matched == 0
            assert plugin._STATS.turns_unmatched == 0

    def test_hooks_inert_when_sidecar_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "HERMES_SAE_TRACE_FILE", str(tmp_path / "does-not-exist.jsonl")
        )
        monkeypatch.setenv("HERMES_SAE_TRACE_OUT_DIR", str(tmp_path / "out"))
        plugin = _fresh_plugin()
        plugin.on_pre_api_request()
        plugin.on_post_api_request(
            session_id="s", model="m", started_at=time.time(), ended_at=time.time()
        )
        assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# Sidecar tailer
# ---------------------------------------------------------------------------

class TestTailer:
    def test_prime_skips_history(self, env):
        plugin, sidecar, _ = env
        _append(sidecar, {"request_id": "old", "ts": _utc_iso(time.time())})
        tailer = plugin._get_tailer()
        tailer.prime()
        _append(sidecar, {"request_id": "new", "ts": _utc_iso(time.time())})
        tailer.poll()
        assert tailer.records_seen == 1

    def test_partial_trailing_line_left_unconsumed(self, env):
        plugin, sidecar, _ = env
        tailer = plugin._get_tailer()
        tailer.prime()
        with open(sidecar, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"request_id": "a"}) + "\n")
            fh.write('{"request_id": "half')  # no newline: mid-write
        tailer.poll()
        assert tailer.records_seen == 1
        assert tailer.malformed_lines == 0
        # Server finishes the line -> next poll picks it up whole.
        with open(sidecar, "a", encoding="utf-8") as fh:
            fh.write('"}\n')
        tailer.poll()
        assert tailer.records_seen == 2

    def test_malformed_lines_skipped(self, env):
        plugin, sidecar, _ = env
        tailer = plugin._get_tailer()
        tailer.prime()
        with open(sidecar, "a", encoding="utf-8") as fh:
            fh.write("not json at all\n")
            fh.write('[1, 2, 3]\n')  # valid JSON, wrong shape
            fh.write(json.dumps({"request_id": "ok"}) + "\n")
        tailer.poll()
        assert tailer.records_seen == 1
        assert tailer.malformed_lines == 2

    def test_truncation_resets_offset(self, env):
        plugin, sidecar, _ = env
        tailer = plugin._get_tailer()
        tailer.prime()
        _append(sidecar, {"request_id": "a" * 40})
        tailer.poll()
        assert tailer.records_seen == 1
        # Rotation: file replaced by a shorter fresh one.
        sidecar.write_text(json.dumps({"request_id": "b"}) + "\n", encoding="utf-8")
        tailer.poll()
        assert tailer.records_seen == 2


# ---------------------------------------------------------------------------
# Correlation + output
# ---------------------------------------------------------------------------

class TestCorrelation:
    def _run_turn(self, plugin, sidecar, records, **hook_kwargs):
        plugin.on_pre_api_request()
        started = time.time()
        for record in records:
            _append(sidecar, record)
        ended = time.time()
        kwargs = dict(
            task_id="task-1",
            turn_id="turn-1",
            api_request_id="req-1",
            session_id="sess-1",
            model="sae-local",
            api_duration=ended - started,
            started_at=started,
            ended_at=ended,
        )
        kwargs.update(hook_kwargs)
        plugin.on_post_api_request(**kwargs)
        return kwargs

    def _read_output(self, out_dir, session_id="sess-1"):
        path = out_dir / f"{session_id}.jsonl"
        assert path.exists(), f"no output file at {path}"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_request_id_match(self, env):
        plugin, sidecar, out_dir = env
        self._run_turn(
            plugin, sidecar,
            [{"request_id": "req-1", "model": "other-model"}],
        )
        (line,) = self._read_output(out_dir)
        assert line["match_confidence"] == "request_id"
        assert line["records"][0]["request_id"] == "req-1"

    def test_session_id_match_nla_server_shape(self, env):
        """activation-capture record: naive local `timestamp`, npz pointer."""
        plugin, sidecar, out_dir = env
        record = {
            "request_id": "abcd1234",
            "session_id": "sess-1",
            "timestamp": datetime.now().isoformat(),  # naive local ISO
            "model": "gemma-4-e2b",
            "layer": 23,
            "d_model": 1536,
            "n_records": 12,
            "n_gen_tokens": 11,
            "npz_path": "/logs/run_abcd1234.npz",
            "gen_text_preview": "hello world",
        }
        self._run_turn(plugin, sidecar, [record], model="my-alias")
        (line,) = self._read_output(out_dir)
        assert line["match_confidence"] == "session_id"
        rec = line["records"][0]
        assert rec["npz_path"] == "/logs/run_abcd1234.npz"
        assert rec["layer"] == 23
        assert rec["gen_text_preview"] == "hello world"
        assert "top_features" not in rec

    def test_time_window_match_sae_serve_shape(self, env):
        """feature-history record: UTC `ts` with Z suffix, feats_topk."""
        plugin, sidecar, out_dir = env
        record = {
            "ts": _utc_iso(time.time()),
            "request_id": "chatcmpl-deadbeef",
            "model": "sae-local",
            "prompt_len": 100,
            "gen_len": 3,
            "same_inference": True,
            "feats_topk": {
                "16": [[0, 7, 1.0], [1, 7, 5.0], [2, 9, 3.0], [0, 3, 0.5]],
            },
            "gen_text": "x" * 500,
        }
        self._run_turn(plugin, sidecar, [record])
        (line,) = self._read_output(out_dir)
        assert line["match_confidence"] == "time_window"
        rec = line["records"][0]
        # Top features sorted by max activation: feat 7 (max 5.0, 2 tokens),
        # then feat 9, then feat 3.
        assert rec["top_features"]["16"] == [[7, 5.0, 2], [9, 3.0, 1], [3, 0.5, 1]]
        assert rec["same_inference"] is True
        assert len(rec["gen_text_preview"]) == 200  # bounded

    def test_output_line_identity_fields(self, env):
        plugin, sidecar, out_dir = env
        kwargs = self._run_turn(
            plugin, sidecar, [{"request_id": "req-1"}],
        )
        (line,) = self._read_output(out_dir)
        for key in ("ts", "session_id", "task_id", "turn_id",
                    "api_request_id", "model", "match_confidence", "records"):
            assert key in line
        assert line["session_id"] == kwargs["session_id"]
        assert line["task_id"] == kwargs["task_id"]
        assert line["model"] == kwargs["model"]

    def test_model_mismatch_blocks_time_window(self, env):
        plugin, sidecar, out_dir = env
        self._run_turn(
            plugin, sidecar,
            [{"ts": _utc_iso(time.time()), "model": "some-other-model"}],
        )
        assert not (out_dir / "sess-1.jsonl").exists()
        with plugin._STATS.lock:
            assert plugin._STATS.turns_unmatched == 1

    def test_stale_record_outside_window_no_match(self, env):
        plugin, sidecar, out_dir = env
        self._run_turn(
            plugin, sidecar,
            [{"ts": _utc_iso(time.time() - 3600), "model": "sae-local"}],
        )
        assert not (out_dir / "sess-1.jsonl").exists()

    def test_records_claimed_once(self, env):
        plugin, sidecar, out_dir = env
        self._run_turn(plugin, sidecar, [{"request_id": "req-1"}])
        # Second turn appends nothing new — must not re-match req-1's record.
        self._run_turn(plugin, sidecar, [], api_request_id="req-2", turn_id="turn-2")
        lines = self._read_output(out_dir)
        assert len(lines) == 1
        with plugin._STATS.lock:
            assert plugin._STATS.turns_matched == 1
            assert plugin._STATS.turns_unmatched == 1

    def test_session_filename_sanitized(self, env):
        plugin, sidecar, out_dir = env
        self._run_turn(
            plugin, sidecar,
            [{"request_id": "req-1"}],
            session_id="../../../etc/pwned",
        )
        # The traversal-shaped ID stays inside out_dir as a single segment.
        files = list(out_dir.iterdir())
        assert files, "no output written"
        assert all(p.parent == out_dir for p in files)
        assert all("/" not in p.name and ".." not in p.name for p in files)
        assert not (out_dir.parent.parent / "etc").exists()

    def test_concurrent_hooks_do_not_raise(self, env):
        plugin, sidecar, out_dir = env
        plugin.on_pre_api_request()
        started = time.time()
        for i in range(20):
            _append(sidecar, {"request_id": f"req-{i}"})
        ended = time.time()
        errors = []

        def fire(i):
            try:
                plugin.on_post_api_request(
                    api_request_id=f"req-{i}",
                    session_id="sess-1",
                    model="sae-local",
                    started_at=started,
                    ended_at=ended,
                )
            except Exception as exc:  # pragma: no cover - the assertion
                errors.append(exc)

        threads = [threading.Thread(target=fire, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        lines = self._read_output(out_dir)
        assert len(lines) == 20
        matched_ids = sorted(
            line["records"][0]["request_id"] for line in lines
        )
        assert matched_ids == sorted(f"req-{i}" for i in range(20))


# ---------------------------------------------------------------------------
# /sae slash command
# ---------------------------------------------------------------------------

class TestSlashCommand:
    def test_unconfigured_message(self, monkeypatch):
        monkeypatch.delenv("HERMES_SAE_TRACE_FILE", raising=False)
        plugin = _fresh_plugin()
        out = plugin._handle_slash("status")
        assert "HERMES_SAE_TRACE_FILE" in out

    def test_help(self, env):
        plugin, _, _ = env
        out = plugin._handle_slash("help")
        assert "status" in out and "last" in out

    def test_status_reports_sidecar_and_counts(self, env):
        plugin, sidecar, _ = env
        plugin.on_pre_api_request()
        started = time.time()
        _append(sidecar, {"request_id": "req-1", "feats_topk": {"3": [[0, 42, 9.0]]}})
        plugin.on_post_api_request(
            api_request_id="req-1", session_id="sess-1", model="sae-local",
            started_at=started, ended_at=time.time(),
        )
        out = plugin._handle_slash("status")
        assert str(sidecar) in out
        assert "turns matched: 1" in out
        assert "request_id" in out  # last-match confidence line

    def test_last_prints_feature_summary(self, env):
        plugin, sidecar, _ = env
        plugin.on_pre_api_request()
        started = time.time()
        _append(sidecar, {"request_id": "req-1", "feats_topk": {"3": [[0, 42, 9.0]]}})
        plugin.on_post_api_request(
            api_request_id="req-1", session_id="sess-1", model="sae-local",
            started_at=started, ended_at=time.time(),
        )
        out = plugin._handle_slash("last")
        assert "layer 3" in out
        assert "#42" in out

    def test_last_without_match(self, env):
        plugin, _, _ = env
        out = plugin._handle_slash("last")
        assert "No correlated turn" in out

    def test_unknown_subcommand(self, env):
        plugin, _, _ = env
        out = plugin._handle_slash("bogus")
        assert "Unknown subcommand" in out

    def test_register_wires_hooks_and_command(self, env):
        plugin, _, _ = env

        class Ctx:
            def __init__(self):
                self.hooks = {}
                self.commands = {}

            def register_hook(self, name, cb):
                self.hooks[name] = cb

            def register_command(self, name, handler, description="", args_hint=""):
                self.commands[name] = handler

        ctx = Ctx()
        plugin.register(ctx)
        assert set(ctx.hooks) == {"pre_api_request", "post_api_request"}
        assert "sae" in ctx.commands


# ---------------------------------------------------------------------------
# Standalone (user-dir) install: module must load and work when
# ``plugins.plugin_utils`` is not importable (the loader imports user
# plugins as ``hermes_plugins.<slug>`` from their own directory).
# ---------------------------------------------------------------------------

class TestStandaloneLoad:
    def _load_standalone(self):
        class _BlockPluginsPkg:
            """Meta-path hook that makes ``plugins.*`` unimportable."""

            def find_spec(self, name, path=None, target=None):
                if name == "plugins" or name.startswith("plugins."):
                    raise ImportError(f"blocked for standalone test: {name}")
                return None

        blocker = _BlockPluginsPkg()
        saved = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if name == "plugins" or name.startswith("plugins.")
        }
        sys.meta_path.insert(0, blocker)
        try:
            spec = importlib.util.spec_from_file_location(
                "hermes_plugins.sae_trace",
                PLUGIN_DIR / "__init__.py",
                submodule_search_locations=[str(PLUGIN_DIR)],
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules["hermes_plugins.sae_trace"] = module
            spec.loader.exec_module(module)
            return module
        finally:
            sys.meta_path.remove(blocker)
            sys.modules.pop("hermes_plugins.sae_trace", None)
            sys.modules.update(saved)

    def test_loads_and_correlates_without_plugin_utils(
        self, tmp_path, monkeypatch
    ):
        sidecar = tmp_path / "sidecar.jsonl"
        sidecar.write_text("", encoding="utf-8")
        out_dir = tmp_path / "out"
        monkeypatch.setenv("HERMES_SAE_TRACE_FILE", str(sidecar))
        monkeypatch.setenv("HERMES_SAE_TRACE_OUT_DIR", str(out_dir))

        module = self._load_standalone()
        module.on_pre_api_request()
        started = time.time()
        _append(sidecar, {"request_id": "req-sa", "feats_topk": {"0": [[0, 1, 2.0]]}})
        module.on_post_api_request(
            api_request_id="req-sa", session_id="sess-sa", model="m",
            started_at=started, ended_at=time.time(),
        )
        lines = (out_dir / "sess-sa.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["match_confidence"] == "request_id"


# ---------------------------------------------------------------------------
# Extended coverage: summaries, tier precedence, timestamp parsing, races
# ---------------------------------------------------------------------------

class TestExtendedCoverage:
    def test_feature_summary_and_preview_bounds(self, env):
        """Top-features are ranked by max activation and payloads stay bounded."""
        plugin, sidecar, out_dir = env
        plugin.on_pre_api_request()
        started = time.time()
        events = [[t, 7, 0.5] for t in range(50)] + [[0, 9, 3.25], [1, 9, 1.0]]
        _append(
            sidecar,
            {
                "request_id": "req-1",
                "feats_topk": {"12": events},
                "gen_text": "x" * 1000,
                "allf": [[0, 1, 0.1]],
            },
        )
        plugin.on_post_api_request(
            api_request_id="req-1", session_id="sess-1", model="sae-local",
            started_at=started, ended_at=time.time(),
        )
        (line,) = [
            json.loads(raw)
            for raw in (out_dir / "sess-1.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        record = line["records"][0]
        assert record["top_features"]["12"][0] == [9, 3.25, 2]
        assert [7, 0.5, 50] in record["top_features"]["12"]
        assert len(record["gen_text_preview"]) == 200
        assert record["has_all_features"] is True
        assert "allf" not in json.dumps(line)

    def test_request_id_tier_reported_over_weaker_matches(self, env):
        """One turn matching several records reports the strongest tier."""
        plugin, sidecar, out_dir = env
        plugin.on_pre_api_request()
        started = time.time()
        _append(sidecar, {"session_id": "sess-1", "timestamp": time.time()})
        _append(sidecar, {"request_id": "req-1"})
        plugin.on_post_api_request(
            api_request_id="req-1", session_id="sess-1", model="sae-local",
            started_at=started, ended_at=time.time(),
        )
        (line,) = [
            json.loads(raw)
            for raw in (out_dir / "sess-1.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert line["match_confidence"] == "request_id"
        assert len(line["records"]) == 2

    def test_timestamp_parsing_forms(self, env):
        plugin, _, _ = env
        assert plugin._parse_record_ts({"ts": "2026-07-21T12:00:00Z"}) == 1784635200.0
        assert plugin._parse_record_ts({"timestamp": 123.5}) == 123.5
        assert plugin._parse_record_ts({"ts": "not-a-date"}) is None
        assert plugin._parse_record_ts({}) is None

    def test_concurrent_turns_claim_each_record_exactly_once(self, env):
        plugin, sidecar, out_dir = env
        plugin.on_pre_api_request()
        started = time.time()
        for i in range(20):
            _append(sidecar, {"request_id": f"req-{i}"})

        def _turn(i: int) -> None:
            plugin.on_post_api_request(
                api_request_id=f"req-{i}", session_id="sess-1", model="sae-local",
                started_at=started, ended_at=time.time(),
            )

        threads = [threading.Thread(target=_turn, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        lines = [
            json.loads(raw)
            for raw in (out_dir / "sess-1.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert len(lines) == 20
        claimed = {line["records"][0]["request_id"] for line in lines}
        assert len(claimed) == 20
