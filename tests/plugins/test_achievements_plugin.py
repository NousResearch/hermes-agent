"""Tests for the bundled hermes-achievements dashboard plugin.

These target the two behaviors that matter for official integration:

* The 200-session scan cap is removed — the plugin now walks the entire
  session history by default. Lifetime badges (tens of thousands of
  tool calls) were unreachable before this fix on long-running installs.
* First-ever scans run in a background thread so the dashboard request
  path never blocks, even on 8000+ session databases where a cold scan
  takes minutes.

The upstream repo ships its own unittest suite under
``plugins/hermes-achievements/tests/`` covering the achievement engine
internals (tier math, secret-state handling, catalog invariants). These
tests live at the hermes-agent level and focus on the integration
contract: the plugin scans ALL of your sessions, not the first 200.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PLUGIN_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "hermes-achievements"
    / "dashboard"
    / "plugin_api.py"
)


@pytest.fixture
def plugin_api(tmp_path, monkeypatch):
    """Load plugin_api with isolated ~/.hermes so state/snapshot files don't collide.

    We load the module fresh per test because the plugin keeps module-level
    caches (``_SNAPSHOT_CACHE``, ``_SCAN_STATUS``, background thread handle).
    Reloading gives each test a clean world.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec = importlib.util.spec_from_file_location(
        f"plugin_api_test_{id(tmp_path)}", PLUGIN_MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Stash monkeypatch so ``_install_fake_session_db`` can use it to
    # swap ``sys.modules['hermes_state']`` with auto-restoration. Without
    # this, a raw ``sys.modules[...] = fake`` assignment would leak the
    # fake into later tests in the same process — breaking every
    # test that does ``from hermes_state import SessionDB``.
    module._test_monkeypatch = monkeypatch
    yield module


class _FakeSessionDB:
    """Stand-in for hermes_state.SessionDB that records scan calls."""

    def __init__(self, session_count: int, scan_delay: float = 0):
        self.session_count = session_count
        self.scan_delay = scan_delay
        self.last_limit: Optional[int] = None
        self.last_include_children: Optional[bool] = None
        self.list_calls = 0
        self.messages_calls = 0

    def list_sessions_rich(
        self,
        source: Optional[str] = None,
        exclude_sources: Optional[list[str]] = None,
        limit: int = 20,
        offset: int = 0,
        include_children: bool = False,
        project_compression_tips: bool = True,
    ) -> list[dict[str, Any]]:
        if self.scan_delay:
            time.sleep(self.scan_delay)
        self.last_limit = limit
        self.last_include_children = include_children
        self.list_calls += 1
        # SQLite semantics: LIMIT -1 = unlimited. Honor that here.
        effective = self.session_count if limit == -1 else min(self.session_count, limit)
        now = int(time.time())
        return [
            {
                "id": f"sess-{i}",
                "title": f"Session {i}",
                "preview": f"preview {i}",
                "started_at": now - (self.session_count - i) * 60,
                "last_active": now - (self.session_count - i) * 60 + 30,
                "source": "cli",
                "model": "test-model",
            }
            for i in range(effective)
        ]

    def get_messages(self, session_id: str, include_compacted: bool = False) -> list[dict[str, Any]]:
        self.messages_calls += 1
        return [
            {"role": "user", "content": f"ask {session_id}"},
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "terminal"}}],
            },
            {"role": "tool", "tool_name": "terminal", "content": "ok"},
        ]

    def close(self) -> None:
        pass


def _install_fake_session_db(plugin_api, fake_db):
    """Inject a fake SessionDB so ``scan_sessions`` finds it via its local import.

    Uses the monkeypatch stashed on ``plugin_api`` by the fixture, so the
    ``sys.modules['hermes_state']`` swap is auto-restored at test teardown
    and cannot leak into unrelated tests in the same process.
    """
    fake_module = type(sys)("hermes_state")
    fake_module.SessionDB = lambda **_kw: fake_db
    plugin_api._test_monkeypatch.setitem(sys.modules, "hermes_state", fake_module)


def test_scan_sessions_default_scans_all_history_not_first_200(plugin_api):
    """Bug regression: ``scan_sessions()`` used to cap at limit=200.

    A user with 8000+ sessions would only see ~2% of their history in
    achievement totals, making lifetime badges unreachable. The default
    now passes ``LIMIT -1`` (SQLite "unlimited") to ``list_sessions_rich``.
    """
    fake_db = _FakeSessionDB(session_count=500)  # > old 200 cap
    _install_fake_session_db(plugin_api, fake_db)

    result = plugin_api.scan_sessions()

    assert fake_db.last_limit == -1, (
        "scan_sessions() must pass LIMIT=-1 (unlimited) to list_sessions_rich "
        f"by default, got {fake_db.last_limit}"
    )
    assert fake_db.last_include_children is True, (
        "scan_sessions() must include subagent/compression child sessions so "
        "tool calls made in delegated agents still count toward achievements"
    )
    assert len(result["sessions"]) == 500
    assert result["scan_meta"]["sessions_total"] == 500


def test_evaluate_all_first_run_returns_pending_and_starts_background_scan(plugin_api):
    """First-ever evaluate_all with no cache returns a pending placeholder
    immediately and kicks off a background scan thread. Cold scans on
    large DBs take minutes — blocking the dashboard request path is not
    acceptable.
    """
    fake_db = _FakeSessionDB(session_count=50)
    _install_fake_session_db(plugin_api, fake_db)

    # Wrap _run_scan_and_update_cache so we can release it on demand,
    # simulating a slow cold scan without actually waiting.
    scan_started = threading.Event()
    allow_scan_finish = threading.Event()
    original_run = plugin_api._run_scan_and_update_cache

    def gated_run(*args, **kwargs):
        scan_started.set()
        allow_scan_finish.wait(timeout=5)
        original_run(*args, **kwargs)

    plugin_api._run_scan_and_update_cache = gated_run

    t0 = time.time()
    result = plugin_api.evaluate_all()
    elapsed = time.time() - t0

    # Immediate return — should not block waiting for the scan.
    assert elapsed < 1.0, f"evaluate_all blocked for {elapsed:.2f}s on first run"
    assert result["scan_meta"]["mode"] == "pending"
    assert result["unlocked_count"] == 0
    # Catalog still rendered so UI has something to draw.
    assert result["total_count"] >= 60

    # Background scan is running.
    assert scan_started.wait(timeout=2), "background scan did not start"

    # Let the scan complete, then a second call returns real data.
    allow_scan_finish.set()
    # Wait for thread to finish.
    thread = plugin_api._BACKGROUND_SCAN_THREAD
    assert thread is not None
    thread.join(timeout=5)
    assert not thread.is_alive()

    second = plugin_api.evaluate_all()
    assert second["scan_meta"]["mode"] != "pending"
    assert second["scan_meta"].get("sessions_total") == 50


def test_start_background_scan_is_idempotent_while_running(plugin_api):
    """Multiple concurrent dashboard requests must not spawn duplicate scans."""
    fake_db = _FakeSessionDB(session_count=5)
    _install_fake_session_db(plugin_api, fake_db)

    release = threading.Event()
    original_run = plugin_api._run_scan_and_update_cache

    def gated_run(*args, **kwargs):
        release.wait(timeout=5)
        original_run(*args, **kwargs)

    plugin_api._run_scan_and_update_cache = gated_run

    plugin_api._start_background_scan()
    first_thread = plugin_api._BACKGROUND_SCAN_THREAD
    assert first_thread is not None and first_thread.is_alive()

    plugin_api._start_background_scan()
    plugin_api._start_background_scan()

    assert plugin_api._BACKGROUND_SCAN_THREAD is first_thread

    release.set()
    first_thread.join(timeout=5)


def test_background_scan_publishes_partial_snapshots(plugin_api):
    """The background scanner publishes intermediate snapshots to the cache
    every ~N sessions. Each dashboard refresh during a long cold scan sees
    more badges unlocked instead of staring at zeros for minutes and then
    having everything pop at the end.
    """
    fake_db = _FakeSessionDB(session_count=750)
    _install_fake_session_db(plugin_api, fake_db)

    # Record every partial snapshot the scanner publishes.
    partial_snapshots: list[dict[str, Any]] = []
    original_compute_from_scan = plugin_api._compute_from_scan

    def recording_compute(scan, *, is_partial=False):
        result = original_compute_from_scan(scan, is_partial=is_partial)
        if is_partial:
            partial_snapshots.append(result)
        return result

    plugin_api._compute_from_scan = recording_compute

    # scan 750 sessions with progress_every=250 → expect 2 intermediate
    # publications (at 250 and 500; the final 750 call goes through the
    # finished, non-partial path).
    plugin_api._run_scan_and_update_cache(publish_partial_snapshots=True)

    assert len(partial_snapshots) >= 2, (
        f"expected at least 2 partial publications on a 750-session scan with "
        f"progress_every=250, got {len(partial_snapshots)}"
    )
    # Partial snapshots should report growing session counts.
    counts = [p["scan_meta"].get("sessions_scanned_so_far") for p in partial_snapshots]
    assert counts == sorted(counts), f"partial session counts not monotonic: {counts}"
    assert counts[0] < 750 and counts[-1] < 750, (
        f"partial counts should be less than the final total; got {counts}"
    )
    # Every partial reports the expected end-state total so the UI can
    # show an accurate progress bar.
    for p in partial_snapshots:
        assert p["scan_meta"].get("sessions_expected_total") == 750

    # Final snapshot in cache is the real (non-partial) one.
    final = plugin_api._SNAPSHOT_CACHE
    assert final is not None
    assert final["scan_meta"].get("mode") != "in_progress"
    assert final["scan_meta"].get("sessions_total") == 750


def test_partial_snapshots_do_not_persist_unlock_timestamps(plugin_api):
    """Intermediate snapshots must not write to state.json — an unlock
    that appears at 30% scan progress could disappear when a later session
    rebalances the aggregate. Only the final snapshot records ``unlocked_at``.
    """
    fake_db = _FakeSessionDB(session_count=10)
    _install_fake_session_db(plugin_api, fake_db)

    # Seed empty state, then invoke partial compute directly.
    plugin_api.save_state({"unlocks": {}})
    partial_scan = {
        "sessions": [{"session_id": "x", "tool_call_count": 99999, "tool_names": set()}],
        "aggregate": {"max_tool_calls_in_session": 99999, "total_tool_calls": 99999},
        "scan_meta": {"mode": "in_progress"},
    }
    result = plugin_api._compute_from_scan(partial_scan, is_partial=True)

    # Some achievements should evaluate as unlocked in this aggregate...
    assert any(a["unlocked"] for a in result["achievements"])

    # ...but state.json on disk stays empty (no timestamps were recorded).
    persisted = plugin_api.load_state()
    assert persisted.get("unlocks", {}) == {}, (
        "partial scans must not record unlock timestamps — a later session "
        "could change whether the badge deserves to be unlocked yet"
    )


def test_scan_sessions_never_opens_a_writable_session_db(plugin_api, tmp_path, monkeypatch):
    """The scan is a pure read that runs inside the dashboard process (per background scan,
    per /rescan). A writable ``SessionDB()`` there was one more writer connection on the
    dashboard's own state.db each time — the same-process handle leak behind the
    ``N live SessionDB handles`` precursor (#100896). Real store, real open, no fakes."""
    import hermes_state
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    seed = SessionDB(db_path=tmp_path / "state.db")
    seed.create_session("s1", source="cli")
    seed.append_message("s1", "user", "hello")
    seed.close()

    writable_opens = []
    real_init = SessionDB.__init__

    def spy(self, *args, **kwargs):
        if not kwargs.get("read_only"):
            writable_opens.append(kwargs)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(SessionDB, "__init__", spy)

    result = plugin_api.scan_sessions()

    assert result.get("error") is None
    assert [s["session_id"] for s in result["sessions"]] == ["s1"]
    assert writable_opens == [], "the achievements scan must attach read-only"


# ---- Locale / i18n (Accept-Language) ----
#
# Localization is a per-request overlay on the English snapshot cache: the API
# translates achievement name/description/category/criteria when the request
# negotiates a catalog we carry (zh-CN); tier/state stay English because the
# dashboard derives CSS classes, tier colors, and filter comparisons from them.
# Static UI chrome is translated by the host's shared web/src/i18n catalogs.


class _FakeRequest:
    """Duck-typed stand-in for fastapi.Request — only .headers.get() is needed."""

    def __init__(self, accept_language: str):
        self.headers = {"accept-language": accept_language}


def test_resolve_locale_maps_simplified_chinese_only(plugin_api):
    for header in ["zh", "zh-CN", "zh-cn", "zh-Hans", "zh-SG", "cmn", "zh-CN,zh;q=0.9,en;q=0.8"]:
        assert plugin_api._resolve_locale(_FakeRequest(header)) == "zh-CN"
    # Traditional Chinese tags (and every other language) fall back to English
    # until a zh-Hant catalog exists — Simplified text is worse than en there.
    for header in ["zh-TW", "zh-Hant", "zh-HK", "zh-MO", "en-US,en;q=0.9", "ja", "*", ""]:
        assert plugin_api._resolve_locale(_FakeRequest(header)) == "en"
    assert plugin_api._resolve_locale(None) == "en"


def test_localized_english_path_is_unchanged(plugin_api):
    for definition in plugin_api.ACHIEVEMENTS:
        for state in ("discovered", "secret"):
            item = {**definition, "state": state}
            assert plugin_api.localized_criteria(item, "en") == plugin_api.criteria_for(item)
            assert plugin_api.localized_display(plugin_api.display_achievement(item), "en") == plugin_api.display_achievement(item)


def test_english_criteria_never_contains_fullwidth_punctuation(plugin_api):
    """Regression: an earlier draft built one template for all locales, leaking
    "。等级阶梯:" into English criteria. English output must stay pure ASCII-
    punctuated English regardless of which catalogs ship beside the plugin."""
    for definition in plugin_api.ACHIEVEMENTS:
        for state in ("discovered", "secret"):
            text = plugin_api.criteria_for({**definition, "state": state})
            for ch in "。；：「」":
                assert ch not in text, f"{definition['id']}: {text!r}"


def test_zh_cn_criteria_translates_ladder_and_uses_local_punctuation(plugin_api):
    threshold = next(a for a in plugin_api.ACHIEVEMENTS if a["id"] == "let_him_cook")
    text = plugin_api.localized_criteria({**threshold, "state": "discovered"}, "zh-CN")
    assert text.startswith("要求：")
    assert "。等级阶梯：" in text
    assert "青铜 200" in text and "奥林匹亚 8000" in text
    assert "Copper" not in text and "Tier ladder" not in text and "Requirement" not in text

    multi = next(a for a in plugin_api.ACHIEVEMENTS if a.get("requirements"))
    text = plugin_api.localized_criteria({**multi, "state": "discovered"}, "zh-CN")
    assert "；" in text, "multi-condition criteria join with the locale separator"
    assert text.endswith("。")


def test_localized_payload_translates_dynamic_fields_keeps_machine_fields(plugin_api):
    snapshot = plugin_api._build_pending_snapshot(int(time.time()))
    catalog = json.loads((plugin_api._LOCALE_DIR / "zh-CN.json").read_text(encoding="utf-8"))

    items = plugin_api._localize_achievements(snapshot["achievements"], "zh-CN")
    assert len(items) == len(snapshot["achievements"])

    normal = next(a for a in plugin_api.ACHIEVEMENTS if not a.get("secret"))
    zh = next(a for a in items if a["id"] == normal["id"])
    assert zh["name"] == catalog[normal["id"]]["name"]
    assert zh["description"] == catalog[normal["id"]]["description"]
    assert zh["category"] == catalog[normal["id"]]["category"]

    # Machine fields stay English: tier names feed ha-tier-* CSS classes and the
    # tierHex color map; state strings are the visibility filter's comparison keys.
    for a in items:
        assert a["state"] in ("discovered", "unlocked", "secret")
        assert a.get("tier") is None or a["tier"] in plugin_api.TIER_NAMES
        assert a.get("next_tier") is None or a["next_tier"] in plugin_api.TIER_NAMES


def test_localized_secret_never_leaks_catalog_identity(plugin_api):
    """Regression: an earlier overlay applied catalog names before the secret
    check, so a zh-CN response revealed secret achievement names/descriptions."""
    snapshot = plugin_api._build_pending_snapshot(int(time.time()))
    secrets = [a for a in snapshot["achievements"] if a["state"] == "secret"]
    assert secrets, "catalog must contain secret achievements for this test to mean anything"

    for a in plugin_api._localize_achievements(secrets, "zh-CN"):
        assert a["name"] == "???"
        assert a["icon"] == "secret"
        assert a["criteria"].startswith("秘密：")
        assert a["description"] == "秘密成就：在 Hermes 检测到首个相关行为之前保持隐藏。"


def test_achievements_route_localizes_per_request(plugin_api, monkeypatch):
    snapshot = plugin_api._build_pending_snapshot(int(time.time()))
    monkeypatch.setattr(plugin_api, "evaluate_all", lambda force=False: snapshot)

    en = asyncio.run(plugin_api.achievements(_FakeRequest("en-US,en;q=0.9")))
    zh = asyncio.run(plugin_api.achievements(_FakeRequest("zh-CN,zh;q=0.9")))

    # English responses are byte-identical to the untranslated snapshot payload.
    assert en["achievements"] == snapshot["achievements"]
    assert "_ui" not in en  # static chrome belongs to web/src/i18n, not the payload

    # Chinese responses localize content but keep shape/counts identical.
    assert zh["total_count"] == en["total_count"]
    assert zh["unlocked_count"] == en["unlocked_count"]
    assert zh["secret_count"] == en["secret_count"]
    assert zh["scan_meta"] == en["scan_meta"]

    normal = next(a for a in plugin_api.ACHIEVEMENTS if not a.get("secret"))
    en_a = next(a for a in en["achievements"] if a["id"] == normal["id"])
    zh_a = next(a for a in zh["achievements"] if a["id"] == normal["id"])
    assert zh_a["name"] != en_a["name"]
    assert zh_a["description"] != en_a["description"]
    assert zh_a["category"] != en_a["category"]


def test_recent_unlocks_and_badges_routes_localize(plugin_api, monkeypatch):
    unlocked_item = {
        **next(a for a in plugin_api.ACHIEVEMENTS if not a.get("secret")),
        "unlocked": True, "discovered": True, "state": "unlocked",
        "unlocked_at": int(time.time()), "tier": "Gold",
    }
    snapshot = {
        **plugin_api._build_pending_snapshot(int(time.time())),
        "achievements": [dict(unlocked_item)],
    }
    monkeypatch.setattr(plugin_api, "evaluate_all", lambda force=False: snapshot)

    zh = asyncio.run(plugin_api.recent_unlocks(_FakeRequest("zh")))
    assert len(zh) == 1
    assert zh[0]["name"] != unlocked_item["name"]
    assert zh[0]["tier"] == "Gold"  # machine field untouched

    en = asyncio.run(plugin_api.recent_unlocks(_FakeRequest("en-GB")))
    assert en[0]["name"] == unlocked_item["name"]
