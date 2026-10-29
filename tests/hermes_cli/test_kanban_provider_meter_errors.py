"""M4 defect regression tests: non-meter errors triggering flap + exhausted token fail-over.

Two root causes confirmed (2026-10-29):
  Root A (kanban_db.py:9382-9389): Worker HTTP 400 -> mark_exhausted_and_rotate(400) ->
    _resolve_anthropic_pool_token skips exhausted entry -> anthropic absent from creds ->
    only codex as candidate.
  Root B (kanban_db.py:9393-9400): resolve_codex_runtime_credentials returns stale tokens
    even when pool entry is STATUS_EXHAUSTED/DEAD (device_code exhausted, 401, etc.) ->
    exhausted codex token becomes a candidate -> 401.

Tests use real AccountUsageSnapshot objects and exercise _read_assignee_credentials /
_remediate_provider_meter with minimal mocking (only external I/O and credential-file reads
are stubbed to isolate the test machine's real credentials).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.kanban_db as kb
from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from agent.credential_pool import PooledCredential, CredentialPool, STATUS_DEAD, STATUS_EXHAUSTED
from hermes_cli.kanban_db import _read_assignee_credentials, _remediate_provider_meter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snapshot(provider: str, used: float) -> AccountUsageSnapshot:
    """Real AccountUsageSnapshot with a single window."""
    return AccountUsageSnapshot(
        provider=provider,
        source="test",
        fetched_at=datetime.now(timezone.utc),
        windows=(AccountUsageWindow(label="Session", used_percent=used),),
    )


def _unavailable(provider: str, reason: str) -> AccountUsageSnapshot:
    return AccountUsageSnapshot(
        provider=provider,
        source="test",
        fetched_at=datetime.now(timezone.utc),
        windows=(),
        unavailable_reason=reason,
    )


def _make_pool_entry_obj(
    provider: str,
    access_token: str,
    *,
    last_status: Optional[str] = None,
    last_error_code: Optional[int] = None,
    source: str = "manual:device_code",
) -> PooledCredential:
    """Build a PooledCredential in-memory (no auth.json needed)."""
    return PooledCredential(
        provider=provider,
        id=f"test-{provider[:3]}-01",
        label=f"test-{provider}",
        auth_type="oauth",
        priority=0,
        source=source,
        access_token=access_token,
        last_status=last_status,
        last_status_at=time.time() - 30 if last_status else None,
        last_error_code=last_error_code,
    )


def _pool_from_entries(provider: str, entries: list[PooledCredential]) -> CredentialPool:
    """Build a CredentialPool from pre-made entries without I/O."""
    pool = CredentialPool(provider=provider, entries=entries)
    return pool


def _meter_settings(
    *,
    enabled: bool = True,
    claude_model: str = "claude-sonnet-4-6",
    codex_model: str = "gpt-5.6-luna",
    minimum_remaining_percent: int = 10,
) -> dict:
    return {
        "kanban": {
            "provider_meter_remediation": {
                "enabled": enabled,
                "claude_model": claude_model,
                "codex_model": codex_model,
                "minimum_remaining_percent": minimum_remaining_percent,
            }
        }
    }


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _make_ready_task(conn, assignee: str = "judo", *, model_override=None, provider_override=None) -> str:
    tid = kb.create_task(
        conn, title="test-task", assignee=assignee,
        model_override=model_override, provider_override=provider_override,
    )
    conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    conn.commit()
    return tid


def _provider_meter_events(conn, task_id: str) -> list:
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id=? ORDER BY id",
        (task_id,),
    ).fetchall()
    result = []
    for kind, payload_str in rows:
        payload = {}
        if payload_str:
            try:
                payload = json.loads(payload_str)
            except Exception:
                pass
        result.append({"kind": kind, "payload": payload})
    return [e for e in result if e["kind"] == "provider_meter_remediation"]


# ---------------------------------------------------------------------------
# Isolation helper: returns a context manager stack that prevents real
# credentials from seeding into pool loads.
# ---------------------------------------------------------------------------

def _no_real_creds():
    """Patch credential sources so tests don't read real tokens from the machine."""
    return (
        # Prevent Claude Code OAuth file read (anthropic singleton seeding)
        patch("agent.anthropic_adapter.read_claude_code_credentials", return_value=None),
        # Prevent anthropic env-var resolution
        patch.dict("os.environ", {
            "ANTHROPIC_TOKEN": "",
            "CLAUDE_CODE_OAUTH_TOKEN": "",
            "ANTHROPIC_API_KEY": "",
            "OPENAI_API_KEY": "",
        }),
    )


# ---------------------------------------------------------------------------
# (a) Root A fix: Worker 400 -> anthropic pool exhausted -> fallback path
#     should still find the token and include anthropic as a meter candidate.
# ---------------------------------------------------------------------------

class TestAnthropicFallbackAfterWorker400:
    """Fix 1: pool entry exhausted due to HTTP 400 must NOT block the meter query.

    _read_assignee_credentials must return the anthropic token even when the
    pool entry is STATUS_EXHAUSTED with last_error_code=400 (billing notice).
    The fallback path skips only 429/402 (quota) and DEAD entries.
    """

    def test_exhausted_400_anthropic_token_found_via_fallback(self, tmp_path):
        """Anthropic pool entry with last_error_code=400 -> token returned by
        _read_assignee_credentials (the new fallback path bypasses exhaustion).

        Uses real PooledCredential objects and the real _read_assignee_credentials path.
        load_pool seeding is suppressed by patching Claude Code credential reads.
        """
        judo_home = tmp_path / "judo"
        judo_home.mkdir()

        # Pool with one exhausted entry (HTTP 400 billing notice)
        exhausted_entry = _make_pool_entry_obj(
            "anthropic", "sk-ant-billing-400-token",
            last_status=STATUS_EXHAUSTED,
            last_error_code=400,
        )
        fake_pool = _pool_from_entries("anthropic", [exhausted_entry])

        with (
            patch("agent.anthropic_adapter.read_claude_code_credentials", return_value=None),
            patch.dict("os.environ", {
                "ANTHROPIC_TOKEN": "", "CLAUDE_CODE_OAUTH_TOKEN": "",
                "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "",
            }),
            patch("agent.credential_pool.load_pool", side_effect=lambda p: (
                fake_pool if p == "anthropic" else _pool_from_entries(p, [])
            )),
            patch("hermes_cli.auth.resolve_codex_runtime_credentials",
                  side_effect=Exception("no codex singleton")),
        ):
            creds = _read_assignee_credentials(judo_home)

        assert "anthropic" in creds, (
            "anthropic must be in creds even when pool entry is exhausted due to 400; "
            "HTTP 400 billing notice is NOT a quota signal"
        )
        assert creds["anthropic"][0] == "sk-ant-billing-400-token", (
            f"wrong token returned: {creds['anthropic'][0]!r}"
        )

    def test_exhausted_429_anthropic_token_NOT_included(self, tmp_path):
        """Anthropic pool entry with last_error_code=429 -> token MUST NOT be included.

        429 is genuine quota exhaustion; the fallback path must skip it.
        """
        judo_home = tmp_path / "judo"
        judo_home.mkdir()

        quota_entry = _make_pool_entry_obj(
            "anthropic", "sk-ant-quota-exhausted",
            last_status=STATUS_EXHAUSTED,
            last_error_code=429,
        )
        fake_pool = _pool_from_entries("anthropic", [quota_entry])

        with (
            patch("agent.anthropic_adapter.read_claude_code_credentials", return_value=None),
            patch.dict("os.environ", {
                "ANTHROPIC_TOKEN": "", "CLAUDE_CODE_OAUTH_TOKEN": "",
                "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "",
            }),
            patch("agent.credential_pool.load_pool", side_effect=lambda p: (
                fake_pool if p == "anthropic" else _pool_from_entries(p, [])
            )),
            patch("hermes_cli.auth.resolve_codex_runtime_credentials",
                  side_effect=Exception("no codex singleton")),
        ):
            creds = _read_assignee_credentials(judo_home)

        assert "anthropic" not in creds, (
            "anthropic must NOT be in creds when pool entry is 429-exhausted (quota signal)"
        )

    def test_dead_anthropic_token_not_included(self, tmp_path):
        """STATUS_DEAD anthropic entry must be excluded even if it has a token."""
        judo_home = tmp_path / "judo"
        judo_home.mkdir()

        dead_entry = _make_pool_entry_obj(
            "anthropic", "sk-ant-dead-token",
            last_status=STATUS_DEAD,
        )
        fake_pool = _pool_from_entries("anthropic", [dead_entry])

        with (
            patch("agent.anthropic_adapter.read_claude_code_credentials", return_value=None),
            patch.dict("os.environ", {
                "ANTHROPIC_TOKEN": "", "CLAUDE_CODE_OAUTH_TOKEN": "",
                "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "",
            }),
            patch("agent.credential_pool.load_pool", side_effect=lambda p: (
                fake_pool if p == "anthropic" else _pool_from_entries(p, [])
            )),
            patch("hermes_cli.auth.resolve_codex_runtime_credentials",
                  side_effect=Exception("no codex singleton")),
        ):
            creds = _read_assignee_credentials(judo_home)

        assert "anthropic" not in creds, (
            "DEAD anthropic entry must never be included as a meter credential"
        )

    def test_anthropic_400_dispatch_keeps_anthropic(self, kanban_home):
        """End-to-end via _remediate_provider_meter: after worker HTTP 400 exhausts the
        pool entry, the next dispatch tick must still select anthropic (not codex) when
        the meter shows anthropic has headroom.

        Before Fix 1: exhausted-by-400 entry not in creds -> only codex -> wrong.
        After Fix 1: fallback reads the token -> anthropic in creds -> meter OK -> selected.
        """
        judo_home = kanban_home / "profiles" / "judo"
        judo_home.mkdir(parents=True)

        # Anthropic: pool entry exhausted due to HTTP 400
        ant_exhausted = _make_pool_entry_obj(
            "anthropic", "sk-ant-valid-key",
            last_status=STATUS_EXHAUSTED,
            last_error_code=400,
        )
        ant_pool = _pool_from_entries("anthropic", [ant_exhausted])
        codex_pool = _pool_from_entries("openai-codex", [])

        with kb.connect() as conn:
            tid = _make_ready_task(conn, assignee="judo")

        spawned_providers: list = []

        def fake_fetch(provider, *, api_key=None, base_url=None):
            if provider == "anthropic":
                return _snapshot("anthropic", 20)  # 80% remaining — healthy
            return None

        with kb.connect() as conn:
            with (
                patch("hermes_cli.config.load_config", return_value=_meter_settings()),
                patch("agent.account_usage.fetch_account_usage", side_effect=fake_fetch),
                patch("agent.credential_pool.load_pool", side_effect=lambda p: (
                    ant_pool if p == "anthropic" else codex_pool
                )),
                patch("agent.anthropic_adapter.read_claude_code_credentials", return_value=None),
                patch.dict("os.environ", {
                    "ANTHROPIC_TOKEN": "", "CLAUDE_CODE_OAUTH_TOKEN": "",
                    "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "",
                }),
                patch("hermes_cli.auth.resolve_codex_runtime_credentials",
                      side_effect=Exception("no codex")),
                patch("hermes_cli.profiles.profile_exists", return_value=True),
            ):
                result = kb.dispatch_once(
                    conn,
                    spawn_fn=lambda task, ws, **_: spawned_providers.append(
                        task.provider_override) or 1234
                )

        assert tid in [t for t, _, _ in result.spawned], (
            "task should be spawned with anthropic despite its 400-exhausted pool entry"
        )
        assert "anthropic" in spawned_providers, (
            f"Expected anthropic but got {spawned_providers}; "
            "400 exhaustion must not block anthropic selection when meter shows headroom"
        )


# ---------------------------------------------------------------------------
# (b) Root B fix: codex pool exhausted -> excluded from candidates
# ---------------------------------------------------------------------------

class TestCodexExhaustedExclusion:
    """Fix 2: codex pool entry exhausted/dead -> codex must be excluded from
    _read_assignee_credentials, preventing fail-over to a 401-guaranteed token.
    """

    def test_exhausted_codex_pool_not_in_creds(self, tmp_path):
        """All codex pool entries STATUS_EXHAUSTED -> codex must not be in creds.

        _available_entries(clear_expired=False) returns nothing -> excluded.
        Uses the real _read_assignee_credentials path with controlled pool state.
        """
        judo_home = tmp_path / "judo"
        judo_home.mkdir()

        codex_exhausted = _make_pool_entry_obj(
            "openai-codex", "stale-codex-token",
            last_status=STATUS_EXHAUSTED,
            last_error_code=401,
            source="manual:device_code",
        )
        codex_pool = _pool_from_entries("openai-codex", [codex_exhausted])

        with (
            patch("agent.credential_pool.load_pool", side_effect=lambda p: (
                codex_pool if p == "openai-codex" else _pool_from_entries(p, [])
            )),
            patch("hermes_cli.auth.resolve_codex_runtime_credentials",
                  return_value={"api_key": "stale-codex-token", "base_url": None}),
            patch.dict("os.environ", {
                "ANTHROPIC_TOKEN": "", "OPENAI_API_KEY": "",
            }),
            patch("agent.anthropic_adapter.read_claude_code_credentials", return_value=None),
        ):
            creds = _read_assignee_credentials(judo_home)

        assert "openai-codex" not in creds, (
            "codex must be excluded when all pool entries are exhausted/dead; "
            "resolve_codex_runtime_credentials returns stale tokens that pool says are invalid"
        )

    def test_exhausted_codex_not_spawned_no_anthropic_either(self, kanban_home):
        """Codex exhausted + no anthropic credentials -> deferred event, no spawn.

        Before Fix 2: stale codex token in creds -> codex candidate -> 401 crash.
        After Fix 2: codex excluded -> no valid candidate -> spawn hold.
        """
        judo_home = kanban_home / "profiles" / "judo"
        judo_home.mkdir(parents=True)

        codex_exhausted = _make_pool_entry_obj(
            "openai-codex", "stale-codex-token",
            last_status=STATUS_EXHAUSTED,
            last_error_code=401,
            source="manual:device_code",
        )
        codex_pool = _pool_from_entries("openai-codex", [codex_exhausted])
        ant_pool = _pool_from_entries("anthropic", [])

        with kb.connect() as conn:
            tid = _make_ready_task(conn, assignee="judo")

        spawn_calls: list = []

        with kb.connect() as conn:
            with (
                patch("hermes_cli.config.load_config", return_value=_meter_settings()),
                patch("agent.account_usage.fetch_account_usage", return_value=None),
                patch("agent.credential_pool.load_pool", side_effect=lambda p: (
                    codex_pool if p == "openai-codex" else ant_pool
                )),
                patch("hermes_cli.auth.resolve_codex_runtime_credentials",
                      return_value={"api_key": "stale-codex-token", "base_url": None}),
                patch.dict("os.environ", {
                    "ANTHROPIC_TOKEN": "", "OPENAI_API_KEY": "",
                }),
                patch("agent.anthropic_adapter.read_claude_code_credentials", return_value=None),
                patch("hermes_cli.profiles.profile_exists", return_value=True),
            ):
                result = kb.dispatch_once(
                    conn,
                    spawn_fn=lambda *a, **kw: spawn_calls.append(1) or 1234,
                )

        assert tid not in [t for t, _, _ in result.spawned], (
            "exhausted codex must not be spawned"
        )
        assert not spawn_calls, "spawn_fn must not be called when no valid candidate"

    def test_codex_excluded_anthropic_selected(self, kanban_home):
        """Codex exhausted -> excluded; anthropic has headroom -> anthropic selected.

        Validates that codex exclusion doesn't cause fail-closed when anthropic is available.
        """
        judo_home = kanban_home / "profiles" / "judo"
        judo_home.mkdir(parents=True)

        ant_ok = _make_pool_entry_obj("anthropic", "sk-ant-valid")
        codex_exhausted = _make_pool_entry_obj(
            "openai-codex", "stale-codex-token",
            last_status=STATUS_EXHAUSTED,
            last_error_code=401,
            source="manual:device_code",
        )
        ant_pool = _pool_from_entries("anthropic", [ant_ok])
        codex_pool = _pool_from_entries("openai-codex", [codex_exhausted])

        with kb.connect() as conn:
            tid = _make_ready_task(conn, assignee="judo")

        spawned_providers: list = []

        def fake_fetch(provider, *, api_key=None, base_url=None):
            if provider == "anthropic":
                return _snapshot("anthropic", 20)
            return None

        with kb.connect() as conn:
            with (
                patch("hermes_cli.config.load_config", return_value=_meter_settings()),
                patch("agent.account_usage.fetch_account_usage", side_effect=fake_fetch),
                patch("agent.credential_pool.load_pool", side_effect=lambda p: (
                    ant_pool if p == "anthropic" else codex_pool
                )),
                patch("hermes_cli.auth.resolve_codex_runtime_credentials",
                      return_value={"api_key": "stale-codex-token", "base_url": None}),
                patch.dict("os.environ", {
                    "ANTHROPIC_TOKEN": "", "OPENAI_API_KEY": "",
                }),
                patch("agent.anthropic_adapter.read_claude_code_credentials", return_value=None),
                patch("hermes_cli.profiles.profile_exists", return_value=True),
            ):
                result = kb.dispatch_once(
                    conn,
                    spawn_fn=lambda task, ws, **_: spawned_providers.append(
                        task.provider_override) or 1234
                )

        assert tid in [t for t, _, _ in result.spawned], "anthropic should be spawned"
        assert "anthropic" in spawned_providers, (
            "anthropic must be selected when codex is excluded but anthropic has headroom"
        )


# ---------------------------------------------------------------------------
# (c) Fix 3+4: Switch event payload includes {from, to, reason, readings},
#     no-headroom event includes outcome='deferred'
# ---------------------------------------------------------------------------

class TestSwitchEventPayload:
    """Fix 3: provider switch event must carry readings so operators can audit
    why the switch happened.
    Fix 4: no_headroom event must carry outcome='deferred'.
    """

    def test_switched_event_has_readings(self, kanban_home):
        """When provider switches, event payload must include 'readings' key.

        Setup: anthropic at 95% used (< 10% threshold) -> switch to codex.
        Readings dict must appear in the event payload for both providers.
        """
        judo_home = kanban_home / "profiles" / "judo"
        judo_home.mkdir(parents=True)

        ant_ok = _make_pool_entry_obj("anthropic", "sk-ant-key")
        codex_ok = _make_pool_entry_obj("openai-codex", "codex-key")
        ant_pool = _pool_from_entries("anthropic", [ant_ok])
        codex_pool = _pool_from_entries("openai-codex", [codex_ok])

        with kb.connect() as conn:
            # Task currently on anthropic; meter will show codex has more headroom
            tid = _make_ready_task(
                conn, assignee="judo",
                model_override="claude-sonnet-4-6", provider_override="anthropic",
            )

        def fake_fetch(provider, *, api_key=None, base_url=None):
            if provider == "anthropic":
                return _snapshot("anthropic", 95)   # 5% remaining — below 10% threshold
            if provider == "openai-codex":
                return _snapshot("openai-codex", 20)  # 80% remaining — healthy
            return None

        with kb.connect() as conn:
            with (
                patch("hermes_cli.config.load_config", return_value=_meter_settings()),
                patch("agent.account_usage.fetch_account_usage", side_effect=fake_fetch),
                patch("agent.credential_pool.load_pool", side_effect=lambda p: (
                    ant_pool if p == "anthropic" else codex_pool
                )),
                patch("hermes_cli.auth.resolve_codex_runtime_credentials",
                      return_value={"api_key": "codex-key", "base_url": None}),
                patch.dict("os.environ", {
                    "ANTHROPIC_TOKEN": "", "OPENAI_API_KEY": "",
                }),
                patch("agent.anthropic_adapter.read_claude_code_credentials", return_value=None),
                patch("hermes_cli.profiles.profile_exists", return_value=True),
            ):
                kb.dispatch_once(conn, spawn_fn=lambda *a, **kw: 1234)

        with kb.connect() as conn:
            events = _provider_meter_events(conn, tid)

        switched = [e for e in events if e["payload"].get("reason") == "switched"]
        assert switched, f"expected a 'switched' event, got: {[e['payload'] for e in events]}"
        payload = switched[0]["payload"]

        assert "readings" in payload, (
            f"switched event must contain 'readings', got keys: {list(payload.keys())}"
        )
        readings = payload["readings"]
        assert isinstance(readings, dict), "readings must be a dict"
        # anthropic reading: 95% used
        assert "anthropic" in readings, "readings must include anthropic"
        ant_r = readings["anthropic"]
        assert ant_r is not None
        ant_windows = ant_r.get("windows", [])
        assert any(w.get("used_percent") == 95.0 for w in ant_windows), (
            f"anthropic window must show 95% used, got: {ant_windows}"
        )
        # codex reading: 20% used
        assert "openai-codex" in readings, "readings must include openai-codex"
        codex_r = readings["openai-codex"]
        assert codex_r is not None
        codex_windows = codex_r.get("windows", [])
        assert any(w.get("used_percent") == 20.0 for w in codex_windows), (
            f"codex window must show 20% used, got: {codex_windows}"
        )

    def test_no_headroom_event_has_outcome_deferred(self, kanban_home):
        """When no provider has headroom, event must carry outcome='deferred'.

        This distinguishes meter-based spawn-hold from credential-missing spawn-hold.
        """
        judo_home = kanban_home / "profiles" / "judo"
        judo_home.mkdir(parents=True)

        ant_ok = _make_pool_entry_obj("anthropic", "sk-ant-key")
        ant_pool = _pool_from_entries("anthropic", [ant_ok])
        codex_pool = _pool_from_entries("openai-codex", [])

        with kb.connect() as conn:
            tid = _make_ready_task(conn, assignee="judo")

        def fake_fetch(provider, *, api_key=None, base_url=None):
            return _snapshot(provider, 95)  # 5% remaining — below 10% threshold for all

        with kb.connect() as conn:
            with (
                patch("hermes_cli.config.load_config", return_value=_meter_settings()),
                patch("agent.account_usage.fetch_account_usage", side_effect=fake_fetch),
                patch("agent.credential_pool.load_pool", side_effect=lambda p: (
                    ant_pool if p == "anthropic" else codex_pool
                )),
                patch.dict("os.environ", {
                    "ANTHROPIC_TOKEN": "", "OPENAI_API_KEY": "",
                }),
                patch("agent.anthropic_adapter.read_claude_code_credentials", return_value=None),
                patch("hermes_cli.auth.resolve_codex_runtime_credentials",
                      side_effect=Exception("no codex")),
                patch("hermes_cli.profiles.profile_exists", return_value=True),
            ):
                kb.dispatch_once(conn, spawn_fn=lambda *a, **kw: 1234)

        with kb.connect() as conn:
            events = _provider_meter_events(conn, tid)

        deferred_events = [
            e for e in events if e["payload"].get("reason") == "no_headroom"
        ]
        assert deferred_events, (
            f"expected a no_headroom event, got: {[e['payload'] for e in events]}"
        )
        payload = deferred_events[0]["payload"]
        assert payload.get("outcome") == "deferred", (
            f"no_headroom event must have outcome='deferred', got {payload.get('outcome')!r}"
        )


# ---------------------------------------------------------------------------
# (d) Credential validation fail-closed: load_pool / _available_entries raise
# ---------------------------------------------------------------------------

class TestCredentialValidationFailClosed:
    """Fix: pool validation exceptions in 2c-validate / 2b-fallback must be fail-closed.

    Before fix: except Exception: pass — stale token survives, codex spawned with 401.
    After fix:  except Exception: result.pop(...) — validation impossible → excluded.
    """

    def test_load_pool_raises_during_codex_validation_excludes_codex(self, tmp_path):
        """2c-validate: load_pool() raises → codex must NOT remain in creds.

        resolve_codex_runtime_credentials resolves a token (codex enters result),
        then load_pool("openai-codex") raises during the validation step.
        Fail-closed: the stale token must be excluded.
        """
        judo_home = tmp_path / "judo"
        judo_home.mkdir()

        def _pool_side_effect(provider):
            if provider == "openai-codex":
                raise RuntimeError("auth.json unreadable — simulated I/O error")
            return _pool_from_entries(provider, [])

        with (
            patch("agent.credential_pool.load_pool", side_effect=_pool_side_effect),
            patch("hermes_cli.auth.resolve_codex_runtime_credentials",
                  return_value={"api_key": "stale-codex-token", "base_url": None}),
            patch("agent.anthropic_adapter.read_claude_code_credentials", return_value=None),
            patch.dict("os.environ", {
                "ANTHROPIC_TOKEN": "", "CLAUDE_CODE_OAUTH_TOKEN": "",
                "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "",
            }),
        ):
            creds = _read_assignee_credentials(judo_home)

        assert "openai-codex" not in creds, (
            "codex must be excluded when load_pool raises during 2c-validate; "
            "validation impossible → fail-closed, not fail-open with stale 401 token"
        )

    def test_available_entries_raises_during_codex_validation_excludes_codex(self, tmp_path):
        """2c-validate: _available_entries() raises → codex must NOT remain in creds.

        resolve_codex_runtime_credentials resolves a token; load_pool returns a mock
        where has_credentials() is True and _available_entries() raises.
        Fail-closed: the unverified token must be excluded.
        """
        judo_home = tmp_path / "judo"
        judo_home.mkdir()

        broken_pool = MagicMock()
        broken_pool.has_credentials.return_value = True
        broken_pool._available_entries.side_effect = RuntimeError("pool corrupted")

        def _pool_side_effect(provider):
            if provider == "openai-codex":
                return broken_pool
            return _pool_from_entries(provider, [])

        with (
            patch("agent.credential_pool.load_pool", side_effect=_pool_side_effect),
            patch("hermes_cli.auth.resolve_codex_runtime_credentials",
                  return_value={"api_key": "stale-codex-token", "base_url": None}),
            patch("agent.anthropic_adapter.read_claude_code_credentials", return_value=None),
            patch.dict("os.environ", {
                "ANTHROPIC_TOKEN": "", "CLAUDE_CODE_OAUTH_TOKEN": "",
                "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "",
            }),
        ):
            creds = _read_assignee_credentials(judo_home)

        assert "openai-codex" not in creds, (
            "codex must be excluded when _available_entries raises during 2c-validate; "
            "partial validation (pool broken) is not sufficient proof of usability"
        )

    def test_load_pool_raises_during_anthropic_2b_fallback_excludes_anthropic(self, tmp_path):
        """2b-fallback: load_pool("anthropic") raises → anthropic must NOT be in creds.

        _resolve_anthropic_pool_token fails (no singleton token), then load_pool
        for the fallback path raises. Fail-closed: no anthropic candidate.
        """
        judo_home = tmp_path / "judo"
        judo_home.mkdir()

        def _pool_side_effect(provider):
            if provider == "anthropic":
                raise RuntimeError("auth.json locked — simulated I/O error")
            return _pool_from_entries(provider, [])

        with (
            patch("agent.credential_pool.load_pool", side_effect=_pool_side_effect),
            # Force 2b to fail so 2b-fallback runs
            patch("agent.anthropic_adapter._resolve_anthropic_pool_token",
                  side_effect=Exception("no singleton token")),
            patch("agent.anthropic_adapter.read_claude_code_credentials", return_value=None),
            patch("hermes_cli.auth.resolve_codex_runtime_credentials",
                  side_effect=Exception("no codex")),
            patch.dict("os.environ", {
                "ANTHROPIC_TOKEN": "", "CLAUDE_CODE_OAUTH_TOKEN": "",
                "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "",
            }),
        ):
            creds = _read_assignee_credentials(judo_home)

        assert "anthropic" not in creds, (
            "anthropic must not be in creds when load_pool raises during 2b-fallback; "
            "validation impossible → fail-closed"
        )

    # ------------------------------------------------------------------
    # Regression: 2c-validate fail-closed must survive OPENAI_API_KEY
    # present in the profile .env (the re-insertion bug, t_9894ca23).
    # ------------------------------------------------------------------

    def test_load_pool_raises_codex_validation_profile_env_key_not_inserted(self, tmp_path):
        """2c-validate fail-closed: load_pool raises, but profile .env has OPENAI_API_KEY.

        Reproduction of t_9894ca23: after result.pop("openai-codex") in the
        except branch, the 2d OPENAI_API_KEY env-var fallback was checking
        ``if "openai-codex" not in result`` (True) and re-inserting the
        provider.  Expected result: ``{'openai-codex': ('env-api-key', None)}``.

        After fix: _codex_fail_closed=True blocks the 2d re-insertion.
        """
        judo_home = tmp_path / "judo"
        judo_home.mkdir()
        # Write OPENAI_API_KEY to the profile .env file — the code reads it from
        # disk, NOT from os.environ.  Existing tests missed this because they only
        # patched os.environ while the code reads hermes_home/.env.
        (judo_home / ".env").write_text("OPENAI_API_KEY=env-api-key\n", encoding="utf-8")

        def _pool_side_effect(provider):
            if provider == "openai-codex":
                raise RuntimeError("auth.json unreadable — simulated I/O error")
            return _pool_from_entries(provider, [])

        with (
            patch("agent.credential_pool.load_pool", side_effect=_pool_side_effect),
            patch("hermes_cli.auth.resolve_codex_runtime_credentials",
                  return_value={"api_key": "stale-codex-token", "base_url": None}),
            patch("agent.anthropic_adapter.read_claude_code_credentials", return_value=None),
            patch.dict("os.environ", {
                "ANTHROPIC_TOKEN": "", "CLAUDE_CODE_OAUTH_TOKEN": "",
                "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "",
            }),
        ):
            creds = _read_assignee_credentials(judo_home)

        assert "openai-codex" not in creds, (
            "codex must remain excluded even when profile .env has OPENAI_API_KEY; "
            "2c-validate fail-closed (load_pool raised) must not be overridden by 2d fallback. "
            f"Actual creds: {creds}"
        )

    def test_exhausted_codex_pool_profile_env_key_not_inserted(self, tmp_path):
        """2c-validate fail-closed: pool exhausted, but profile .env has OPENAI_API_KEY.

        Same re-insertion bug as above but triggered via the _available_entries
        branch: pool.has_credentials() True, _available_entries returns [].
        The 2d fallback must NOT re-insert openai-codex via the env key.
        """
        judo_home = tmp_path / "judo"
        judo_home.mkdir()
        (judo_home / ".env").write_text("OPENAI_API_KEY=env-api-key\n", encoding="utf-8")

        codex_exhausted = _make_pool_entry_obj(
            "openai-codex", "stale-codex-token",
            last_status=STATUS_EXHAUSTED,
            last_error_code=401,
        )
        codex_pool = _pool_from_entries("openai-codex", [codex_exhausted])

        with (
            patch("agent.credential_pool.load_pool",
                  side_effect=lambda p: codex_pool if p == "openai-codex" else _pool_from_entries(p, [])),
            patch("hermes_cli.auth.resolve_codex_runtime_credentials",
                  return_value={"api_key": "stale-codex-token", "base_url": None}),
            patch("agent.anthropic_adapter.read_claude_code_credentials", return_value=None),
            patch.dict("os.environ", {
                "ANTHROPIC_TOKEN": "", "CLAUDE_CODE_OAUTH_TOKEN": "",
                "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "",
            }),
        ):
            creds = _read_assignee_credentials(judo_home)

        assert "openai-codex" not in creds, (
            "codex must remain excluded even when profile .env has OPENAI_API_KEY; "
            "2c-validate (exhausted pool) must not be overridden by 2d env fallback. "
            f"Actual creds: {creds}"
        )

    def test_2c_resolver_raises_profile_env_key_is_valid_fallback(self, tmp_path):
        """2c resolver itself raises (not 2c-validate): OPENAI_API_KEY env fallback must work.

        When resolve_codex_runtime_credentials raises entirely, openai-codex is never
        added to result — 2c-validate's inner block never runs — and _codex_fail_closed
        stays False.  The 2d env-key fallback is the intended last resort and must fire.

        This test guards against over-restricting the fix: the _codex_fail_closed flag
        must NOT block 2d when 2c simply failed to resolve anything.
        """
        judo_home = tmp_path / "judo"
        judo_home.mkdir()
        (judo_home / ".env").write_text("OPENAI_API_KEY=env-api-key\n", encoding="utf-8")

        with (
            patch("agent.credential_pool.load_pool",
                  side_effect=lambda p: _pool_from_entries(p, [])),
            # 2c raises entirely → openai-codex never enters result
            patch("hermes_cli.auth.resolve_codex_runtime_credentials",
                  side_effect=RuntimeError("no singleton configured")),
            patch("agent.anthropic_adapter.read_claude_code_credentials", return_value=None),
            patch.dict("os.environ", {
                "ANTHROPIC_TOKEN": "", "CLAUDE_CODE_OAUTH_TOKEN": "",
                "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "",
            }),
        ):
            creds = _read_assignee_credentials(judo_home)

        assert "openai-codex" in creds, (
            "OPENAI_API_KEY in profile .env must be used as fallback when "
            "resolve_codex_runtime_credentials raises entirely (2c-validate never ran); "
            "_codex_fail_closed must NOT block 2d in this path. "
            f"Actual creds: {creds}"
        )
        assert creds["openai-codex"] == ("env-api-key", None), (
            f"Expected ('env-api-key', None), got {creds['openai-codex']!r}"
        )
