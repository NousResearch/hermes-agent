"""Regression guards for the Discord slash-command sync loss/no-recovery class.

Measured incident (2026-09-20): an app whose tree defines ``/stop`` and
``/restart`` had BOTH commands absent from Discord's live global registry while
every sync attempt that day was rate-limited (8 attempts, zero successes). Three
defects compose into that outcome:

1. ``_safe_sync_slash_commands`` handled a "recreated" command as
   ``delete_global_command`` FOLLOWED BY ``upsert_global_command``. A 429
   landing between the two removes the command from Discord with nothing left
   to restore it (inferred mechanism for the two lost commands).
2. A 429 aborted the whole sync and persisted ``retry_after_until``, but
   NOTHING re-ran the sync when that deadline elapsed — recovery depended on
   the next reconnect.
3. Each reconnect restarted the full diff and burned Discord's small per-app
   command-management bucket again.

Plus: nothing ever compared the tree's desired command names against Discord's
live registry, so a missing command was only discoverable by a user typing it.
"""

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    if sys.modules.get("discord") is None:
        discord_mod = MagicMock()
        discord_mod.Intents.default.return_value = MagicMock()
        sys.modules["discord"] = discord_mod
        sys.modules.setdefault("discord.ext", MagicMock())
        sys.modules.setdefault("discord.ext.commands", MagicMock())


_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class _RateLimited(Exception):
    """Duck-typed stand-in for discord.py's RateLimited (429)."""

    def __init__(self, retry_after: float):
        super().__init__(f"rate limited, retry in {retry_after}s")
        self.retry_after = retry_after
        self.status = 429


class _DesiredCommand:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self, tree):
        return dict(self._payload)


class _ExistingCommand:
    def __init__(self, command_id, payload):
        self.id = command_id
        self.name = payload["name"]
        self.type = SimpleNamespace(value=payload.get("type", 1))
        self._payload = payload

    def to_dict(self):
        return {
            "id": self.id,
            "application_id": 999,
            **self._payload,
            "name_localizations": {},
            "description_localizations": {},
        }


def _payload(name, **overrides):
    base = {
        "name": name,
        "description": f"{name} command",
        "type": 1,
        "options": [],
        "nsfw": False,
        "dm_permission": True,
        "default_member_permissions": None,
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _no_mutation_pacing(monkeypatch):
    monkeypatch.setattr(
        DiscordAdapter,
        "_command_sync_mutation_interval_seconds",
        lambda self: 0.0,
    )


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    return DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))


# ---------------------------------------------------------------------------
# (a) A 429 on a "recreated" command must never leave the command deleted.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limited_recreate_never_deletes_the_command(adapter):
    """A 429 during the sync must not vacate a live command.

    ``/stop`` exists on Discord but needs a metadata-only replace (the payload
    is patch-identical, so the old code took the delete-then-upsert
    "recreated" branch). We fail the mutation with a 429. The command must
    still be present in Discord's registry afterwards — i.e. the sync must
    never open a delete-before-create window.
    """
    live = {"stop": _ExistingCommand(11, _payload("stop"))}
    desired = _payload("stop", nsfw=True)  # patch-identical, canonical differs

    async def _delete(app_id, command_id):
        live.pop("stop", None)

    async def _upsert(app_id, payload):
        raise _RateLimited(120.0)

    async def _bulk_sync():
        # Discord's PUT bulk-overwrite is atomic: a 429 changes nothing.
        raise _RateLimited(120.0)

    fake_tree = SimpleNamespace(
        get_commands=lambda: [_DesiredCommand(desired)],
        fetch_commands=AsyncMock(return_value=list(live.values())),
        sync=AsyncMock(side_effect=_bulk_sync),
    )
    adapter._client = SimpleNamespace(
        tree=fake_tree,
        http=SimpleNamespace(
            upsert_global_command=AsyncMock(side_effect=_upsert),
            edit_global_command=AsyncMock(),
            delete_global_command=AsyncMock(side_effect=_delete),
        ),
        application_id=999,
        user=SimpleNamespace(id=999),
    )

    with pytest.raises(Exception):
        await adapter._safe_sync_slash_commands()

    assert "stop" in live, (
        "a rate-limited sync deleted /stop and never restored it — the "
        "delete-then-create window is exactly how a live command vanishes"
    )


@pytest.mark.asyncio
async def test_sync_with_a_recreate_uses_one_atomic_bulk_overwrite(adapter):
    """Under the 100-command cap, any create/recreate goes through one PUT."""
    existing = _ExistingCommand(11, _payload("stop"))
    desired = _payload("stop", nsfw=True)

    fake_tree = SimpleNamespace(
        get_commands=lambda: [_DesiredCommand(desired)],
        fetch_commands=AsyncMock(return_value=[existing]),
        sync=AsyncMock(return_value=[]),
    )
    fake_http = SimpleNamespace(
        upsert_global_command=AsyncMock(),
        edit_global_command=AsyncMock(),
        delete_global_command=AsyncMock(),
    )
    adapter._client = SimpleNamespace(
        tree=fake_tree,
        http=fake_http,
        application_id=999,
        user=SimpleNamespace(id=999),
    )

    summary = await adapter._safe_sync_slash_commands()

    fake_tree.sync.assert_awaited_once()
    fake_http.delete_global_command.assert_not_awaited()
    fake_http.upsert_global_command.assert_not_awaited()
    assert summary["recreated"] == 1
    assert summary["total"] == 1


# ---------------------------------------------------------------------------
# (b) After a 429 the sync re-runs itself — no reconnect required.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limit_schedules_an_in_process_retry(adapter, monkeypatch):
    """A 429 must arm an asyncio timer that re-runs the sync on its own."""
    adapter._client = SimpleNamespace(
        tree=SimpleNamespace(get_commands=lambda: [_DesiredCommand(_payload("stop"))]),
        application_id=999,
        user=SimpleNamespace(id=999),
    )

    summary = {
        "total": 1,
        "unchanged": 1,
        "updated": 0,
        "recreated": 0,
        "created": 0,
        "deleted": 0,
    }
    sync = AsyncMock(side_effect=[_RateLimited(90.0), summary])
    monkeypatch.setattr(adapter, "_safe_sync_slash_commands", sync)

    slept = []

    async def _fast_sleep(delay):
        slept.append(delay)

    monkeypatch.setattr(discord_platform.asyncio, "sleep", _fast_sleep)

    await adapter._run_post_connect_initialization()

    task = adapter._command_sync_retry_task
    assert task is not None, "a 429 must schedule an in-process sync retry"
    await task

    assert sync.await_count == 2, (
        "the scheduled retry must re-run the sync without waiting for a reconnect"
    )
    assert slept and slept[0] >= 90.0, (
        f"retry must not fire before Discord's retry_after (slept={slept})"
    )

    state_path = (
        adapter._command_sync_state_path()
    )
    entry = json.loads(state_path.read_text(encoding="utf-8"))["999"]
    assert entry.get("last_success_at"), "the retry's success must be persisted"


@pytest.mark.asyncio
async def test_retry_backoff_is_bounded_and_persisted(adapter, monkeypatch):
    """Repeated 429s escalate the backoff and stop at the attempt cap."""
    adapter._client = SimpleNamespace(
        tree=SimpleNamespace(get_commands=lambda: [_DesiredCommand(_payload("stop"))]),
        application_id=999,
        user=SimpleNamespace(id=999),
    )
    monkeypatch.setattr(
        adapter,
        "_safe_sync_slash_commands",
        AsyncMock(side_effect=_RateLimited(1.0)),
    )

    async def _fast_sleep(delay):
        return None

    monkeypatch.setattr(discord_platform.asyncio, "sleep", _fast_sleep)

    await adapter._run_post_connect_initialization()
    for _ in range(discord_platform._DISCORD_COMMAND_SYNC_RETRY_MAX_ATTEMPTS + 2):
        task = adapter._command_sync_retry_task
        if task is None:
            break
        await task

    entry = json.loads(
        adapter._command_sync_state_path().read_text(encoding="utf-8")
    )["999"]
    attempts = entry.get("retry_attempts", 0)
    assert attempts <= discord_platform._DISCORD_COMMAND_SYNC_RETRY_MAX_ATTEMPTS, (
        f"retries must be bounded; got {attempts}"
    )
    assert adapter._command_sync_retry_task is None, (
        "the retry chain must stop once the attempt cap is reached"
    )
    assert entry.get("retry_backoff_seconds", 0) > 1.0, (
        "the backoff must escalate and be persisted across attempts"
    )


# ---------------------------------------------------------------------------
# (c) A reconnect inside the backoff window must not burn the bucket.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconnect_inside_backoff_window_skips_the_sync(adapter, monkeypatch, caplog):
    """A reconnect shortly after a rate-limited attempt must skip, not re-diff.

    ``retry_after_until`` has already elapsed, but the escalating backoff window
    has not. Apollo reconnected 5+ times in one day; each reconnect restarted
    the diff and re-burned Discord's per-app command bucket.
    """
    adapter._client = SimpleNamespace(
        tree=SimpleNamespace(get_commands=lambda: [_DesiredCommand(_payload("stop"))]),
        application_id=999,
        user=SimpleNamespace(id=999),
    )
    fingerprint = adapter._desired_command_sync_fingerprint()

    import time as _time

    now = _time.time()
    state_path = adapter._command_sync_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "999": {
                    "fingerprint": fingerprint,
                    "last_attempt_at": now - 10,
                    "retry_after_until": now - 5,  # elapsed
                    "retry_after": 5.0,
                    "retry_attempts": 2,
                    "retry_backoff_seconds": 300.0,
                    "backoff_until": now + 290,  # still inside the window
                }
            }
        ),
        encoding="utf-8",
    )

    sync = AsyncMock()
    monkeypatch.setattr(adapter, "_safe_sync_slash_commands", sync)

    with caplog.at_level("INFO"):
        await adapter._run_post_connect_initialization()

    sync.assert_not_awaited()
    assert any(
        "backoff" in record.getMessage().lower() for record in caplog.records
    ), f"the skip must log WHY; got {[r.getMessage() for r in caplog.records]}"


# ---------------------------------------------------------------------------
# (d) Drift detector.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_drift_detector_logs_missing_commands(adapter, caplog):
    """A desired command absent from Discord's live registry must WARN."""
    desired = [
        _DesiredCommand(_payload("stop")),
        _DesiredCommand(_payload("restart")),
        _DesiredCommand(_payload("status")),
    ]
    live = [
        _ExistingCommand(1, _payload("status")),
        _ExistingCommand(2, _payload("ghost")),
    ]
    adapter._client = SimpleNamespace(
        tree=SimpleNamespace(
            get_commands=lambda: desired,
            fetch_commands=AsyncMock(return_value=live),
        ),
        http=SimpleNamespace(),
        application_id=999,
        user=SimpleNamespace(id=999),
    )

    with caplog.at_level("WARNING"):
        drift = await adapter._check_command_registry_drift()

    assert drift is not None
    assert drift["missing"] == ["restart", "stop"]
    assert drift["extra"] == ["ghost"]

    messages = [r.getMessage() for r in caplog.records]
    assert any("PHASE=discord_command_registry_drift" in m for m in messages), (
        f"drift must be emitted as a greppable PHASE line; got {messages}"
    )
    phase_line = next(m for m in messages if "PHASE=discord_command_registry_drift" in m)
    assert "missing=['restart', 'stop']" in phase_line
    assert "extra=['ghost']" in phase_line


@pytest.mark.asyncio
async def test_registry_drift_detector_silent_when_in_sync(adapter, caplog):
    """No drift → no warning (the detector must not become noise)."""
    desired = [_DesiredCommand(_payload("stop"))]
    live = [_ExistingCommand(1, _payload("stop"))]
    adapter._client = SimpleNamespace(
        tree=SimpleNamespace(
            get_commands=lambda: desired,
            fetch_commands=AsyncMock(return_value=live),
        ),
        http=SimpleNamespace(),
        application_id=999,
        user=SimpleNamespace(id=999),
    )

    with caplog.at_level("WARNING"):
        drift = await adapter._check_command_registry_drift()

    assert drift == {"missing": [], "extra": []}
    assert not [
        r for r in caplog.records if "discord_command_registry_drift" in r.getMessage()
    ]


@pytest.mark.asyncio
async def test_successful_sync_runs_the_drift_check(adapter, monkeypatch):
    """The detector must run right after a successful sync, not only hourly."""
    adapter._client = SimpleNamespace(
        tree=SimpleNamespace(get_commands=lambda: [_DesiredCommand(_payload("stop"))]),
        application_id=999,
        user=SimpleNamespace(id=999),
    )
    summary = {
        "total": 1,
        "unchanged": 1,
        "updated": 0,
        "recreated": 0,
        "created": 0,
        "deleted": 0,
    }
    monkeypatch.setattr(
        adapter, "_safe_sync_slash_commands", AsyncMock(return_value=summary)
    )
    drift_check = AsyncMock(return_value={"missing": [], "extra": []})
    monkeypatch.setattr(adapter, "_check_command_registry_drift", drift_check)

    await adapter._run_post_connect_initialization()

    drift_check.assert_awaited()


# ---------------------------------------------------------------------------
# (e) Review follow-ups (#117299): the retry arithmetic must honour the backoff.
# ---------------------------------------------------------------------------


def _write_state(adapter, fingerprint, **fields):
    state_path = adapter._command_sync_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"fingerprint": fingerprint}
    entry.update(fields)
    state_path.write_text(json.dumps({"999": entry}), encoding="utf-8")


def _stub_client(adapter, name="stop"):
    adapter._client = SimpleNamespace(
        tree=SimpleNamespace(get_commands=lambda: [_DesiredCommand(_payload(name))]),
        application_id=999,
        user=SimpleNamespace(id=999),
    )
    return adapter._desired_command_sync_fingerprint()


def test_retry_delay_waits_out_the_escalating_backoff(adapter):
    """The in-process retry must not spend the budget inside one backoff window.

    Deriving the delay from ``retry_after_until`` alone burns all 5 attempts
    within ~25s of the first 429 (5s retry_after) while the first backoff
    window is 300s -- the same bucket-burning this change exists to stop, and
    contradicting the documented "exponential backoff".
    """
    import time as _time

    fingerprint = _stub_client(adapter)
    now = _time.time()
    _write_state(
        adapter,
        fingerprint,
        retry_after_until=now + 5,      # Discord's short retry-after
        backoff_until=now + 300,        # the escalating window
        retry_attempts=1,
    )

    delay = adapter._command_sync_retry_delay(999)
    assert delay is not None
    # Jitter is additive and bounded; the floor is the backoff, not retry_after.
    assert delay >= 290, f"retry fired inside the backoff window: {delay}s"


def test_changed_command_set_is_not_held_by_a_stale_backoff(adapter, monkeypatch):
    """A CHANGED fingerprint is fresh intent, not continued refusal.

    ``_record_command_sync_rate_limit`` already resets ``retry_attempts`` on a
    changed fingerprint. The backoff gate did not, so a newly registered
    command could stay missing from the picker for up to an hour after the
    restart that introduced it.
    """
    import time as _time

    fingerprint = _stub_client(adapter)
    now = _time.time()
    # State was written for a DIFFERENT desired command set.
    _write_state(
        adapter,
        "a-different-fingerprint",
        last_attempt_at=now - 10,
        retry_after_until=now - 5,   # elapsed
        backoff_until=now + 3000,    # a long stale window
        retry_attempts=5,
    )

    reason = adapter._command_sync_skip_reason(999, fingerprint)
    assert reason is None, f"a changed command set must not be gated: {reason}"

    # ...and the SAME fingerprint is still gated.
    _write_state(
        adapter,
        fingerprint,
        last_attempt_at=now - 10,
        retry_after_until=now - 5,
        backoff_until=now + 3000,
        retry_attempts=5,
    )
    assert "backoff" in (adapter._command_sync_skip_reason(999, fingerprint) or "").lower()


@pytest.mark.asyncio
async def test_failing_drift_check_is_throttled_like_a_successful_one(adapter):
    """A FAILING check must stamp the attempt, or it re-fires every tick.

    The caller is the liveness probe (default 15s). Stamping only on success
    turns a persistently failing check -- the incident's own state -- into a
    GET every 15s instead of once an hour.
    """
    failing_tree = SimpleNamespace(
        get_commands=lambda: [_DesiredCommand(_payload("stop"))],
        fetch_commands=AsyncMock(side_effect=_RateLimited(30.0)),
    )
    adapter._client = SimpleNamespace(tree=failing_tree, application_id=999,
                                      user=SimpleNamespace(id=999))

    await adapter._maybe_check_command_registry_drift()
    assert failing_tree.fetch_commands.await_count == 1

    # A second tick inside the interval must be throttled despite the failure.
    await adapter._maybe_check_command_registry_drift()
    assert failing_tree.fetch_commands.await_count == 1, (
        "a failing drift check re-fired instead of being throttled"
    )


@pytest.mark.asyncio
async def test_drift_check_respects_a_disabled_sync_policy(adapter, monkeypatch):
    """policy=off means the operator does not want us policing the registry."""
    tree = SimpleNamespace(
        get_commands=lambda: [_DesiredCommand(_payload("stop"))],
        fetch_commands=AsyncMock(return_value=[]),
    )
    adapter._client = SimpleNamespace(tree=tree, application_id=999,
                                      user=SimpleNamespace(id=999))
    monkeypatch.setattr(adapter, "_get_discord_command_sync_policy", lambda: "off")

    await adapter._maybe_check_command_registry_drift()
    tree.fetch_commands.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_post_connect_inits_are_serialized(adapter, monkeypatch):
    """A retry firing AT backoff_until can race a reconnect; only one may sync."""
    _stub_client(adapter)
    monkeypatch.setattr(adapter, "_command_sync_skip_reason", lambda *a, **k: None)
    monkeypatch.setattr(adapter, "_maybe_check_command_registry_drift", AsyncMock())

    inflight = 0
    peak = 0

    async def _slow_sync():
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        await asyncio.sleep(0.05)
        inflight -= 1
        return {"created": [], "updated": [], "deleted": []}

    monkeypatch.setattr(adapter, "_safe_sync_slash_commands", _slow_sync)

    await asyncio.gather(
        adapter._run_post_connect_initialization(),
        adapter._run_post_connect_initialization(is_rate_limit_retry=True),
    )

    assert peak == 1, f"two syncs overlapped (peak concurrency {peak})"
