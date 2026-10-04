"""Provider-wall notice: one line per unusable provider:model pair, once per incident."""

import json
import time

import pytest

from agent import provider_wall_notice as wall


class _Entry:
    """Minimal stand-in for ``credential_pool.PooledCredential`` (only the fields the notice reads)."""

    def __init__(self, **fields):
        self.last_status = None
        self.last_error_code = None
        self.last_error_reason = None
        self.last_error_message = None
        self.last_error_reset_at = None
        self.failure_reason = None
        self.label = "cred"
        for key, value in fields.items():
            setattr(self, key, value)


class _Pool:
    def __init__(self, entries):
        self._entries = list(entries)

    def entries(self):
        return list(self._entries)


class _Agent:
    def __init__(self, *, provider="custom", model="deepseek-v4-flash", chain=None, ctx=None):
        self.provider = provider
        self.model = model
        self._fallback_chain = list(chain or [])
        self._last_api_error_context = dict(ctx or {})
        self.notice_callback = None


@pytest.fixture
def isolated_marker(tmp_path, monkeypatch):
    """Keep the marker in tmp_path and the profile name deterministic."""
    monkeypatch.setattr(wall, "marker_path", lambda home=None: tmp_path / wall.WALL_MARKER_NAME)
    monkeypatch.setattr(wall, "_profile_name", lambda: "default")
    return tmp_path / wall.WALL_MARKER_NAME


def _patch_pool(monkeypatch, pools):
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: pools[provider])
    monkeypatch.setattr(wall, "_pool_key", lambda provider, base_url: provider)


# ── route rows ───────────────────────────────────────────────────────────


def test_billing_wall_row_reads_the_pool(monkeypatch):
    _patch_pool(monkeypatch, {"deepseek": _Pool([_Entry(
        last_status="exhausted", last_error_code=402, failure_reason="billing",
        last_error_message="Insufficient Balance (request_id: 70c3)",
    )])})

    row = wall.row_for_route("deepseek", "deepseek-v4-flash")

    assert row.status == wall.WALLED
    assert "Insufficient Balance" in row.detail
    assert row.emoji == "🔴"


def test_long_rate_limit_reset_reads_as_a_wall_and_a_short_one_as_cooling(monkeypatch):
    now = time.time()
    _patch_pool(monkeypatch, {"custom": _Pool([_Entry(
        last_status="exhausted", last_error_code=429, last_error_message="Requests are too frequent",
        last_error_reset_at=now + 6 * 3600,
    )])})
    assert wall.row_for_route("custom", "m").status == wall.WALLED

    _patch_pool(monkeypatch, {"custom": _Pool([_Entry(
        last_status="exhausted", last_error_code=429, last_error_message="Requests are too frequent",
        last_error_reset_at=now + 60,
    )])})
    assert wall.row_for_route("custom", "m").status == wall.COOLING


def test_healthy_pool_reads_as_available(monkeypatch):
    _patch_pool(monkeypatch, {"openrouter": _Pool([_Entry(request_count=3)])})

    assert wall.row_for_route("openrouter", "x/y").status == wall.AVAILABLE


def test_pool_read_failure_is_unknown_not_an_exception(monkeypatch):
    def _boom(provider):
        raise RuntimeError("auth.json unreadable")

    monkeypatch.setattr("agent.credential_pool.load_pool", _boom)
    monkeypatch.setattr(wall, "_pool_key", lambda provider, base_url: provider)

    row = wall.row_for_route("custom", "m")

    assert row.status == wall.UNKNOWN
    assert "unreadable" in row.detail


# ── incident rows ────────────────────────────────────────────────────────


def test_incident_rows_are_primary_plus_unusable_chain_entries(monkeypatch):
    _patch_pool(monkeypatch, {
        "deepseek": _Pool([_Entry(last_status="exhausted", last_error_code=402, failure_reason="billing",
                                  last_error_message="Insufficient Balance")]),
        "openrouter": _Pool([_Entry()]),
    })
    agent = _Agent(
        provider="custom", model="deepseek-v4-flash",
        chain=[
            {"provider": "deepseek", "model": "deepseek-v4-flash"},
            {"provider": "openrouter", "model": "anthropic/claude"},
        ],
        ctx={"reset_at": time.time() + 6 * 3600},
    )

    rows = wall.incident_rows(agent, reason=None)

    assert [row.provider for row in rows] == ["custom", "deepseek"]  # the healthy one is not listed
    assert rows[0].status == wall.WALLED
    assert rows[1].detail == "Insufficient Balance"


# ── message ──────────────────────────────────────────────────────────────


def test_message_lists_every_route_on_its_own_line():
    rows = [
        wall.RouteRow(provider="custom", model="deepseek-v4-flash", status=wall.WALLED,
                      detail="weekly usage quota exceeded", reset_at=time.time() + 6 * 3600),
        wall.RouteRow(provider="deepseek", model="deepseek-v4-flash", status=wall.COOLING,
                      detail="Insufficient Balance", reset_at=time.time() + 1800),
    ]

    message = wall.build_message(rows, profile="default")
    lines = message.splitlines()

    assert lines[0] != "provider_wall.title_blocked"  # the locale actually resolved
    route_lines = [line for line in lines if line.startswith(("🔴", "🟠"))]
    assert len(route_lines) == 2
    assert "custom · deepseek-v4-flash" in route_lines[0]
    assert "weekly usage quota exceeded" in route_lines[0]
    assert "resets" in route_lines[0]
    assert "deepseek · deepseek-v4-flash" in route_lines[1]
    assert "available again in ~" in route_lines[1]
    assert "hermes model" in message


def test_message_uses_the_profile_scoped_remedy_command():
    rows = [wall.RouteRow(provider="custom", model="m", status=wall.WALLED, detail="x")]

    assert "hermes -p comms model" in wall.build_message(rows, profile="comms")


# ── recording / dedupe ───────────────────────────────────────────────────


def test_record_is_once_per_incident(isolated_marker, monkeypatch):
    _patch_pool(monkeypatch, {"deepseek": _Pool([_Entry(
        last_status="exhausted", last_error_code=402, failure_reason="billing",
        last_error_message="Insufficient Balance",
    )])})
    agent = _Agent(chain=[{"provider": "deepseek", "model": "deepseek-v4-flash"}])

    first = wall.record_provider_wall(agent)
    second = wall.record_provider_wall(agent)

    assert first and second is None  # the same wall does not page twice
    payload = json.loads(isolated_marker.read_text())
    assert payload["signature"] == first
    assert len(payload["routes"]) == 2


def test_a_second_route_walling_is_a_new_incident(isolated_marker, monkeypatch):
    _patch_pool(monkeypatch, {"deepseek": _Pool([_Entry()])})
    agent = _Agent(chain=[{"provider": "deepseek", "model": "deepseek-v4-flash"}])
    first = wall.record_provider_wall(agent)

    _patch_pool(monkeypatch, {"deepseek": _Pool([_Entry(
        last_status="exhausted", last_error_code=402, failure_reason="billing",
        last_error_message="Insufficient Balance",
    )])})
    second = wall.record_provider_wall(agent)

    assert second and second != first


def test_a_different_wall_inside_the_quiet_window_does_not_page_twice(isolated_marker, monkeypatch):
    _patch_pool(monkeypatch, {"deepseek": _Pool([_Entry(
        last_status="exhausted", last_error_code=402, failure_reason="billing",
        last_error_message="Insufficient Balance",
    )])})
    agent = _Agent(chain=[{"provider": "deepseek", "model": "deepseek-v4-flash"}])
    first = wall.record_provider_wall(agent)
    wall.mark_delivered([("telegram", "HOME", "")])

    monkeypatch.setattr(wall, "_pool_key", lambda provider, base_url: provider)
    agent.provider = "another"
    second = wall.record_provider_wall(agent)

    assert second is None
    payload = json.loads(isolated_marker.read_text())
    assert payload["signature"] != first  # the merged incident's text is the current one
    assert payload["delivered_targets"] == [["telegram", "HOME", ""]]  # ledger survives the merge


def test_record_never_raises_when_the_pool_layer_is_broken(isolated_marker, monkeypatch):
    def _boom(provider):
        raise RuntimeError("no pool store")

    monkeypatch.setattr("agent.credential_pool.load_pool", _boom)
    monkeypatch.setattr(wall, "_pool_key", lambda provider, base_url: provider)

    assert wall.record_provider_wall(_Agent())  # primary row is still worth telling the operator


def test_expired_marker_is_dropped(isolated_marker):
    wall._write({"v": 1, "signature": "s", "text": "⚠️ stale", "created_at": time.time() - 10,
                 "expires_at": time.time() - 1})

    assert wall.read_pending() is None
    assert not isolated_marker.exists()


# ── which profiles are affected ──────────────────────────────────────────


def test_route_pairs_reads_the_primary_and_the_chain():
    pairs = wall.route_pairs({
        "model": {"default": "deepseek-v4-flash", "provider": "Custom"},
        "fallback_providers": [
            {"provider": "deepseek", "model": "deepseek-v4-flash"},
            {"model": "no-provider-inherits"},
            "garbage",
        ],
    })

    assert pairs == {
        ("custom", "deepseek-v4-flash"),
        ("deepseek", "deepseek-v4-flash"),
        ("", "no-provider-inherits"),
    }


def test_route_pairs_of_an_empty_config_is_empty():
    assert wall.route_pairs({}) == set()
    assert wall.route_pairs(None) == set()


def test_routes_affected_matches_the_provider_and_model_pair():
    incident = [{"provider": "custom", "model": "deepseek-v4-flash"}]

    assert wall.routes_affected({("custom", "deepseek-v4-flash")}, incident) is True
    assert wall.routes_affected({("deepseek", "deepseek-v4-flash")}, incident) is False
    assert wall.routes_affected({("custom", "some-other-model")}, incident) is False


def test_routes_affected_treats_an_unset_provider_as_inherited():
    incident = [{"provider": "custom", "model": "deepseek-v4-flash"}]

    assert wall.routes_affected({("", "deepseek-v4-flash")}, incident) is True


def test_routes_affected_fails_open_when_the_route_info_is_missing():
    # An older marker with no route list, or a profile whose config could not be read.
    assert wall.routes_affected({("xai", "grok-4")}, None) is True
    assert wall.routes_affected(set(), [{"provider": "custom", "model": "m"}]) is True

