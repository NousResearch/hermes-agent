"""Multi-profile client isolation tests.

Pin the cross-tenant bleed class (#69123 multiplexed gateway, #74065
dashboard): a process-wide first-config-wins client singleton baked the
first profile's workspace_id and bearer into one shared client, so every
later profile's memory landed in the first profile's workspace.

The tests drive the REAL resolution chain — HonchoClientConfig.from_global_config
against real honcho.json files under temp HERMES_HOMEs, with the same
ContextVar override the gateway multiplexer / dashboard use — and assert
client identity, not internals.

The two-profile repro mirrors issue #69123's minimal in-process repro;
per-config-identity caching was first proposed in #69142 (NaMinhyeok) and
extended in #81401 (angel12).
"""

import json
import threading

import pytest

import plugins.memory.honcho.client as client_mod
import plugins.memory.honcho.client_cache as client_cache_mod
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.memory.honcho.client import (
    HonchoClientConfig,
    get_honcho_client,
    reset_honcho_client,
)

pytestmark = pytest.mark.skipif(
    not pytest.importorskip("honcho", reason="honcho SDK not installed"),
    reason="honcho SDK not installed",
)


@pytest.fixture(autouse=True)
def _clean_client_cache():
    reset_honcho_client()
    yield
    reset_honcho_client()


def _make_profile(tmp_path, name: str, workspace: str, api_key: str,
                  host: str | None = None, oauth: dict | None = None):
    home = tmp_path / name
    home.mkdir(parents=True, exist_ok=True)
    host = host or "hermes"
    block: dict = {"apiKey": api_key, "workspace": workspace}
    if oauth:
        block["oauth"] = oauth
    (home / "honcho.json").write_text(json.dumps({"hosts": {host: block}}))
    return home


class _FakeHoncho:
    """Stands in for honcho.Honcho; records constructor kwargs."""

    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeHoncho.instances.append(self)


@pytest.fixture
def fake_honcho(monkeypatch):
    _FakeHoncho.instances = []
    import honcho

    monkeypatch.setattr(honcho, "Honcho", _FakeHoncho)
    return _FakeHoncho


class TestTwoProfileIsolation:
    def test_profiles_get_distinct_clients_and_workspaces(self, tmp_path, fake_honcho):
        """#69123's minimal repro: override -> client -> reset -> override -> client."""
        home_a = _make_profile(tmp_path, "profiles/a", "tenant-a", "key-a")
        home_b = _make_profile(tmp_path, "profiles/b", "tenant-b", "key-b")

        token = set_hermes_home_override(home_a)
        try:
            cfg_a = HonchoClientConfig.from_global_config()
            client_a = get_honcho_client(cfg_a)
        finally:
            reset_hermes_home_override(token)

        token = set_hermes_home_override(home_b)
        try:
            cfg_b = HonchoClientConfig.from_global_config()
            client_b = get_honcho_client(cfg_b)
        finally:
            reset_hermes_home_override(token)

        assert client_a is not client_b
        assert client_a.kwargs["workspace_id"] == "tenant-a"
        assert client_b.kwargs["workspace_id"] == "tenant-b"
        assert client_a.kwargs["api_key"] == "key-a"
        assert client_b.kwargs["api_key"] == "key-b"

    def test_same_profile_reuses_client(self, tmp_path, fake_honcho):
        home_a = _make_profile(tmp_path, "profiles/a", "tenant-a", "key-a")

        token = set_hermes_home_override(home_a)
        try:
            cfg1 = HonchoClientConfig.from_global_config()
            c1 = get_honcho_client(cfg1)
            cfg2 = HonchoClientConfig.from_global_config()
            c2 = get_honcho_client(cfg2)
        finally:
            reset_hermes_home_override(token)

        assert c1 is c2
        assert len(fake_honcho.instances) == 1


class TestBackgroundThreadIsolation:
    def test_bound_config_wins_on_bare_thread(self, tmp_path, fake_honcho):
        """A manager's bound config must acquire ITS profile's client even
        from a thread that cannot see the profile ContextVar — the pattern
        of every plugin daemon thread (async writer, prefetch, sync)."""
        home_a = _make_profile(tmp_path, "profiles/a", "tenant-a", "key-a")
        home_b = _make_profile(tmp_path, "profiles/b", "tenant-b", "key-b")

        # Default-profile client exists first (the "pinning" client).
        token = set_hermes_home_override(home_a)
        try:
            cfg_a = HonchoClientConfig.from_global_config()
            get_honcho_client(cfg_a)
        finally:
            reset_hermes_home_override(token)

        # Profile B's config resolved inside its scope (as initialize() does).
        token = set_hermes_home_override(home_b)
        try:
            cfg_b = HonchoClientConfig.from_global_config()
        finally:
            reset_hermes_home_override(token)

        # A bare thread (empty context — no profile override visible)
        # acquires via the bound config, as manager.honcho now does.
        box: dict = {}

        def _worker():
            box["client"] = get_honcho_client(cfg_b)

        t = threading.Thread(target=_worker)
        t.start()
        t.join(timeout=10)

        assert box["client"].kwargs["workspace_id"] == "tenant-b"
        assert box["client"].kwargs["api_key"] == "key-b"

    def test_spawn_context_thread_sees_profile_override(self, tmp_path):
        """spawn_context_thread must carry the caller's HERMES_HOME override."""
        from hermes_constants import get_hermes_home
        from plugins.memory.honcho.client import spawn_context_thread

        home_b = tmp_path / "profiles" / "b"
        home_b.mkdir(parents=True)
        seen: dict = {}

        def _probe():
            seen["home"] = get_hermes_home()

        token = set_hermes_home_override(home_b)
        try:
            t = spawn_context_thread(_probe, name="probe")
            t.start()
            t.join(timeout=10)
        finally:
            reset_hermes_home_override(token)

        assert seen["home"] == home_b

    def test_plain_thread_does_not_see_override(self, tmp_path):
        """Control: documents WHY propagation is needed — a plain thread
        resolves the process home, not the caller's profile override."""
        from hermes_constants import get_hermes_home

        home_b = tmp_path / "profiles" / "b"
        home_b.mkdir(parents=True)
        seen: dict = {}

        def _probe():
            seen["home"] = get_hermes_home()

        token = set_hermes_home_override(home_b)
        try:
            t = threading.Thread(target=_probe)
            t.start()
            t.join(timeout=10)
        finally:
            reset_hermes_home_override(token)

        assert seen["home"] != home_b


class TestCredentialIdentity:
    def test_account_swap_creates_new_client_and_evicts_old(self, tmp_path, fake_honcho):
        """Switching accounts via setup (same path/host, new apiKey) must not
        keep serving the old account's client — the collision a
        provenance-only cache key cannot close."""
        home = _make_profile(tmp_path, "profiles/a", "tenant-a", "key-account-1")

        token = set_hermes_home_override(home)
        try:
            cfg1 = HonchoClientConfig.from_global_config()
            c1 = get_honcho_client(cfg1)

            # Operator re-runs setup: same file, new account credentials.
            (home / "honcho.json").write_text(json.dumps({
                "hosts": {"hermes": {"apiKey": "key-account-2", "workspace": "tenant-a"}},
            }))
            cfg2 = HonchoClientConfig.from_global_config()
            c2 = get_honcho_client(cfg2)
        finally:
            reset_hermes_home_override(token)

        assert c1 is not c2
        assert c2.kwargs["api_key"] == "key-account-2"
        # Old slot evicted: the stale client is no longer reachable via the map.
        with client_mod._client_slots_lock:
            cached_clients = [
                s.peek() for s in client_mod._client_slots.values()
            ]
        assert c1 not in cached_clients

    def test_oauth_refresh_token_is_fingerprint_basis(self, tmp_path):
        """Fingerprint must survive access-token rotation (in-place bearer
        swap) but change when the refresh token (re-auth) changes."""
        home = tmp_path / "p"
        home.mkdir()
        oauth_block = {
            "refreshToken": "refresh-1",
            "tokenEndpoint": "https://auth.example/token",
            "clientId": "cid",
            "expiresAt": 9999999999,
        }
        (home / "honcho.json").write_text(json.dumps({
            "hosts": {"hermes": {"apiKey": "access-token-1", "workspace": "w",
                                  "oauth": oauth_block}},
        }))

        token = set_hermes_home_override(home)
        try:
            cfg1 = HonchoClientConfig.from_global_config()
            fp1 = client_cache_mod._credential_fingerprint(cfg1)

            # Access token rotates in place; refresh token unchanged.
            cfg_rotated = HonchoClientConfig.from_global_config()
            cfg_rotated.api_key = "access-token-2"
            fp_rotated = client_cache_mod._credential_fingerprint(cfg_rotated)

            # Re-auth: new refresh token.
            oauth_block2 = dict(oauth_block, refreshToken="refresh-2")
            (home / "honcho.json").write_text(json.dumps({
                "hosts": {"hermes": {"apiKey": "access-token-3", "workspace": "w",
                                      "oauth": oauth_block2}},
            }))
            cfg2 = HonchoClientConfig.from_global_config()
            fp2 = client_cache_mod._credential_fingerprint(cfg2)
        finally:
            reset_hermes_home_override(token)

        assert fp1 == fp_rotated, "access-token rotation must not change identity"
        assert fp1 != fp2, "re-auth must change identity"

    def test_timeout_change_rebuilds_via_key(self, tmp_path, fake_honcho):
        """The old singleton had an explicit timeout-staleness check; with
        timeout in the key, a change produces a new identity + eviction."""
        home = _make_profile(tmp_path, "profiles/a", "w", "k")
        token = set_hermes_home_override(home)
        try:
            cfg1 = HonchoClientConfig.from_global_config()
            c1 = get_honcho_client(cfg1)

            raw = json.loads((home / "honcho.json").read_text())
            raw["hosts"]["hermes"]["timeout"] = 77
            (home / "honcho.json").write_text(json.dumps(raw))
            cfg2 = HonchoClientConfig.from_global_config()
            c2 = get_honcho_client(cfg2)
        finally:
            reset_hermes_home_override(token)

        assert c1 is not c2
        assert c2.kwargs["timeout"] == 77.0


class TestProvenance:
    def test_from_global_config_captures_provenance(self, tmp_path):
        home = _make_profile(tmp_path, "profiles/a", "w", "k")
        token = set_hermes_home_override(home)
        try:
            cfg = HonchoClientConfig.from_global_config()
        finally:
            reset_hermes_home_override(token)

        assert cfg.config_path == home / "honcho.json"
        assert cfg.hermes_home == home
        assert cfg.bound_config_path() == home / "honcho.json"

    def test_bound_path_stable_outside_scope(self, tmp_path):
        """The captured path must not drift when read outside the profile
        scope (the daemon-thread situation)."""
        home = _make_profile(tmp_path, "profiles/a", "w", "k")
        token = set_hermes_home_override(home)
        try:
            cfg = HonchoClientConfig.from_global_config()
        finally:
            reset_hermes_home_override(token)

        # Now OUTSIDE the scope — bound path still points at profile a.
        assert cfg.bound_config_path() == home / "honcho.json"
