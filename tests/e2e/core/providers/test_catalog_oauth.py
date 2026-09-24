"""Real-process E2E: OAuth / device-code providers against a loopback vendor.

Every cell drives the real ``python -m hermes_cli.main`` in a hermetic HOME (fake HOME, HERMES_HOME
under it, no real credentials) against ``tests.fakes.providers.catalog_oauth.OAuthFake`` — the
vendor's OAuth authorization server and a bearer-checking inference server on 127.0.0.1. All other
egress goes through the ``CatalogFake`` sentinel proxy, which refuses and records any non-loopback
host. Cells assert user-visible outcomes: the device-code polling cadence the vendor sees, the
reply on stdout, the bearer on the next wire request, and the tokens persisted to auth.json.

Open bugs are strict, message-gated run-time xfails (``KNOWN`` + ``strict_known``): a cell XFAILs
only while it fails with that bug's signature, and FAILS once the fix lands so the entry is dropped.

Not redirectable to a loopback fake, so not covered here (explicit skips below): openai-codex and
qwen-oauth refresh (token URLs are module constants, no env/config override) and the Copilot token
exchange (hardcoded api.github.com).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.core.providers._catalog_helpers import strict_known
from tests.fakes.providers.catalog_fake import CatalogFake
from tests.fakes.providers.catalog_oauth import NOUS_INVOKE_SCOPE, OAuthFake, make_jwt

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX harness")

REPO_ROOT = Path(__file__).resolve().parents[4]
PROD_NOUS_INFERENCE = "https://inference-api.nousresearch.com/v1"
TURN_TIMEOUT = 120.0
# Public, credential-free model-metadata catalog (pricing/context lookups); never carries a vendor token.
CREDENTIAL_FREE_HOSTS = frozenset({"models.dev:443"})

# (pattern matched against the failing assertion text, reason). Delete an entry when its fix lands.
KNOWN: dict[str, tuple[str, str]] = {
    "device_interval": (
        r"device poll gap .* < server interval",
        "#121163 Nous device-code login polls at 1s, ignoring the server's interval"),
    "nous_401_retry_route": (
        r"401 recovery retry left NOUS_INFERENCE_BASE_URL",
        "#121323 Nous 401 pool recovery retries on the stored production host, "
        "dropping the NOUS_INFERENCE_BASE_URL override"),
}

_PASSTHROUGH_ENV = frozenset({"PATH", "LANG", "LANGUAGE", "USER", "LOGNAME", "SHELL", "TZ"})
_SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_ACCESS_KEY")


# --- hermetic home ---------------------------------------------------------------------------


class Home:
    def __init__(self, root: Path, sentinel: CatalogFake, provider: str, model: str, base_url: str = "") -> None:
        self.home = root / "home"
        self.hermes_home = self.home / ".hermes"
        self.hermes_home.mkdir(parents=True)
        self.sentinel = sentinel
        (self.hermes_home / "config.yaml").write_text(
            f"model:\n  provider: {provider}\n  default: {model}\n"
            # The startup cost guard probes ``model.base_url``/models for pricing (credential-free),
            # else the provider's production host; point it at the fake so any production-host
            # egress the sentinel records is the credential-bearing turn itself.
            + (f"  base_url: {base_url}\n" if base_url else "") +
            "updates:\n  check: false\n"
            "agent:\n  api_max_retries: 1\n  auto_recovery_cycles: 0\n")

    @property
    def auth_path(self) -> Path:
        return self.hermes_home / "auth.json"

    def seed_auth(self, store: dict[str, Any]) -> None:
        self.auth_path.write_text(json.dumps(store, indent=2))

    def auth(self) -> dict[str, Any]:
        return json.loads(self.auth_path.read_text())

    def env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items()
               if (k in _PASSTHROUGH_ENV or k.startswith("LC_")) and not k.endswith(_SECRET_SUFFIXES)}
        env.update({
            "HOME": str(self.home), "HERMES_HOME": str(self.hermes_home), "PYTHONPATH": str(REPO_ROOT),
            "PYTHONUNBUFFERED": "1", "NO_COLOR": "1", "TERM": "dumb",
            "TMPDIR": str(self.home), "HERMES_SHARED_AUTH_DIR": str(self.home / "shared"),
            "CODEX_HOME": str(self.home / ".codex"),
            # Child HOME is the fixture home, so its state.db is tmp_path's (guard's documented escape).
            "HERMES_STATE_DB_GUARD_BYPASS": "1",
            **self.sentinel.proxy_env()})
        env.update(extra or {})
        return env

    def run(self, argv: list[str], extra_env: dict[str, str] | None = None,
            timeout: float = TURN_TIMEOUT) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", *argv], cwd=str(self.home), env=self.env(extra_env),
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)


def _iso(delta_s: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=delta_s)).isoformat()


def _nous_state(fake: OAuthFake, access: str, refresh: str, ttl_s: int) -> dict[str, Any]:
    return {"access_token": access, "refresh_token": refresh, "token_type": "Bearer",
            "scope": NOUS_INVOKE_SCOPE, "client_id": "hermes-cli", "portal_base_url": fake.origin,
            "inference_base_url": PROD_NOUS_INFERENCE, "obtained_at": _iso(-60), "expires_at": _iso(ttl_s),
            "agent_key": access, "agent_key_expires_at": _iso(ttl_s),
            "tls": {"insecure": False, "ca_bundle": None}}


OTHER_NOUS_ROW = "nous-manual-2"
OPENROUTER_ROW = {"id": "or-1", "label": "openrouter-key", "auth_type": "api_key", "priority": 0,
                  "source": "manual", "access_token": "sk-or-oauth-e2e-untouched",
                  "base_url": "https://openrouter.ai/api/v1"}


def _nous_store(fake: OAuthFake, access: str, refresh: str, ttl_s: int) -> dict[str, Any]:
    state = _nous_state(fake, access, refresh, ttl_s)
    other_access = make_jwt("other-row", ttl_s=7200)
    other = {**_nous_state(fake, other_access, "rt-other-row-untouched", 7200), "id": OTHER_NOUS_ROW,
             "label": "second-login", "auth_type": "oauth", "priority": 1, "source": "manual:device_code"}
    return {"version": 1, "active_provider": "nous", "providers": {"nous": state},
            "credential_pool": {
                "nous": [{**state, "id": "nous-dc-1", "label": "seed", "auth_type": "oauth", "priority": 0,
                          "source": "device_code"}, other],
                "openrouter": [OPENROUTER_ROW]}}


def _row(store: dict[str, Any], provider: str, row_id: str) -> dict[str, Any]:
    rows = [r for r in store.get("credential_pool", {}).get(provider, []) if r.get("id") == row_id]
    assert rows, f"credential_pool.{provider} lost row {row_id}: {store.get('credential_pool', {}).get(provider)}"
    return rows[0]


def _assert_untouched(before: dict[str, Any], after: dict[str, Any]) -> None:
    for provider, row_id in (("nous", OTHER_NOUS_ROW), ("openrouter", "or-1")):
        want, got = _row(before, provider, row_id), _row(after, provider, row_id)
        for key in ("access_token", "refresh_token"):
            assert got.get(key) == want.get(key), (
                f"persisting the rotation clobbered credential_pool.{provider}[{row_id}].{key}: "
                f"{want.get(key)!r} -> {got.get(key)!r}")


def _describe(proc: subprocess.CompletedProcess[str], fake: OAuthFake, sentinel: CatalogFake) -> str:
    wire = [(r.method, r.path, r.bearer[-12:]) for r in fake.requests]
    return (f"rc={proc.returncode}\nstdout={proc.stdout[-1500:]}\nstderr={proc.stderr[-2500:]}\n"
            f"wire={wire}\negress={sentinel.egress_hosts()}")


def _vendor_egress(sentinel: CatalogFake) -> list[str]:
    """Non-loopback hosts the child tried to reach, minus the credential-free metadata catalog."""
    return [h for h in sentinel.egress_hosts() if h not in CREDENTIAL_FREE_HOSTS]


@pytest.fixture
def sentinel():
    with CatalogFake() as s:
        yield s


# --- Nous device-code login ------------------------------------------------------------------

# Two pendings around one slow_down: the gaps before slow_down show the base interval the client
# honours, the gaps after it show the +1s RFC 8628 growth.
DEVICE_SCRIPT = ["authorization_pending", "slow_down", "authorization_pending"]
DEVICE_INTERVAL = 2


@pytest.fixture(scope="module")
def device_login(tmp_path_factory):
    """One real ``hermes auth add nous --type oauth`` device-code login against the fake Portal."""
    root = tmp_path_factory.mktemp("nous-device")
    with CatalogFake() as sentinel, OAuthFake(device_interval=DEVICE_INTERVAL, poll_script=DEVICE_SCRIPT) as fake:
        home = Home(root, sentinel, "nous", "oauth-e2e/model")
        seed = {"version": 1, "credential_pool": {"openrouter": [OPENROUTER_ROW]}}
        home.seed_auth(seed)
        proc = home.run(["auth", "add", "nous", "--type", "oauth", "--no-browser", "--portal-url", fake.origin],
                        extra_env={"NOUS_INFERENCE_BASE_URL": f"{fake.origin}/v1"}, timeout=90)
        yield {"proc": proc, "fake": fake, "home": home, "seed": seed, "sentinel": sentinel,
               "gaps": [b.t - a.t for a, b in zip(fake.device_polls(), fake.device_polls()[1:])]}


def test_nous_device_login_persists_tokens(device_login) -> None:
    proc, fake, home = device_login["proc"], device_login["fake"], device_login["home"]
    info = _describe(proc, fake, device_login["sentinel"])
    assert proc.returncode == 0, f"device-code login failed\n{info}"
    assert len(fake.device_polls()) == len(DEVICE_SCRIPT) + 1, f"login stopped polling early\n{info}"
    assert "OAUT-HE2E" in proc.stdout, f"user code never shown to the user\n{info}"
    issued = fake.issued[0]
    store = home.auth()
    state = store.get("providers", {}).get("nous") or {}
    assert state.get("refresh_token") == issued["refresh_token"], (
        f"providers.nous did not persist the login's refresh token: {state.get('refresh_token')!r}\n{info}")
    assert state.get("access_token") == issued["access_token"], f"providers.nous access token not persisted\n{info}"
    pooled = [r.get("refresh_token") for r in store.get("credential_pool", {}).get("nous", [])]
    assert issued["refresh_token"] in pooled, f"credential_pool.nous missing the login: {pooled}\n{info}"
    assert _row(store, "openrouter", "or-1")["access_token"] == OPENROUTER_ROW["access_token"], (
        "login clobbered an unrelated credential_pool row")
    assert not _vendor_egress(device_login["sentinel"]), f"login leaked egress: {_vendor_egress(device_login['sentinel'])}"


def test_nous_device_login_slow_down_grows_interval(device_login) -> None:
    gaps = device_login["gaps"]
    assert len(gaps) == len(DEVICE_SCRIPT), f"unexpected poll count, gaps={gaps}"
    before, after = gaps[0], gaps[1:]
    assert all(g >= before + 0.9 for g in after), (
        f"slow_down did not grow the poll interval by >=1s: gap before slow_down {before:.2f}s, "
        f"after {[round(g, 2) for g in after]}")


def test_nous_device_login_honors_server_interval(device_login) -> None:
    gaps = device_login["gaps"]
    assert len(gaps) == len(DEVICE_SCRIPT), f"unexpected poll count, gaps={gaps}"
    with strict_known(*KNOWN["device_interval"]):
        assert gaps[0] >= DEVICE_INTERVAL - 0.1, (
            f"device poll gap {gaps[0]:.2f}s < server interval {DEVICE_INTERVAL}s (gaps={[round(g, 2) for g in gaps]})")


# --- Nous token refresh ----------------------------------------------------------------------


def _nous_turn(tmp_path: Path, sentinel: CatalogFake, fake: OAuthFake, access: str, ttl_s: int):
    home = Home(tmp_path, sentinel, "nous", "oauth-e2e/model", base_url=f"{fake.origin}/v1")
    seed = _nous_store(fake, access, "rt-seed-0", ttl_s)
    home.seed_auth(seed)
    proc = home.run(["-z", "Say hi", "--provider", "nous", "-m", "oauth-e2e/model"],
                    extra_env={"NOUS_INFERENCE_BASE_URL": f"{fake.origin}/v1"})
    return home, seed, proc


def _assert_rotation_persisted(home: Home, seed: dict[str, Any], fake: OAuthFake, info: str) -> str:
    refreshes = fake.refreshes()
    assert refreshes, f"no refresh-token exchange reached the Portal\n{info}"
    assert refreshes[0].headers.get("x-nous-refresh-token") == "rt-seed-0", (
        f"refresh redeemed the wrong token: {refreshes[0].headers.get('x-nous-refresh-token')!r}\n{info}")
    rotated = fake.issued[-1]
    store = home.auth()
    assert store["providers"]["nous"].get("refresh_token") == rotated["refresh_token"], (
        f"providers.nous kept a spent refresh token {store['providers']['nous'].get('refresh_token')!r} "
        f"instead of the rotation {rotated['refresh_token']!r}\n{info}")
    assert _row(store, "nous", "nous-dc-1").get("refresh_token") == rotated["refresh_token"], (
        f"credential_pool.nous[nous-dc-1] kept a spent refresh token "
        f"{_row(store, 'nous', 'nous-dc-1').get('refresh_token')!r}\n{info}")
    _assert_untouched(seed, store)
    return rotated["access_token"]


def test_nous_expired_access_token_refreshes_before_turn(tmp_path, sentinel) -> None:
    stale = make_jwt("expired", ttl_s=-60)
    with OAuthFake(valid_refresh={"rt-seed-0"}, revoked={stale}) as fake:
        home, seed, proc = _nous_turn(tmp_path, sentinel, fake, stale, ttl_s=-60)
        info = _describe(proc, fake, sentinel)
        assert proc.returncode == 0 and fake.reply in proc.stdout, f"turn with an expired token failed\n{info}"
        fresh = _assert_rotation_persisted(home, seed, fake, info)
        bearers = {r.bearer for r in fake.inference()}
        assert bearers == {fresh}, f"inference used {bearers}, expected only the refreshed token\n{info}"
        assert len(fake.refreshes()) == 1, f"refresh token redeemed {len(fake.refreshes())}x\n{info}"
        assert not _vendor_egress(sentinel), f"turn leaked egress: {_vendor_egress(sentinel)}"


def test_nous_inference_401_refreshes_rotates_and_retries(tmp_path, sentinel) -> None:
    revoked = make_jwt("revoked-by-server", ttl_s=7200)
    with OAuthFake(valid_refresh={"rt-seed-0"}, revoked={revoked}) as fake:
        home, seed, proc = _nous_turn(tmp_path, sentinel, fake, revoked, ttl_s=7200)
        info = _describe(proc, fake, sentinel)
        assert any(r.bearer == revoked for r in fake.inference()), f"the stale bearer was never tried\n{info}"
        fresh = _assert_rotation_persisted(home, seed, fake, info)
        with strict_known(*KNOWN["nous_401_retry_route"]):
            assert not _vendor_egress(sentinel), (
                f"401 recovery retry left NOUS_INFERENCE_BASE_URL: egress to {_vendor_egress(sentinel)}\n{info}")
            assert any(r.bearer == fresh for r in fake.inference()), f"no retry with the refreshed token\n{info}"
            assert proc.returncode == 0 and fake.reply in proc.stdout, f"turn failed after refresh\n{info}"


# --- MiniMax OAuth refresh -------------------------------------------------------------------


def test_minimax_oauth_expired_token_refreshes_and_persists(tmp_path, sentinel) -> None:
    with OAuthFake(valid_refresh={"rt-mm-seed"}, revoked={"mm-stale-access"}) as fake:
        home = Home(tmp_path, sentinel, "minimax-oauth", "MiniMax-M2")
        seed = {"version": 1, "active_provider": "minimax-oauth", "providers": {"minimax-oauth": {
            "portal_base_url": fake.origin, "inference_base_url": f"{fake.origin}/anthropic",
            "client_id": "oauth-e2e-client", "access_token": "mm-stale-access", "refresh_token": "rt-mm-seed",
            "expires_at": _iso(-60), "region": "global", "token_type": "Bearer", "scope": "group_id profile"}},
            "credential_pool": {"openrouter": [OPENROUTER_ROW]}}
        home.seed_auth(seed)
        proc = home.run(["-z", "Say hi", "--provider", "minimax-oauth", "-m", "MiniMax-M2"])
        info = _describe(proc, fake, sentinel)
        assert proc.returncode == 0 and fake.reply in proc.stdout, f"MiniMax turn failed\n{info}"
        assert fake.refreshes() and fake.refreshes()[0].form.get("refresh_token") == "rt-mm-seed", (
            f"MiniMax refresh did not redeem the stored refresh token\n{info}")
        rotated = fake.issued[-1]
        bearers = {r.bearer for r in fake.inference()}
        assert bearers == {rotated["access_token"]}, f"inference used {bearers}, not the refreshed token\n{info}"
        state = home.auth()["providers"]["minimax-oauth"]
        assert state.get("refresh_token") == rotated["refresh_token"], (
            f"providers.minimax-oauth kept spent refresh token {state.get('refresh_token')!r}\n{info}")
        assert state.get("access_token") == rotated["access_token"], f"MiniMax access token not persisted\n{info}"
        assert _row(home.auth(), "openrouter", "or-1")["access_token"] == OPENROUTER_ROW["access_token"]
        assert not _vendor_egress(sentinel), f"turn leaked egress: {_vendor_egress(sentinel)}"


# --- not redirectable ------------------------------------------------------------------------

_UNREDIRECTABLE = {
    "openai-codex": "token refresh URL is the module constant CODEX_OAUTH_TOKEN_URL (auth.openai.com); "
                    "no env/config override, so a refresh cannot reach a loopback fake",
    "qwen-oauth": "token refresh URL is the module constant QWEN_OAUTH_TOKEN_URL (chat.qwen.ai); "
                  "no env/config override",
    "copilot": "token exchange URL is hardcoded to api.github.com",
}


@pytest.mark.parametrize("provider", sorted(_UNREDIRECTABLE))
def test_oauth_refresh_not_redirectable(provider: str) -> None:
    pytest.skip(f"{provider}: {_UNREDIRECTABLE[provider]}")

