"""Real-process E2E: OAuth providers against a loopback vendor.

Every cell drives the real ``python -m rabbit_cli.main`` in a hermetic HOME (fake HOME, RABBIT_HOME
under it, no real credentials) against ``tests.fakes.providers.catalog_oauth.OAuthFake`` — the
vendor's OAuth authorization server and a bearer-checking inference server on 127.0.0.1. All other
egress goes through the ``CatalogFake`` sentinel proxy, which refuses and records any non-loopback
host. Cells assert user-visible outcomes: the reply on stdout, the bearer on the next wire request,
and the tokens persisted to auth.json.

NOT COVERED (not redirectable to a loopback fake): openai-codex and qwen-oauth refresh (token URLs
are module constants, no env/config override) and the Copilot token exchange (hardcoded
api.github.com).
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

from tests.fakes.providers.catalog_fake import CatalogFake
from tests.fakes.providers.catalog_oauth import OAuthFake

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX harness")

REPO_ROOT = Path(__file__).resolve().parents[4]
TURN_TIMEOUT = 120.0
# Public, credential-free model-metadata catalog (pricing/context lookups); never carries a vendor token.
CREDENTIAL_FREE_HOSTS = frozenset({"models.dev:443"})

_PASSTHROUGH_ENV = frozenset({"PATH", "LANG", "LANGUAGE", "USER", "LOGNAME", "SHELL", "TZ"})
_SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_ACCESS_KEY")


# --- hermetic home ---------------------------------------------------------------------------


class Home:
    def __init__(self, root: Path, sentinel: CatalogFake, provider: str, model: str, base_url: str = "") -> None:
        self.home = root / "home"
        self.rabbit_home = self.home / ".rabbit"
        self.rabbit_home.mkdir(parents=True)
        self.sentinel = sentinel
        (self.rabbit_home / "config.yaml").write_text(
            f"model:\n  provider: {provider}\n  default: {model}\n"
            # The startup cost guard probes ``model.base_url``/models for pricing (credential-free),
            # else the provider's production host; point it at the fake so any production-host
            # egress the sentinel records is the credential-bearing turn itself.
            + (f"  base_url: {base_url}\n" if base_url else "") +
            "updates:\n  check: false\n"
            "agent:\n  api_max_retries: 1\n  auto_recovery_cycles: 0\n")

    @property
    def auth_path(self) -> Path:
        return self.rabbit_home / "auth.json"

    def seed_auth(self, store: dict[str, Any]) -> None:
        self.auth_path.write_text(json.dumps(store, indent=2), encoding="utf-8")

    def auth(self) -> dict[str, Any]:
        return json.loads(self.auth_path.read_text(encoding="utf-8"))

    def env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items()
               if (k in _PASSTHROUGH_ENV or k.startswith("LC_")) and not k.endswith(_SECRET_SUFFIXES)}
        env.update({
            "HOME": str(self.home), "RABBIT_HOME": str(self.rabbit_home), "PYTHONPATH": str(REPO_ROOT),
            "PYTHONUNBUFFERED": "1", "NO_COLOR": "1", "TERM": "dumb",
            "TMPDIR": str(self.home), "RABBIT_SHARED_AUTH_DIR": str(self.home / "shared"),
            "CODEX_HOME": str(self.home / ".codex"),
            # Child HOME is the fixture home, so its state.db is tmp_path's (guard's documented escape).
            "RABBIT_STATE_DB_GUARD_BYPASS": "1",
            **self.sentinel.proxy_env()})
        env.update(extra or {})
        return env

    def run(self, argv: list[str], extra_env: dict[str, str] | None = None,
            timeout: float = TURN_TIMEOUT) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "rabbit_cli.main", *argv], cwd=str(self.home), env=self.env(extra_env),
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)


def _iso(delta_s: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=delta_s)).isoformat()


OPENROUTER_ROW = {"id": "or-1", "label": "openrouter-key", "auth_type": "api_key", "priority": 0,
                  "source": "manual", "access_token": "sk-or-oauth-e2e-untouched",
                  "base_url": "https://openrouter.ai/api/v1"}


def _row(store: dict[str, Any], provider: str, row_id: str) -> dict[str, Any]:
    rows = [r for r in store.get("credential_pool", {}).get(provider, []) if r.get("id") == row_id]
    assert rows, f"credential_pool.{provider} lost row {row_id}: {store.get('credential_pool', {}).get(provider)}"
    return rows[0]


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
