"""Hermetic subprocess harness for the OAuth E2E tests (fake HOME, tagged tree).

Every ``hermes`` child runs with an allowlisted environment (no inherited
credentials or ``HERMES_*``), ``HOME``/``HERMES_HOME`` under ``tmp_path``, and
a unique ``OAUTH_E2E_TAG`` so cleanup signals only this test's processes.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from tests.fakes.providers.anthropic_messages import ApiError, AnthropicMessagesServer, Reply, Response, Text, ToolUse

REPO_ROOT = Path(__file__).resolve().parents[4]
TAG_VAR = "OAUTH_E2E_TAG"

_SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_ACCESS_KEY")
_PASSTHROUGH_ENV = frozenset({"PATH", "LANG", "LANGUAGE", "USER", "LOGNAME", "SHELL", "TMPDIR", "TZ"})


@dataclass
class FakeHome:
    root: Path
    home: Path
    hermes_home: Path
    tag: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def auth_path(self) -> Path:
        return self.hermes_home / "auth.json"

    def env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        import pwd  # POSIX-only; the suite is Linux-gated

        real_root = Path(pwd.getpwuid(os.getuid()).pw_dir, ".hermes").resolve()
        fixture = self.hermes_home.resolve()
        assert fixture != real_root and fixture.parent != real_root / "profiles", (
            f"fixture HERMES_HOME {fixture} is the real install's live home")
        env = {
            k: v for k, v in os.environ.items()
            if (k in _PASSTHROUGH_ENV or k.startswith("LC_")) and not k.endswith(_SECRET_ENV_SUFFIXES)
        }
        env.update({
            "HOME": str(self.home), "HERMES_HOME": str(self.hermes_home),
            "PYTHONPATH": str(REPO_ROOT), "PYTHONUNBUFFERED": "1", "NO_COLOR": "1", "TERM": "dumb",
            TAG_VAR: self.tag,
            # The child's ~/.hermes/state.db IS the tmp home's db; see parity/_helpers.py.
            "HERMES_STATE_DB_GUARD_BYPASS": "1",
        })
        env.update(extra or {})
        return env

    def write_config(self, cfg: dict[str, Any]) -> None:
        base = {"updates": {"check": False}, "display": {"compact": True}}
        base.update(cfg)
        (self.hermes_home / "config.yaml").write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    def write_auth(self, store: dict[str, Any]) -> None:
        self.auth_path.write_text(json.dumps(store, indent=2), encoding="utf-8")
        self.auth_path.chmod(0o600)

    def read_auth(self) -> dict[str, Any]:
        return json.loads(self.auth_path.read_text(encoding="utf-8"))


def make_home(root: Path) -> FakeHome:
    home = root / "home"
    (home / ".hermes").mkdir(parents=True, exist_ok=True)
    fh = FakeHome(root=root, home=home, hermes_home=home / ".hermes")
    fh.write_config({})
    return fh


def hermes_argv(*args: str) -> list[str]:
    return [sys.executable, "-m", "hermes_cli.main", *args]


def run_hermes(fh: FakeHome, args: list[str], *, extra_env: dict[str, str] | None = None,
               timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(hermes_argv(*args), env=fh.env(extra_env), cwd=str(fh.root), stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=timeout)


def spawn_hermes(fh: FakeHome, args: list[str], *, extra_env: dict[str, str] | None = None,
                 log: Path) -> subprocess.Popen:
    out = open(log, "w", encoding="utf-8")  # noqa: SIM115 - closed when the child is reaped
    try:
        return subprocess.Popen(hermes_argv(*args), env=fh.env(extra_env), cwd=str(fh.root),
                                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, text=True)
    finally:
        out.close()


def kill_tagged(tag: str) -> None:
    """SIGKILL every live process carrying ``OAUTH_E2E_TAG=<tag>`` (this test's tree only)."""
    needle = f"{TAG_VAR}={tag}".encode()
    me = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == me:
            continue
        try:
            with open(f"/proc/{entry}/environ", "rb") as fh:
                env = fh.read()
        except OSError:
            continue
        if needle in env.split(b"\0"):
            try:
                os.kill(int(entry), signal.SIGKILL)
            except OSError:
                pass


def wait_until(pred: Callable[[], Any], timeout: float, what: str, interval: float = 0.05) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        value = pred()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {what}")
        time.sleep(interval)


# ---- Anthropic Messages endpoint --------------------------------------------
#
# The SDK-oracle fake (tests/fakes/providers/anthropic_messages.py) validates every
# body and builds every reply from SDK models. A test's ``decide(record)`` returns a
# ``Decision``: a scripted ``Response`` (``Reply``/``ApiError``), a ``Hold`` that
# parks the reply on the handler thread until an event fires, or a zero-arg
# callable evaluated at send time. ``record["bearer"]`` / ``record["x_api_key"]``
# carry the credential each call presented.


@dataclass
class Hold:
    event: threading.Event
    then: Any  # Decision
    timeout: float = 120.0


def resolve(decision: Any) -> Response:
    """Follow ``Hold``s and callables down to the ``Response`` sent on the wire."""
    while not isinstance(decision, (Reply, ApiError)):
        if isinstance(decision, Hold):
            decision.event.wait(decision.timeout)
            decision = decision.then
        else:
            decision = decision()
    return decision


def text(s: str) -> Reply:
    return Reply([Text(s)])


def tool(name: str, args: dict[str, Any]) -> Reply:
    return Reply([ToolUse(name, args)])


def credential(record: dict[str, Any]) -> str:
    return record["bearer"] or record["x_api_key"]


# ---- Anthropic OAuth rig ----------------------------------------------------
#
# The Anthropic token URL is hardcoded (no supported override), so the child
# reaches the fake token server through TLSInterceptProxy: HTTPS_PROXY + a test
# CA in SSL_CERT_FILE; NO_PROXY keeps inference to the loopback fake direct.

ANTHROPIC_TOKEN_HOSTS = ("platform.claude.com", "console.anthropic.com")
OLD_ACCESS = "sk-ant-oat01-e2e-old-access"
EXPIRED = ApiError(401, "authentication_error", "OAuth token has expired")
MODEL = "claude-sonnet-4-5"


@dataclass
class AnthropicOAuthRig:
    fh: FakeHome
    tokens: Any  # OAuthTokenServer
    proxy: Any  # TLSInterceptProxy
    messages: AnthropicMessagesServer
    seed_refresh: str
    child_env: dict[str, str]

    def stop(self) -> None:
        kill_tagged(self.fh.tag)
        self.messages.stop()
        self.proxy.stop()
        self.tokens.stop()

    def pool_row(self, entry_id: str = "e1") -> dict[str, Any]:
        rows = self.fh.read_auth().get("credential_pool", {}).get("anthropic") or []
        return next((r for r in rows if r.get("id") == entry_id), {})


def start_anthropic_rig(root: Path, decide: Callable[[dict[str, Any]], Any], *,
                        title_generation: bool, entries: int = 1,
                        expires_at_ms: int | None = None) -> AnthropicOAuthRig:
    """``expires_at_ms`` is the seeded row's clock expiry; default: an hour ahead, so only the
    vendor's 401 (early revocation/expiry) can trigger the refresh."""
    from tests.fakes.providers.oauth_token_server import OAuthTokenServer, TLSInterceptProxy, make_test_ca

    ca = make_test_ca(root / "ca", ANTHROPIC_TOKEN_HOSTS)
    tokens = OAuthTokenServer().start()
    proxy = TLSInterceptProxy(tokens, ca, ANTHROPIC_TOKEN_HOSTS).start()
    def responder(record: dict[str, Any]) -> Response:
        return resolve(decide(record))

    messages = AnthropicMessagesServer(responder, aux=responder, models=[MODEL]).start()
    fh = make_home(root)
    fh.write_config({
        "model": {"provider": "anthropic", "default": MODEL, "base_url": messages.base_url},
        "auxiliary": {"title_generation": {"enabled": title_generation}},
    })
    seed = tokens.seed_refresh_token()
    rows = [{"id": "e1", "label": "acct-1", "auth_type": "oauth", "priority": 0, "source": "manual",
             "access_token": OLD_ACCESS, "refresh_token": seed,
             "expires_at_ms": expires_at_ms if expires_at_ms is not None else int(time.time() * 1000) + 3_600_000}]
    for i in range(2, entries + 1):
        rows.append({"id": f"e{i}", "label": f"acct-{i}", "auth_type": "oauth", "priority": i - 1,
                     "source": "manual", "access_token": f"sk-ant-oat01-e2e-spare-{i}",
                     "refresh_token": tokens.seed_refresh_token(), "expires_at_ms": int(time.time() * 1000) + 3_600_000})
    fh.write_auth({"version": 1, "credential_pool": {"anthropic": rows}})
    loopback = "127.0.0.1,localhost"
    env = {"HTTPS_PROXY": proxy.url, "https_proxy": proxy.url, "NO_PROXY": loopback, "no_proxy": loopback,
           "SSL_CERT_FILE": str(ca.ca_pem)}
    return AnthropicOAuthRig(fh=fh, tokens=tokens, proxy=proxy, messages=messages, seed_refresh=seed, child_env=env)
