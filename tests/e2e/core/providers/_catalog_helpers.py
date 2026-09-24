"""Shared harness for the provider-catalog E2E matrix.

The provider list is NEVER hardcoded: :func:`discover_catalog` runs the real plugin discovery
(``providers.list_providers()``) in a clean child interpreter, so a new plugin under
``plugins/model-providers/`` joins every matrix automatically. Each row is driven through the
real ``python -m hermes_cli.main`` in a hermetic HOME against
:class:`tests.fakes.providers.catalog_fake.CatalogFake`, redirected the way the product documents
(``model.provider`` + ``model.base_url`` in config.yaml; ``/anthropic`` path for the Anthropic
Messages dialect). Every other provider's key is present as a decoy, and all non-loopback egress
goes through the fake's sentinel proxy, so a credential sent to the wrong host is observable.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
import re

import pytest
import yaml

from tests.fakes.providers.catalog_fake import USAGE_IN, USAGE_OUT, CatalogFake, Recorded

REPO_ROOT = Path(__file__).resolve().parents[4]
TURN_TIMEOUT = 75.0
SHARDS = 3
FINAL = "CATALOG-TURN-COMPLETE"

# Wire dialect the fake must see for each transport the runtime can resolve.
DIALECT_OF_API_MODE = {"chat_completions": "chat", "anthropic_messages": "anthropic", "codex_responses": "responses"}
# Header that must carry the key, per dialect (Anthropic Messages = x-api-key; OpenAI wire = Bearer).
AUTH_HEADER_OF_DIALECT = {"chat": "authorization", "responses": "authorization", "anthropic": "x-api-key"}
# Where the documented base-URL override points, per dialect (Anthropic needs a ``/anthropic``
# path: the product only trusts an Anthropic-protocol override that looks like one).
URL_SUFFIX_OF_DIALECT = {"chat": "/v1", "responses": "/v1", "anthropic": "/anthropic"}

# auth types whose transport cannot be pointed at a loopback HTTP fake by config/env alone.
UNREDIRECTABLE_AUTH = {
    "aws_sdk": "AWS SigV4 via boto3 default chain; no HTTP fake for bedrock-runtime in this lane",
    "vertex": "Google ADC/OAuth2 token minting is required before any request",
    "external_process": "speaks ACP to a local vendor CLI subprocess, not HTTP",
    "copilot": "the GitHub token exchange URL is hardcoded (api.github.com/copilot_internal/v2/token)",
    "oauth_device_code": "login + refresh covered by test_catalog_oauth.py",
    "oauth_external": "login + refresh covered by test_catalog_oauth.py (where redirectable)",
}
# Hosts a hermetic run may reach without carrying any vendor credential.
CREDENTIAL_FREE_HOSTS = frozenset({"models.dev:443"})

_SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_ACCESS_KEY", "_KEY")
_PASSTHROUGH = frozenset({"PATH", "LANG", "LANGUAGE", "USER", "LOGNAME", "SHELL", "TMPDIR", "TZ"})


@dataclass(frozen=True)
class Row:
    name: str
    api_mode: str
    auth_type: str
    key_env: str | None
    base_url: str
    aliases: tuple[str, ...]
    supports_model_listing: bool
    # The declared transport is mandated by the provider's OWN host (e.g. a Responses-native
    # endpoint); at a foreign base URL the product speaks plain chat completions instead.
    host_mandated: bool = False

    @property
    def dialect(self) -> str | None:
        return DIALECT_OF_API_MODE.get(self.api_mode)

    def skip_reason(self) -> str | None:
        if self.auth_type in UNREDIRECTABLE_AUTH:
            return f"{self.name}: {UNREDIRECTABLE_AUTH[self.auth_type]}"
        if self.dialect is None:
            return f"{self.name}: transport {self.api_mode!r} has no loopback dialect in CatalogFake"
        if self.key_env is None:
            return f"{self.name}: declares no credential env var (keyless/custom endpoint)"
        return None


_CATALOG: list[Row] | None = None

_DISCOVER = """
import json, providers
from hermes_cli.providers import host_mandated_api_mode
out = []
for p in providers.list_providers():
    key = next((e for e in p.env_vars if not e.endswith("_BASE_URL")), None)
    out.append(dict(name=p.name, api_mode=p.api_mode, auth_type=p.auth_type, key_env=key,
                    base_url=p.base_url, aliases=list(p.aliases),
                    supports_model_listing=bool(p.supports_model_listing),
                    host_mandated=host_mandated_api_mode(p.base_url) is not None))
print("CATALOG=" + json.dumps(out))
"""


def discover_catalog() -> list[Row]:
    """Real plugin discovery in a clean child (no user plugins, no credentials)."""
    global _CATALOG
    if _CATALOG is None:
        with tempfile.TemporaryDirectory(prefix="catalog-discover-") as d:
            env = {k: v for k, v in os.environ.items() if k in _PASSTHROUGH}
            env.update(HOME=d, HERMES_HOME=str(Path(d) / ".hermes"), PYTHONPATH=str(REPO_ROOT))
            proc = subprocess.run([sys.executable, "-c", _DISCOVER], env=env, capture_output=True,
                                  text=True, timeout=120, cwd=str(REPO_ROOT))
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("CATALOG=")), None)
        assert line, f"provider discovery failed: {proc.stderr[-2000:]}"
        _CATALOG = sorted((Row(**{**r, "aliases": tuple(r["aliases"])}) for r in json.loads(line[8:])),
                          key=lambda r: r.name)
    return _CATALOG


def shard_of(name: str, shards: int = SHARDS) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % shards


def decoy_keys(catalog: list[Row]) -> dict[str, str]:
    """One distinct fake secret per credential env var any provider declares."""
    return {r.key_env: f"sk-cat-{r.key_env.lower()}" for r in catalog if r.key_env}


def hermetic_env(home: Path, extra: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if (k in _PASSTHROUGH or k.startswith("LC_")) and not k.endswith(_SECRET_SUFFIXES)}
    env.update({
        "HOME": str(home), "HERMES_HOME": str(home / ".hermes"), "PYTHONPATH": str(REPO_ROOT),
        "PYTHONUNBUFFERED": "1", "NO_COLOR": "1", "TERM": "dumb",
        # The child's HOME is tmp_path; this is the state-db guard's documented child escape hatch.
        "HERMES_STATE_DB_GUARD_BYPASS": "1",
    })
    env.update(extra)
    return env


def write_home(root: Path, model: dict[str, Any], extra_cfg: dict[str, Any] | None = None) -> Path:
    home = root / "home"
    (home / ".hermes").mkdir(parents=True, exist_ok=True)
    cfg = {"model": {"default": "catalog-model-a", "context_length": 128000, **model},
           "agent": {"api_max_retries": 1}, "updates": {"check": False}, **(extra_cfg or {})}
    (home / ".hermes" / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return home


def run_hermes(home: Path, cwd: Path, env_extra: dict[str, str], *args: str,
               timeout: float = TURN_TIMEOUT) -> subprocess.CompletedProcess:
    try:
        return subprocess.run([sys.executable, "-m", "hermes_cli.main", *args], cwd=cwd,
                              env=hermetic_env(home, env_extra), capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        return subprocess.CompletedProcess(exc.cmd, -9, out, f"TIMEOUT after {timeout}s")


def session_usage(home: Path) -> dict[str, Any] | None:
    db = home / ".hermes" / "state.db"
    if not db.exists():
        return None
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT input_tokens, output_tokens, estimated_cost_usd, cost_status "
                          "FROM sessions ORDER BY started_at DESC LIMIT 1").fetchone()
        return dict(row) if row else None
    finally:
        con.close()


def credential_values(rec: Recorded, secrets: set[str]) -> dict[str, str]:
    """header -> the secret it carries, for every header carrying any known secret."""
    return {h: s for h, v in rec.headers.items() for s in secrets if s in v}


@dataclass
class TurnResult:
    row: Row
    rc: int
    stdout: str
    stderr: str
    requests: list[Recorded]
    egress: list[str]
    own_key: str
    secrets: set[str]
    usage: dict[str, Any] | None
    canary: str
    wall_s: float
    cells: dict[str, bool] = field(default_factory=dict)

    def detail(self) -> str:
        reqs = [f"{r.method} {r.path} creds={credential_values(r, self.secrets)}" for r in self.requests]
        return (f"rc={self.rc} wall={self.wall_s}s egress={self.egress}\n  usage={self.usage}\n"
                f"  requests={reqs}\n  stdout={self.stdout[-600:]!r}\n  stderr={self.stderr[-1200:]!r}")


def drive_turn(row: Row, root: Path, catalog: list[Row]) -> TurnResult:
    """One oneshot turn with one tool round trip for ``row`` against its own fake."""
    project = root / "project"
    project.mkdir(parents=True, exist_ok=True)
    canary = f"CANARY-{row.name}-{os.urandom(4).hex()}"
    (project / "canary.txt").write_text(canary + "\n", encoding="utf-8")
    keys = decoy_keys(catalog)
    started = time.monotonic()
    with CatalogFake(tool_args={"path": str(project / "canary.txt")}, final_text=FINAL) as fake:
        base = f"{fake.origin}/{row.name}{URL_SUFFIX_OF_DIALECT[row.dialect or 'chat']}"
        home = write_home(root, {"provider": row.name, "base_url": base})
        proc = run_hermes(home, project, {**keys, **fake.proxy_env()}, "-z", "Read canary.txt and report.")
        requests = list(fake.requests)
        egress = fake.egress_hosts()
    return TurnResult(row=row, rc=proc.returncode, stdout=proc.stdout, stderr=proc.stderr, requests=requests,
                      egress=egress, own_key=keys[row.key_env or ""], secrets=set(keys.values()),
                      usage=session_usage(home), canary=canary, wall_s=round(time.monotonic() - started, 1))


def provider_hosts(catalog: list[Row]) -> set[str]:
    from urllib.parse import urlsplit
    return {urlsplit(r.base_url).hostname or "" for r in catalog if r.base_url.startswith("https://")} - {""}


def evaluate(t: TurnResult, catalog: list[Row]) -> dict[str, bool]:
    """Every invariant of one row. Relationship checks only — never literals of today's output."""
    inference = [r for r in t.requests if r.method == "POST"]
    # A transport mandated by the provider's own host may fall back to chat at a foreign URL.
    expected = {t.row.dialect} | ({"chat"} if t.row.host_mandated else set())
    main = [r for r in inference if isinstance(r.body, dict) and r.body.get("tools")]
    foreign = t.secrets - {t.own_key}
    foreign_hosts = provider_hosts([r for r in catalog if r.name != t.row.name]) - provider_hosts([t.row])
    return {
        "turn_completed": t.rc == 0 and FINAL in t.stdout,
        "reached_own_endpoint": bool(inference) and all(r.path.startswith(f"/{t.row.name}/") for r in inference),
        "dialect_matches_transport": bool(main) and all(r.dialect in expected for r in main),
        "tool_round_trip": any(t.canary in json.dumps(r.body) for r in main),
        "own_key_in_auth_header": bool(inference) and all(
            t.own_key in r.headers.get(AUTH_HEADER_OF_DIALECT.get(r.dialect, "authorization"), "") for r in inference),
        "no_foreign_key_on_wire": not any(s in v for r in t.requests for v in r.headers.values() for s in foreign),
        # Egress sentinel: nothing may leave for ANOTHER provider's host (CONNECT target).
        "no_egress_to_foreign_provider_hosts": not [h for h in t.egress if h.rsplit(":", 1)[0] in foreign_hosts],
        "usage_recorded": bool(t.usage) and (t.usage["input_tokens"] or 0) >= USAGE_IN
        and (t.usage["output_tokens"] or 0) >= USAGE_OUT,
        # Unknown pricing (a model no catalog prices) must be explicit, never a silent $0 estimate.
        # (usage_recorded owns absence; this cell judges only a row that exists.)
        "cost_not_silent_zero": not t.usage or not (
            (t.usage.get("estimated_cost_usd") in (0, 0.0)) and t.usage.get("cost_status") not in (None, "unknown")),
    }


@contextlib.contextmanager
def strict_known(pattern: str, reason: str) -> Iterator[None]:
    """Strict run-time xfail for a filed bug: an AssertionError whose text matches ``pattern`` XFAILs
    the cell; any other failure propagates; a clean pass FAILS, so the fix PR must drop the entry
    (the campaign's strict-KNOWN rule, applied to cells whose assertions run after a live wait)."""
    try:
        yield
    except AssertionError as exc:
        if not re.search(pattern, str(exc)):
            raise
        pytest.xfail(f"{reason} [observed: {str(exc).splitlines()[0][:240]}]")
    pytest.fail(f"KNOWN bug now fixed — drop its KNOWN entry: {reason}")
