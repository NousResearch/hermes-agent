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
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import pytest
import yaml

from tests.e2e.core._pending_fixes import known_failure
from tests.fakes.providers.catalog_fake import USAGE_IN, USAGE_OUT, CatalogFake, Recorded, bare_path

REPO_ROOT = Path(__file__).resolve().parents[4]
TURN_TIMEOUT = 75.0
# A child that CONNECTs to a provider's real host and then sends nothing to its loopback fake for
# this long is talking to the vendor, not the fake: kill it and judge what it did. Healthy rows
# (zai, deepinfra, router) CONNECT their vendor host at startup and reach the fake <1 s later.
VENDOR_CONNECT_GRACE_S = 8.0
SHARDS = 3
FINAL = "CATALOG-TURN-COMPLETE"

# Wire dialect the fake must see for each transport the runtime can resolve. A transport missing
# here FAILS the matrix row (teach CatalogFake the dialect); it never becomes a skip.
DIALECT_OF_API_MODE = {"chat_completions": "chat", "anthropic_messages": "anthropic", "codex_responses": "responses"}
# Header that must carry the key, per dialect (Anthropic Messages = x-api-key; OpenAI wire = Bearer).
AUTH_HEADER_OF_DIALECT = {"chat": "authorization", "responses": "authorization", "anthropic": "x-api-key"}
# Where the documented base-URL override points, per dialect (Anthropic needs a ``/anthropic``
# path: the product only trusts an Anthropic-protocol override that looks like one).
URL_SUFFIX_OF_DIALECT = {"chat": "/v1", "responses": "/v1", "anthropic": "/anthropic"}
# Inference endpoint appended to the configured base URL, per dialect (the Anthropic SDK appends
# ``/v1/messages`` to an unversioned base; the OpenAI SDK appends to a ``/v1`` base).
ENDPOINT_OF_DIALECT = {"chat": "/chat/completions", "responses": "/responses", "anthropic": "/v1/messages"}

# The ONLY reasons a discovered provider may leave the matrix: auth types whose transport cannot
# be pointed at a loopback HTTP fake by config/env alone, and named keyless providers.
UNREDIRECTABLE_AUTH = {
    "aws_sdk": "AWS SigV4 via boto3 default chain; no HTTP fake for bedrock-runtime in this lane",
    "vertex": "Google ADC/OAuth2 token minting is required before any request",
    "external_process": "speaks ACP to a local vendor CLI subprocess, not HTTP",
    "copilot": "the GitHub token exchange URL is hardcoded (api.github.com/copilot_internal/v2/token)",
    "oauth_device_code": "login + refresh covered by test_catalog_oauth.py",
    "oauth_external": "login + refresh covered by test_catalog_oauth.py (where redirectable)",
}
KEYLESS_PROVIDERS = {
    "custom": "user-defined endpoint keyed by config api_key; covered by test_chat_custom_endpoint.py",
}
# Hosts a hermetic run may reach without carrying any vendor credential.
CREDENTIAL_FREE_HOSTS = frozenset({"models.dev:443"})

_SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_ACCESS_KEY", "_KEY")
_PASSTHROUGH = frozenset({"PATH", "LANG", "LANGUAGE", "USER", "LOGNAME", "SHELL", "TMPDIR", "TZ"})


class CatalogGap(AssertionError):
    """Raised ONLY by a gated final assertion, so ``known_failure(raises=CatalogGap)`` can never
    swallow a harness failure (timeout, boot crash, precondition assert)."""


def gate(ok: bool, message: str) -> None:
    if not ok:
        raise CatalogGap(message)


@dataclass(frozen=True)
class Known:
    """An open bug, gated on its OWN observed signature (a regex on the gated message)."""

    pattern: str
    reason: str
    cells: frozenset[str] = frozenset()


def known_gate(known: Known | None) -> contextlib.AbstractContextManager:
    return known_failure(known.pattern, known.reason, raises=CatalogGap) if known else contextlib.nullcontext()


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
        if self.name in KEYLESS_PROVIDERS:
            return f"{self.name}: {KEYLESS_PROVIDERS[self.name]}"
        return None

    @property
    def base_path(self) -> str:
        """Path of the base URL every file configures for this row on its loopback fake."""
        return f"/{self.name}{URL_SUFFIX_OF_DIALECT[self.dialect or 'chat']}"

    def routes(self) -> dict[str, str]:
        """Exact path -> dialect the fake answers for this row; everything else is a 404."""
        assert self.dialect, f"{self.name}: transport {self.api_mode!r} has no loopback dialect"
        p = self.base_path
        out = {p + ENDPOINT_OF_DIALECT[self.dialect]: self.dialect}
        if self.dialect == "responses" or self.host_mandated:
            out[p + ENDPOINT_OF_DIALECT["chat"]] = "chat"
        # Listing: ``/models`` under the configured base; under the Anthropic override also the
        # Anthropic API's own ``/v1/models`` (the SDK's versioned path).
        out[p + "/models"] = "listing"
        if self.dialect == "anthropic":
            out[p + "/v1/models"] = "listing"
        return out

    def listing_routes(self) -> set[str]:
        return {p for p, d in self.routes().items() if d == "listing"}

    def vendor_host(self) -> str:
        return f"{urlsplit(self.base_url).hostname}:443" if self.base_url.startswith("https://") else ""


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
           # auto_recovery_cycles: 0 — the documented post-exhaustion ladder (15/30/60/60/60 s) would
           # otherwise park a transport-failed turn for minutes; one bounded attempt is the contract here.
           "agent": {"api_max_retries": 1, "auto_recovery_cycles": 0}, "updates": {"check": False},
           **(extra_cfg or {})}
    (home / ".hermes" / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return home


def _vendor_stall(watch: Iterable[CatalogFake], sentinel: CatalogFake, vendor_hosts: frozenset[str]) -> str:
    """Why the child should be killed now, or "" — it CONNECTed to a provider's real host and
    no watched fake has seen an inference request since, for ``VENDOR_CONNECT_GRACE_S``."""
    hits = [r for r in list(sentinel.egress) if r.path in vendor_hosts]
    if not hits:
        return ""
    since = hits[0].t
    if any(r.method == "POST" and r.t >= since for f in watch for r in list(f.requests)):
        return ""
    if time.time() - since < VENDOR_CONNECT_GRACE_S:
        return ""
    return f"killed {VENDOR_CONNECT_GRACE_S}s after CONNECT {hits[0].path} with no inference at the fake"


def run_hermes(home: Path, cwd: Path, env_extra: dict[str, str], *args: str, timeout: float = TURN_TIMEOUT,
               sentinel: CatalogFake | None = None, watch: Iterable[CatalogFake] = (),
               vendor_hosts: frozenset[str] = frozenset()) -> subprocess.CompletedProcess:
    """Run the real CLI; a child that stalls on a vendor host (see :func:`_vendor_stall`) or
    outlives ``timeout`` is killed (rc -9) and the reason lands in stderr."""
    watch = tuple(watch)
    with tempfile.TemporaryFile("w+", encoding="utf-8") as out, tempfile.TemporaryFile("w+", encoding="utf-8") as err:
        proc = subprocess.Popen([sys.executable, "-m", "hermes_cli.main", *args], cwd=cwd,
                                env=hermetic_env(home, env_extra), stdout=out, stderr=err, text=True,
                                stdin=subprocess.DEVNULL)
        deadline, why = time.monotonic() + timeout, ""
        while proc.poll() is None:
            why = (f"TIMEOUT after {timeout}s" if time.monotonic() > deadline else
                   _vendor_stall(watch, sentinel, vendor_hosts) if sentinel is not None else "")
            if why:
                proc.kill()
                break
            time.sleep(0.2)
        rc = proc.wait()
        out.seek(0)
        err.seek(0)
        return subprocess.CompletedProcess(proc.args, -9 if why else rc, out.read(), err.read() + (f"\n{why}" if why else ""))


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

    def signature(self) -> str:
        """Observed facts a KNOWN bug's pattern keys on (one line, first in every gated message)."""
        n = sum(1 for r in self.requests if r.method == "POST")
        return f"fake_inference={n} egress={self.egress}"

    def detail(self) -> str:
        reqs = [f"{r.method} {r.path} [{r.dialect}] creds={credential_values(r, self.secrets)}" for r in self.requests]
        return (f"rc={self.rc} wall={self.wall_s}s\n  usage={self.usage}\n"
                f"  requests={reqs}\n  stdout={self.stdout[-600:]!r}\n  stderr={self.stderr[-1200:]!r}")


def drive_turn(row: Row, root: Path, catalog: list[Row]) -> TurnResult:
    """One oneshot turn with one tool round trip for ``row`` against its own fake."""
    project = root / "project"
    project.mkdir(parents=True, exist_ok=True)
    canary = f"CANARY-{row.name}-{os.urandom(4).hex()}"
    (project / "canary.txt").write_text(canary + "\n", encoding="utf-8")
    keys = decoy_keys(catalog)
    started = time.monotonic()
    with CatalogFake(tool_args={"path": str(project / "canary.txt")}, final_text=FINAL, routes=row.routes()) as fake:
        home = write_home(root, {"provider": row.name, "base_url": fake.origin + row.base_path})
        proc = run_hermes(home, project, {**keys, **fake.proxy_env()}, "-z", "Read canary.txt and report.",
                          sentinel=fake, watch=[fake], vendor_hosts=frozenset(provider_hosts(catalog)))
        requests = list(fake.requests)
        egress = fake.egress_hosts()
    return TurnResult(row=row, rc=proc.returncode, stdout=proc.stdout, stderr=proc.stderr, requests=requests,
                      egress=egress, own_key=keys[row.key_env or ""], secrets=set(keys.values()),
                      usage=session_usage(home), canary=canary, wall_s=round(time.monotonic() - started, 1))


def provider_hosts(catalog: list[Row]) -> set[str]:
    """``host:443`` of every provider's canonical HTTPS endpoint (the egress sentinel's CONNECT form)."""
    return {r.vendor_host() for r in catalog} - {""}


def evaluate(t: TurnResult, catalog: list[Row]) -> dict[str, bool]:
    """Every invariant of one row. Relationship checks only — never literals of today's output."""
    inference = [r for r in t.requests if r.method == "POST"]
    # A transport mandated by the provider's own host may fall back to chat at a foreign URL.
    expected = {t.row.dialect} | ({"chat"} if t.row.host_mandated else set())
    main = [r for r in inference if isinstance(r.body, dict) and r.body.get("tools")]
    answered = [r for r in main if r.dialect != "unknown"]
    foreign = t.secrets - {t.own_key}
    foreign_hosts = provider_hosts([r for r in catalog if r.name != t.row.name]) - provider_hosts([t.row])
    return {
        "turn_completed": t.rc == 0 and FINAL in t.stdout,
        # The fake answers ONLY the exact configured base path + dialect endpoint (else 404,
        # dialect "unknown"), so a mangled path under the right prefix is red here.
        "reached_own_endpoint": bool(inference) and all(r.dialect != "unknown" for r in inference),
        "dialect_matches_transport": bool(main) and all(r.dialect in expected for r in main),
        "tool_round_trip": any(t.canary in json.dumps(r.body) for r in main),
        "own_key_in_auth_header": bool(inference) and all(
            t.own_key in r.headers.get(AUTH_HEADER_OF_DIALECT.get(r.dialect, "authorization"), "") for r in inference),
        "no_foreign_key_on_wire": not any(s in v for r in t.requests for v in r.headers.values() for s in foreign),
        # Egress sentinel: nothing may leave for ANOTHER provider's host (CONNECT target).
        "no_egress_to_foreign_provider_hosts": not [h for h in t.egress if h in foreign_hosts],
        # The session row charges exactly what the fake reported for every main-turn call it
        # answered (auxiliary title calls are not charged to the session row).
        "usage_matches_wire": bool(answered) and bool(t.usage)
        and t.usage["input_tokens"] == USAGE_IN * len(answered) and t.usage["output_tokens"] == USAGE_OUT * len(answered),
        # Unknown pricing (a model no catalog prices) must be explicit, never a silent $0 estimate.
        # (usage_matches_wire owns absence; this cell judges only a row that exists.)
        "cost_not_silent_zero": not t.usage or not (
            (t.usage.get("estimated_cost_usd") in (0, 0.0)) and t.usage.get("cost_status") not in (None, "unknown")),
    }


# --- matrix shards ------------------------------------------------------------------------------

# Open bugs per provider row: the listed cells may be red ONLY while the gated message carries the
# bug's own signature; any other red cell fails the row, and a fixed bug simply passes.
MATRIX_KNOWN: dict[str, Known] = {
    "xai": Known(
        pattern=r"fake_inference=0 egress=\[[^\]]*'api\.x\.ai:443'",
        reason="#121347 xai ignores model.base_url; request + key go to api.x.ai",
        cells=frozenset({"turn_completed", "reached_own_endpoint", "dialect_matches_transport", "tool_round_trip",
                         "own_key_in_auth_header", "usage_matches_wire"})),
}


def shard_rows(shard: int) -> list[Row]:
    return [r for r in discover_catalog() if shard_of(r.name) == shard]


def shard_params(shard: int) -> list:
    """Every discovered row of the shard; only explicit auth types / keyless names skip."""
    return [pytest.param(r, id=r.name, marks=[pytest.mark.skip(reason=r.skip_reason())] if r.skip_reason() else [])
            for r in shard_rows(shard)]


def drive_shard(shard: int, tmp_path_factory: pytest.TempPathFactory) -> dict[str, TurnResult]:
    """Every runnable row of the shard driven concurrently (own home, fake and process each)."""
    catalog = discover_catalog()
    runnable = [r for r in shard_rows(shard) if r.skip_reason() is None and r.dialect]
    roots = {r.name: Path(tmp_path_factory.mktemp(f"cat-{r.name}")) for r in runnable}
    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="catalog") as pool:
        futs = {r.name: pool.submit(drive_turn, r, roots[r.name], catalog) for r in runnable}
        return {name: f.result() for name, f in futs.items()}


def check_row(row: Row, turns: dict[str, TurnResult]) -> None:
    assert row.dialect, (f"{row.name}: transport {row.api_mode!r} has no loopback dialect in CatalogFake — "
                         f"add it to DIALECT_OF_API_MODE (a provider may not leave the matrix as a skip)")
    t = turns[row.name]
    cells = evaluate(t, discover_catalog())
    failed = sorted(c for c, ok in cells.items() if not ok)
    known = MATRIX_KNOWN.get(row.name)
    unlisted = [c for c in failed if not known or c not in known.cells]
    assert not unlisted, f"{row.name} ({row.api_mode}/{row.auth_type}): cells red: {unlisted}\n{t.signature()}\n{t.detail()}"
    with known_gate(known):
        gate(not failed, f"{t.signature()}\n{row.name}: cells red: {failed}\n{t.detail()}")
