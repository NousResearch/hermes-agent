"""Model listing and ``/model --provider`` switching per discovered provider.

Each runnable provider is configured at a custom ``model.base_url`` (a loopback fake serving
provider-unique model ids at the EXACT listing path under that base; anything else 404s). Child
processes per row call the real product functions:

* listing — ``hermes_cli.models.provider_model_ids`` (the catalog the ``/model`` picker renders)
  must query the CONFIGURED endpoint at an exact listing path under it, never the provider's
  canonical host (#120844 class): the relay's key must not be addressed to the vendor; rows that
  already do so must also render the relay's own model ids;
* switch — ``hermes_cli.model_switch.switch_model(explicit_provider=<row>)`` must resolve to that
  provider at its configured endpoint, not to an alias on another endpoint (#120295 class);
* listing degradation — for the rows that already list from the configured endpoint, a 404 and a
  hanging listing must still return (bounded) without crashing and without falling back to the
  vendor host.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.e2e.core.providers._catalog_helpers import (
    Known, Row, decoy_keys, discover_catalog, gate, hermetic_env, known_gate, write_home,
)
from tests.fakes.providers.catalog_fake import CatalogFake, bare_path

CATALOG = discover_catalog()
# Rows with no loopback dialect FAIL in the matrix shards; they cannot be configured here.
ROWS = [r for r in CATALOG if r.skip_reason() is None and r.dialect]
PROBE_TIMEOUT = 90.0
# A hanging listing must give up well before the fake would release it.
HANG_S = 60.0
HANG_BOUND_S = 40.0

_LISTING_BROKEN = (
    "ai-gateway", "alibaba", "alibaba-cn", "alibaba-coding-plan", "alibaba-coding-plan-cn",
    "alibaba-token-plan", "alibaba-token-plan-cn", "arcee", "commandcode", "commandcode-anthropic",
    "deepinfra", "deepseek", "fireworks", "gemini", "gmi", "huggingface", "kilocode", "kimi-coding",
    "kimi-coding-cn", "meta-ai", "minimax", "minimax-cn", "nebius-token-factory", "novita", "nvidia",
    "ollama-cloud", "opencode-go", "opencode-zen", "openrouter", "router", "stepfun", "upstage", "xai",
    "xiaomi", "zai",
)
# Signature: the picker went to the vendor host and returned without any probe error (it renders
# the vendor/static list instead of the relay's).
LISTING_KNOWN: dict[str, Known] = {n: Known(
    r"^canonical_host_hit=True probe_error=None",
    "#121387 picker listing ignores model.base_url; queries the vendor host") for n in _LISTING_BROKEN}
SWITCH_KNOWN: dict[str, Known] = {
    "nebius-token-factory": Known(
        r"^ok=False .*not found in this provider's catalog.*'api\.tokenfactory\.nebius\.com:443'",
        "#121388 nebius /model switch validates against the vendor host, ignoring model.base_url"),
    "xai": Known(r"^ok=True got='xai' base_url='https://api\.x\.ai/v1'", "#121347 xai ignores model.base_url"),
}
# Rows that list from the configured endpoint today: they also carry the 404 / hang cells.
CONTROL_ROWS = [r for r in ROWS if r.supports_model_listing and r.name not in LISTING_KNOWN]

_LIST = r"""
import json, sys
from hermes_cli.models import provider_model_ids
try:
    print("RESULT=" + json.dumps({"ids": provider_model_ids(sys.argv[1], force_refresh=True)}))
except Exception as exc:
    print("RESULT=" + json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
"""
_SWITCH = r"""
import json, sys
from hermes_cli.model_switch import switch_model
from hermes_cli.providers import normalize_provider
r = switch_model(raw_input=sys.argv[2], current_provider="custom", current_model="x", current_base_url="",
                 current_api_key="", explicit_provider=sys.argv[1])
print("RESULT=" + json.dumps({"ok": bool(r.success), "provider": r.target_provider, "base_url": r.base_url,
                              "error": r.error_message, "want": normalize_provider(sys.argv[1]),
                              "got": normalize_provider(r.target_provider or "")}))
"""
_KINDS = {"list": (_LIST, {}), "switch": (_SWITCH, {}),
          "list404": (_LIST, {"models_status": 404}), "listhang": (_LIST, {"models_hang_s": HANG_S})}


def _probe(row: Row, root: Path, kind: str) -> dict:
    script, fake_kwargs = _KINDS[kind]
    unique = f"catalog-{row.name}-alpha"
    keys = decoy_keys(CATALOG)
    with CatalogFake(models=[unique, f"catalog-{row.name}-beta"], routes=row.routes(), **fake_kwargs) as fake:
        base = fake.origin + row.base_path
        home = write_home(root, {"provider": row.name, "base_url": base})
        started = time.monotonic()
        try:
            proc = subprocess.run([sys.executable, "-c", script, row.name, unique], cwd=root, capture_output=True,
                                  text=True, env=hermetic_env(home, {**keys, **fake.proxy_env()}),
                                  timeout=PROBE_TIMEOUT, stdin=subprocess.DEVNULL)
            out, err = proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            out, err = "", f"TIMEOUT after {PROBE_TIMEOUT}s"
        wall = round(time.monotonic() - started, 1)
        listings, egress = fake.listings(), fake.egress_hosts()
    line = next((ln for ln in out.splitlines() if ln.startswith("RESULT=")), None)
    res = json.loads(line[7:]) if line else {"error": f"probe crashed: {err[-1200:]}"}
    return {**res, "unique": unique, "base": base, "wall_s": wall, "egress": egress,
            "listing_paths": [r.path for r in listings], "canonical_host_hit": row.vendor_host() in egress}


@pytest.fixture(scope="module")
def probes(tmp_path_factory: pytest.TempPathFactory) -> dict[tuple[str, str], dict]:
    jobs = [(r, k) for r in ROWS for k in ("list", "switch")] + [
        (r, k) for r in CONTROL_ROWS for k in ("list404", "listhang")]
    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="listing") as pool:
        futs = {(r.name, k): pool.submit(_probe, r, Path(tmp_path_factory.mktemp(f"{k}-{r.name}")), k)
                for r, k in jobs}
        return {key: f.result() for key, f in futs.items()}


def _listing_facts(ls: dict) -> str:
    return (f"canonical_host_hit={ls['canonical_host_hit']} probe_error={ls.get('error')!r} "
            f"paths={ls['listing_paths']} egress={ls['egress']} wall={ls['wall_s']}s ids={(ls.get('ids') or [])[:6]}")


@pytest.mark.parametrize("row", [pytest.param(r, id=r.name) for r in ROWS])
def test_listing_uses_configured_endpoint(row: Row, probes: dict) -> None:
    ls = probes[(row.name, "list")]
    routes = row.listing_routes()
    ok = (not row.supports_model_listing) or (
        "error" not in ls and bool(ls["listing_paths"]) and all(bare_path(p) in routes for p in ls["listing_paths"])
        and not ls["canonical_host_hit"])
    with known_gate(LISTING_KNOWN.get(row.name)):
        gate(ok, f"{_listing_facts(ls)}\n{row.name}: picker did not list the configured endpoint (want {sorted(routes)})")


@pytest.mark.parametrize("row", [pytest.param(r, id=r.name) for r in CONTROL_ROWS])
def test_listing_renders_relay_models(row: Row, probes: dict) -> None:
    """The rows that list from the configured endpoint render the relay's own model ids."""
    ls = probes[(row.name, "list")]
    assert ls["unique"] in (ls.get("ids") or []), f"{row.name}: relay models not rendered\n{_listing_facts(ls)}"


@pytest.mark.parametrize("row", [pytest.param(r, id=r.name) for r in ROWS])
def test_switch_resolves_requested_provider(row: Row, probes: dict) -> None:
    sw = probes[(row.name, "switch")]
    # Same provider (alias-normalised: ai-gateway == vercel) at the configured endpoint.
    ok = bool(sw.get("ok")) and bool(sw.get("want")) and sw.get("got") == sw["want"] and \
        str(sw.get("base_url") or "").rstrip("/") == sw["base"].rstrip("/")
    facts = (f"ok={sw.get('ok')} got={sw.get('got')!r} base_url={sw.get('base_url')!r} error={sw.get('error')!r} "
             f"egress={sw['egress']}")
    with known_gate(SWITCH_KNOWN.get(row.name)):
        gate(ok, f"{facts}\n{row.name}: switch did not resolve to {sw.get('want')!r} at {sw['base']}")


@pytest.mark.parametrize("kind", ["list404", "listhang"])
@pytest.mark.parametrize("row", [pytest.param(r, id=r.name) for r in CONTROL_ROWS])
def test_listing_failure_degrades_without_vendor_host(row: Row, kind: str, probes: dict) -> None:
    """A 404 or a hanging listing at the configured endpoint: the picker still returns (within
    ``HANG_BOUND_S``), asked the exact listing path, never crashed and never fell back to the vendor."""
    ls = probes[(row.name, kind)]
    routes = row.listing_routes()
    assert "error" not in ls, f"{row.name} [{kind}]: listing crashed\n{_listing_facts(ls)}"
    assert ls["wall_s"] < HANG_BOUND_S, f"{row.name} [{kind}]: listing blocked the picker\n{_listing_facts(ls)}"
    assert ls["listing_paths"] and all(bare_path(p) in routes for p in ls["listing_paths"]), (
        f"{row.name} [{kind}]: never asked {sorted(routes)}\n{_listing_facts(ls)}")
    assert not ls["canonical_host_hit"], f"{row.name} [{kind}]: fell back to the vendor host\n{_listing_facts(ls)}"
    assert ls["unique"] not in (ls.get("ids") or []), f"{row.name} [{kind}]: fake ids without a listing?"
