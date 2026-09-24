"""Model listing and ``/model --provider`` switching per discovered provider.

Each runnable provider is configured at a custom ``model.base_url`` (a loopback fake serving
provider-unique model ids). Two child processes per row call the real product functions:

* listing — ``hermes_cli.models.provider_model_ids`` (the catalog the ``/model`` picker renders)
  must query the CONFIGURED endpoint when the provider declares a listing endpoint, and never the
  provider's canonical host (#120844 class): a relay user's picker must list the relay's models,
  and the relay's key must not be addressed to the vendor;
* switch — ``hermes_cli.model_switch.switch_model(explicit_provider=<row>)`` must resolve to that
  provider at its configured endpoint, not to an alias on another endpoint (#120295 class).
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from tests.e2e.core.providers._catalog_helpers import (
    URL_SUFFIX_OF_DIALECT, Row, decoy_keys, discover_catalog, hermetic_env, write_home,
)
from tests.fakes.providers.catalog_fake import CatalogFake

CATALOG = discover_catalog()
ROWS = [r for r in CATALOG if r.skip_reason() is None]
_LISTING_IGNORES_BASE_URL = "#121387 picker listing ignores model.base_url; queries the vendor host"
_LISTING_BROKEN = (
    "ai-gateway", "alibaba", "alibaba-cn", "alibaba-coding-plan", "alibaba-coding-plan-cn",
    "alibaba-token-plan", "alibaba-token-plan-cn", "arcee", "commandcode", "commandcode-anthropic",
    "deepinfra", "deepseek", "fireworks", "gemini", "gmi", "huggingface", "kilocode", "kimi-coding",
    "kimi-coding-cn", "meta-ai", "minimax", "minimax-cn", "nebius-token-factory", "novita", "nvidia",
    "ollama-cloud", "opencode-go", "opencode-zen", "openrouter", "router", "stepfun", "upstage", "xai",
    "xiaomi", "zai",
)
# Strict: a listed cell that turns green fails the row until its entry is dropped.
KNOWN: dict[tuple[str, str], str] = {
    **{(n, "listing_uses_configured_endpoint"): _LISTING_IGNORES_BASE_URL for n in _LISTING_BROKEN},
    ("nebius-token-factory", "switch_resolves_requested_provider"):
        "#121388 nebius /model switch validates against the vendor host, ignoring model.base_url",
    ("xai", "switch_resolves_requested_provider"): "#121347 xai ignores model.base_url",
}

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


def _probe(row: Row, root: Path, script: str) -> dict:
    unique = f"catalog-{row.name}-alpha"
    keys = decoy_keys(CATALOG)
    with CatalogFake(models=[unique, f"catalog-{row.name}-beta"]) as fake:
        base = f"{fake.origin}/{row.name}{URL_SUFFIX_OF_DIALECT[row.dialect or 'chat']}"
        home = write_home(root, {"provider": row.name, "base_url": base})
        try:
            proc = subprocess.run([sys.executable, "-c", script, row.name, unique], cwd=root, capture_output=True,
                                  text=True, env=hermetic_env(home, {**keys, **fake.proxy_env()}), timeout=90,
                                  stdin=subprocess.DEVNULL)
            out, err = proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            out, err = "", "TIMEOUT after 90s"
        listings, egress = fake.listings(), fake.egress_hosts()
    line = next((ln for ln in out.splitlines() if ln.startswith("RESULT=")), None)
    res = json.loads(line[7:]) if line else {"error": f"probe crashed: {err[-1200:]}"}
    return {**res, "base": base, "listing_paths": [r.path for r in listings], "egress": egress,
            "canonical_host_hit": f"{urlsplit(row.base_url).hostname}:443" in egress}


@pytest.fixture(scope="module")
def probes(tmp_path_factory: pytest.TempPathFactory) -> dict[tuple[str, str], dict]:
    jobs = [(r, kind, script) for r in ROWS for kind, script in (("list", _LIST), ("switch", _SWITCH))]
    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="listing") as pool:
        futs = {(r.name, kind): pool.submit(_probe, r, Path(tmp_path_factory.mktemp(f"{kind}-{r.name}")), script)
                for r, kind, script in jobs}
        return {k: f.result() for k, f in futs.items()}


def _cells(row: Row, ls: dict, sw: dict) -> dict[str, bool]:
    return {
        "listing_uses_configured_endpoint": (not row.supports_model_listing) or (
            "error" not in ls and bool(ls["listing_paths"])
            and all(p.startswith(f"/{row.name}/") for p in ls["listing_paths"]) and not ls["canonical_host_hit"]),
        # Same provider (alias-normalised: ai-gateway == vercel) at the configured endpoint.
        "switch_resolves_requested_provider": bool(sw.get("ok")) and bool(sw.get("want")) and sw.get("got") == sw["want"]
        and str(sw.get("base_url") or "").rstrip("/") == sw["base"].rstrip("/"),
    }


@pytest.mark.parametrize("row", [pytest.param(r, id=r.name) for r in ROWS])
def test_listing_and_switch(row: Row, probes: dict) -> None:
    ls, sw = probes[(row.name, "list")], probes[(row.name, "switch")]
    cells = _cells(row, ls, sw)
    known = {c: ref for (p, c), ref in KNOWN.items() if p == row.name}
    fixed = sorted(c for c in known if cells.get(c))
    assert not fixed, f"{row.name}: {fixed} now green — drop their KNOWN entries ({set(known.values())})"
    failed = sorted(c for c, ok in cells.items() if not ok and c not in known)
    detail = {"list": {**ls, "ids": (ls.get("ids") or [])[:6]}, "switch": sw}
    assert not failed, f"{row.name}: cells red: {failed}\n{json.dumps(detail, default=str)[:2500]}"
    if known:
        pytest.xfail(f"{sorted(known)}: {'; '.join(sorted(set(known.values())))}")
