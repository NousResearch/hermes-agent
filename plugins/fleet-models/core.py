"""fleet-models core — ONE file of model settings for the WeRoll Hermes fleet, compiled into the nine
config.yaml files Hermes actually reads.

    ~/.hermes/fleet/models.yaml      the source of truth (registry + per-agent waterfalls + policy)
    ~/.hermes/fleet/history/<id>/    pre-images of every apply (models.yaml + 9 configs) for revert
    ~/.hermes/fleet/history.jsonl    one line per apply / revert

Why a compiler and not a runtime lookup: Hermes reads model routing from each profile's own
config.yaml in a dozen places (gateway, workers, cron, delegation, auxiliary tasks, the desktop
backend). Rewriting every reader would fork upstream; compiling ONE document into exactly the keys
Hermes reads keeps upstream untouched and makes a change one write, validated and verified.

Stdlib + ruamel (round-trip: comments and key order in config.yaml survive). Importable by path
(the dashboard API and scripts load it with importlib) and runnable:

    core.py show | plan [--diff] | apply [--by NAME] | verify | import [--write] | souls [--write] | history | revert ID
"""
from __future__ import annotations

import copy
import datetime as _dt
import difflib
import fcntl
import io
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROFILES = ["root", "axel", "bob", "brain", "jobsy", "karl", "rodge", "steve-o", "switch"]
PROVIDERS = ("modelark", "openrouter")
OR_BASE = "https://openrouter.ai/api/v1"
ARK_CODING = "https://ark.ap-southeast.bytepluses.com/api/coding/v3"
ARK_KEY_ENV = "HERMES_CUSTOM_MODELARK_API_KEY"
HOST_KEYS = ("only", "order", "ignore", "sort", "require_parameters", "quantizations")
ROUTING_KEYS = ("only", "order", "ignore", "sort", "require_parameters")
SLOTS = ("main", "subagents", "cron")
REASONING = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


# ── locations ────────────────────────────────────────────────────────────────────────────────
def fleet_root() -> Path:
    """The fleet's root HERMES_HOME (``~/.hermes``) — also from inside a profile home."""
    if os.environ.get("FLEET_MODELS_ROOT"):
        return Path(os.environ["FLEET_MODELS_ROOT"]).expanduser()
    h = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()
    return h.parent.parent if h.parent.name == "profiles" else h


def doc_path(root: Optional[Path] = None) -> Path:
    return (root or fleet_root()) / "fleet" / "models.yaml"


def cfg_path(p: str, root: Optional[Path] = None) -> Path:
    r = root or fleet_root()
    return r / "config.yaml" if p == "root" else r / "profiles" / p / "config.yaml"


# ── yaml ─────────────────────────────────────────────────────────────────────────────────────
def _ruamel():
    from ruamel.yaml import YAML
    y = YAML()
    y.preserve_quotes = True
    y.width = 80
    return y


def _plain(o):
    """ruamel containers -> plain python (for comparisons / JSON)."""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    return o


def _load_plain(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_rt(text: str):
    """Round-trip load keeping each file's own list style (root indents '  - x', profiles '- x')."""
    y = _ruamel()
    aligned = len(re.findall(r"(?m)^( *)[^\s#-][^\n]*:\n\1- ", text))
    indented = len(re.findall(r"(?m)^( *)[^\s#-][^\n]*:\n\1  - ", text)) > aligned
    y.indent(mapping=2, sequence=4 if indented else 2, offset=2 if indented else 0)
    return y, y.load(text)


def _dump_rt(y, data) -> str:
    buf = io.StringIO()
    y.dump(data, buf)
    return buf.getvalue()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-fleet-models-{os.getpid()}")
    tmp.write_text(text)
    try:
        os.chmod(tmp, os.stat(path).st_mode & 0o777) if path.exists() else None
    except OSError:
        pass
    os.replace(tmp, path)


# ── the document ─────────────────────────────────────────────────────────────────────────────
_doc_cache: Dict[str, Tuple[float, dict]] = {}


def load_doc(root: Optional[Path] = None, *, fresh: bool = False) -> dict:
    """models.yaml as a plain dict (mtime-cached; cheap enough for per-request readers)."""
    p = doc_path(root)
    key = str(p)
    try:
        mt = p.stat().st_mtime
    except FileNotFoundError:
        raise FileNotFoundError(f"{p} not found — run `core.py import --write` first")
    if not fresh and key in _doc_cache and _doc_cache[key][0] == mt:
        return _doc_cache[key][1]
    d = _load_plain(p)
    _doc_cache[key] = (mt, d)
    return d


def chain_of(spec) -> List[str]:
    """A slot spec is either a list of aliases or {chain: [...], reasoning: ...}."""
    if spec is None:
        return []
    if isinstance(spec, dict):
        return list(spec.get("chain") or [])
    return list(spec)


def reasoning_of(spec) -> Optional[str]:
    return spec.get("reasoning") if isinstance(spec, dict) else None


def models_used(doc: dict, p: str) -> List[str]:
    a = doc["agents"][p]
    seen: List[str] = []
    specs = [a.get(s) for s in SLOTS] + list((a.get("aux") or {}).values())
    for spec in specs:
        for alias in chain_of(spec):
            if alias not in seen:
                seen.append(alias)
    return seen


# ── validation ───────────────────────────────────────────────────────────────────────────────
def validate(doc: dict, *, previous: Optional[dict] = None, unlock: Tuple[str, ...] = (),
             snapshot: Optional[dict] = None) -> Tuple[List[str], List[str]]:
    """(errors, warnings). Errors block an apply; warnings are shown with the plan."""
    E: List[str] = []
    W: List[str] = []
    pol = doc.get("policy") or {}
    if pol.get("data_collection") != "deny":
        E.append("policy.data_collection must be 'deny' — the no-training rule is not editable")
    models = doc.get("models") or {}
    agents = doc.get("agents") or {}
    for alias, m in models.items():
        where = f"models.{alias}"
        if not isinstance(m, dict) or not m.get("id"):
            E.append(f"{where}: needs an id"); continue
        if m.get("provider") not in PROVIDERS:
            E.append(f"{where}: provider must be one of {PROVIDERS}")
        if m.get("reasoning") is not None and str(m["reasoning"]) not in REASONING:
            E.append(f"{where}: reasoning '{m['reasoning']}' is not a level ({', '.join(REASONING)})")
        if m.get("provider") == "modelark":
            if m.get("billing") != "subscription":
                W.append(f"{where}: ModelArk Coding Plan models are billing: subscription")
            ce = m.get("cap_equivalent") or {}
            if not all(isinstance(ce.get(k), (int, float)) for k in ("input", "output", "cache_read")):
                E.append(f"{where}: cap_equivalent needs input/output/cache_read $/M (the $1 cap counts it)")
            if m.get("hosts"):
                E.append(f"{where}: host pins are OpenRouter-only")
        hosts = m.get("hosts") or {}
        for k in hosts:
            if k not in HOST_KEYS:
                E.append(f"{where}.hosts.{k}: unknown key (allowed {', '.join(HOST_KEYS)})")
        rules = m.get("rules") or {}
        first = rules.get("first_host")
        if first:
            lst = hosts.get("order") or hosts.get("only") or []
            if not lst or not str(lst[0]).split("/")[0] == first:
                E.append(f"{where}: rule — '{first}' must be the first host (Richie 2026-09-11)")
        # a model's own rule is binding (error); the fleet-wide figure is advice (warning)
        mu = rules.get("min_uptime") or pol.get("min_host_uptime")
        binding = bool(rules.get("min_uptime"))
        if snapshot and mu and m.get("provider") == "openrouter":
            eps = ((snapshot.get(m["id"]) or {}).get("endpoints") or {})
            for tag in (hosts.get("only") or hosts.get("order") or []):
                ep = eps.get(tag) or next((v for k, v in eps.items() if k.split("/")[0] == tag), None)
                if ep is None:
                    W.append(f"{where}: host '{tag}' not in the latest OpenRouter snapshot (unroutable pins fail SILENTLY)")
                elif isinstance(ep.get("uptime"), (int, float)) and ep["uptime"] < mu:
                    first_ok = first and tag.split("/")[0] == first
                    (W if first_ok or not binding else E).append(
                        f"{where}: host '{tag}' uptime {ep['uptime']:.1f}% < {mu}%" + (" (allowed: pinned first by rule)" if first_ok else ""))
    for p in PROFILES:
        if p not in agents:
            E.append(f"agents.{p}: missing"); continue
    for p, a in agents.items():
        if p not in PROFILES:
            E.append(f"agents.{p}: not a fleet profile"); continue
        if not chain_of(a.get("main")):
            E.append(f"agents.{p}.main: empty — every agent needs a primary")
        if a.get("reasoning") is not None and str(a["reasoning"]) not in REASONING:
            E.append(f"agents.{p}.reasoning: '{a['reasoning']}' is not a level")
        specs = [(s, a.get(s)) for s in SLOTS] + [(f"aux.{t}", s) for t, s in (a.get("aux") or {}).items()]
        for name, spec in specs:
            ch = chain_of(spec)
            where = f"agents.{p}.{name}"
            if spec is not None and not ch:
                E.append(f"{where}: empty chain (remove the slot to inherit instead)")
            if len(set(ch)) != len(ch):
                E.append(f"{where}: a model appears twice")
            for alias in ch:
                if alias not in models:
                    E.append(f"{where}: unknown model '{alias}'")
            if reasoning_of(spec) is not None and str(reasoning_of(spec)) not in REASONING:
                E.append(f"{where}: reasoning '{reasoning_of(spec)}' is not a level")
            known = [models[x] for x in ch if x in models]
            if name in SLOTS and any(m.get("tools") is False for m in known):
                E.append(f"{where}: a model without tool calling cannot run an agent loop")
            if name == "aux.vision" and any(not m.get("vision") for m in known):
                E.append(f"{where}: every vision rung must accept images")
            vendors = {m.get("vendor") for m in known}
            if len(known) > 1 and len(vendors) == 1:
                W.append(f"{where}: every rung is {vendors.pop()} — one vendor outage takes the whole chain down")
    if previous:
        for p, a in agents.items():
            pa = (previous.get("agents") or {}).get(p)
            if pa is not None and pa.get("locked") and p not in unlock and _plain(pa) != _plain(a):
                E.append(f"agents.{p} is locked ({a.get('name') or p}) — unlock it explicitly to change it")
    return E, W


# ── compile: doc -> one config.yaml ──────────────────────────────────────────────────────────
def _entry(m: dict, carry: Optional[dict] = None):
    from ruamel.yaml.comments import CommentedMap
    e = CommentedMap()
    e["provider"] = m["provider"]
    e["model"] = m["id"]
    for k, v in (carry or {}).items():  # e.g. a per-entry timeout someone tuned
        if k not in ("provider", "model", "extra_body", "base_url"):
            e[k] = v
    return e


def _seq(items):
    from ruamel.yaml.comments import CommentedSeq
    return CommentedSeq(items)


def _set(block, key, value, ch: List[str], where: str) -> None:
    if _plain(block.get(key)) != _plain(value):
        ch.append(f"{where}.{key}: {json.dumps(_plain(block.get(key)))} -> {json.dumps(_plain(value))}")
        block[key] = value


def _last_holder(node):
    """(container, key, slot) of the deepest last element — where ruamel parks the comment that
    FOLLOWS a block (e.g. a section banner after it). None for scalars/empties."""
    from ruamel.yaml.comments import CommentedMap, CommentedSeq
    if isinstance(node, CommentedMap) and len(node):
        k = list(node.keys())[-1]
        return _last_holder(node[k]) or (node, k, 2)
    if isinstance(node, CommentedSeq) and len(node):
        i = len(node) - 1
        return _last_holder(node[i]) or (node, i, 0)
    return None


def _take_trailing(node):
    """Detach and return the comment token hanging after *node*'s last element (if any)."""
    h = _last_holder(node)
    while h:
        cont, k, slot = h
        items = getattr(cont, "ca", None) and cont.ca.items.get(k)
        if items and len(items) > slot and items[slot] is not None:
            tok = items[slot]; items[slot] = None
            return tok, slot
        return None
    return None


def _pop(block, key, ch: List[str], where: str) -> None:
    if isinstance(block, dict) and key in block:
        ch.append(f"{where}.{key}: {json.dumps(_plain(block[key]))} -> (removed)")
        # keep a comment that trails the removed block (ruamel parks e.g. a banner for the NEXT
        # section there): re-hang it on the previous sibling so deleting a key never eats prose
        keys = list(block.keys())
        idx = keys.index(key)
        tok = _take_trailing(block[key])
        if tok is None and getattr(block, "ca", None) and block.ca.items.get(key):
            it = block.ca.items.get(key)
            tok = (it[2], 2) if len(it) > 2 and it[2] is not None else None
        del block[key]
        if tok is not None and idx > 0:
            prev = keys[idx - 1]
            h = _last_holder(block[prev]) or (block, prev, 2)
            cont, k, slot = h
            want = 4 if slot == 2 else 2
            items = cont.ca.items.setdefault(k, [None] * want)
            while len(items) < want:
                items.append(None)
            items[slot] = tok[0]


def _map(d, key):
    from ruamel.yaml.comments import CommentedMap
    if not isinstance(d.get(key), dict):
        d[key] = CommentedMap()
    return d[key]


def _fallback_list(M: dict, aliases: List[str], existing) -> Any:
    old = {str(e.get("model")): e for e in (existing or []) if isinstance(e, dict)}
    return _seq([_entry(M[a], old.get(M[a]["id"])) for a in aliases])


def compile_profile(doc: dict, p: str, d) -> List[str]:
    """Write doc's settings for profile *p* into the round-trip config *d*. Returns change lines.
    Only touches the keys fleet-models owns; everything else in config.yaml is left alone."""
    from ruamel.yaml.comments import CommentedMap
    ch: List[str] = []
    M = doc["models"]
    A = doc["agents"][p]
    used = models_used(doc, p)

    # 1. main loop: model + fallback_providers (the list Hermes reads; the nested and legacy forms go)
    main = chain_of(A["main"])
    m = _map(d, "model")
    prim = M[main[0]]
    _set(m, "provider", prim["provider"], ch, "model")
    _set(m, "default", prim["id"], ch, "model")
    if prim["provider"] == "modelark" and "openrouter.ai" in str(m.get("base_url") or ""):
        _pop(m, "base_url", ch, "model")
    if prim["provider"] == "openrouter" and "bytepluses.com" in str(m.get("base_url") or ""):
        _pop(m, "base_url", ch, "model")
    _pop(m, "fallback_model", ch, "model")  # read by nothing
    legacy = d.get("fallback_model")
    existing = list(d.get("fallback_providers") or []) + ([legacy] if isinstance(legacy, dict) else [])
    _pop(d, "fallback_model", ch, "")
    if main[1:]:
        _set(d, "fallback_providers", _fallback_list(M, main[1:], existing), ch, "")
    else:
        _pop(d, "fallback_providers", ch, "")

    # 2. subagents (delegation): pinned route + its OWN chain (a pinned child never borrows the parent's)
    sub = chain_of(A.get("subagents"))
    if sub:
        dl = _map(d, "delegation")
        _set(dl, "provider", M[sub[0]]["provider"], ch, "delegation")
        _set(dl, "model", M[sub[0]]["id"], ch, "delegation")
        if "openrouter.ai" in str(dl.get("base_url") or "") and M[sub[0]]["provider"] != "openrouter":
            _pop(dl, "base_url", ch, "delegation")
        if sub[1:]:
            _set(dl, "fallback_providers", _fallback_list(M, sub[1:], dl.get("fallback_providers")), ch, "delegation")
        else:
            _pop(dl, "fallback_providers", ch, "delegation")
    elif isinstance(d.get("delegation"), dict):
        for k in ("provider", "model", "fallback_providers"):
            _pop(d["delegation"], k, ch, "delegation")

    # 3. cron: fleet default route + cron.fallback_providers (honoured by the fleet-models plugin)
    cr = chain_of(A.get("cron"))
    if cr:
        c = _map(d, "cron")
        _set(c, "model_provider", M[cr[0]]["provider"], ch, "cron")
        _set(c, "model", M[cr[0]]["id"], ch, "cron")
        if cr[1:]:
            _set(c, "fallback_providers", _fallback_list(M, cr[1:], c.get("fallback_providers")), ch, "cron")
        else:
            _pop(c, "fallback_providers", ch, "cron")
    elif isinstance(d.get("cron"), dict):
        for k in ("model_provider", "model", "fallback_providers"):
            _pop(d["cron"], k, ch, "cron")

    # 4. auxiliary tasks
    want = A.get("aux") or {}
    aux = d.get("auxiliary") if isinstance(d.get("auxiliary"), dict) else None
    if want and aux is None:
        aux = _map(d, "auxiliary")
    for task, spec in want.items():
        chn = chain_of(spec)
        blk = _map(aux, task)
        w = f"auxiliary.{task}"
        _set(blk, "provider", M[chn[0]]["provider"], ch, w)
        _set(blk, "model", M[chn[0]]["id"], ch, w)
        if chn[1:]:
            _set(blk, "fallback_chain", _fallback_list(M, chn[1:], blk.get("fallback_chain")), ch, w)
        else:
            _pop(blk, "fallback_chain", ch, w)
        r = reasoning_of(spec)
        if r is not None:
            _set(blk, "reasoning_effort", r, ch, w)
        else:
            _pop(blk, "reasoning_effort", ch, w)
        eb = blk.get("extra_body") if isinstance(blk.get("extra_body"), dict) else None
        if M[chn[0]]["provider"] == "openrouter":
            if eb is None:
                eb = _map(blk, "extra_body")
            # host pins live per MODEL now (provider_routing.models) so they follow the model into
            # any fallback; the task keeps only the no-training floor.
            pv = CommentedMap(); pv["data_collection"] = "deny"
            _set(eb, "provider", pv, ch, w + ".extra_body")
        elif eb is not None:
            _pop(eb, "provider", ch, w + ".extra_body")
        if isinstance(blk.get("extra_body"), dict) and not blk["extra_body"]:
            del blk["extra_body"]
    if aux is not None:
        for task in list(aux.keys()):
            blk = aux[task]
            if task in want or not isinstance(blk, dict):
                continue
            if blk.get("provider") not in (None, "", "auto") or blk.get("model"):
                for k in ("provider", "model", "fallback_chain", "reasoning_effort"):
                    _pop(blk, k, ch, f"auxiliary.{task}")
                if isinstance(blk.get("extra_body"), dict):
                    _pop(blk["extra_body"], "provider", ch, f"auxiliary.{task}.extra_body")
                    if not blk["extra_body"]:
                        del blk["extra_body"]
                if not blk:
                    del aux[task]; ch.append(f"auxiliary.{task}: (removed — back to Hermes' auto routing)")

    # 5. OpenRouter routing: profile default (flat) + a per-MODEL pin for every OpenRouter model this
    #    agent can land on. Per-model entries ride with the model into fallbacks, subagents and aux
    #    calls, and each carries data_collection: deny (a pinned child resets the flat filters).
    pr = _map(d, "provider_routing")
    rt = A.get("routing") or {}
    for k in ROUTING_KEYS:
        if k in rt and rt[k] not in (None, [], ""):
            _set(pr, k, copy.deepcopy(rt[k]), ch, "provider_routing")
        else:
            _pop(pr, k, ch, "provider_routing")
    _set(pr, "data_collection", "deny", ch, "provider_routing")
    per = CommentedMap()
    for alias in used:
        mm = M[alias]
        if mm["provider"] != "openrouter":
            continue
        e = CommentedMap()
        for k in HOST_KEYS:
            v = (mm.get("hosts") or {}).get(k)
            if v not in (None, [], ""):
                e[k] = copy.deepcopy(v)
        if (mm.get("hosts") or {}).get("only"):
            e.setdefault("ignore", [])  # `only` is the whole host set; a profile ignore list was written for other models
        e["data_collection"] = "deny"
        per[mm["id"]] = e
    if per:
        _set(pr, "models", per, ch, "provider_routing")
    else:
        _pop(pr, "models", ch, "provider_routing")

    # 6. reasoning: agent default + per-model pins (re-resolved by Hermes on every fallback swap)
    ag = _map(d, "agent")
    if A.get("reasoning") is not None:
        _set(ag, "reasoning_effort", A["reasoning"], ch, "agent")
    else:
        _pop(ag, "reasoning_effort", ch, "agent")
    ro = CommentedMap()
    for alias in used:
        if M[alias].get("reasoning"):
            ro[M[alias]["id"]] = M[alias]["reasoning"]
    if ro:
        _set(ag, "reasoning_overrides", ro, ch, "agent")
    else:
        _pop(ag, "reasoning_overrides", ch, "agent")

    # 7. the ModelArk provider block (context lengths come from the registry)
    ark = [M[a] for a in M if M[a]["provider"] == "modelark"]
    if ark:
        provs = _map(d, "providers")
        blk = _map(provs, "modelark")
        _set(blk, "name", blk.get("name") or "ModelArk", ch, "providers.modelark")
        _set(blk, "base_url", ARK_CODING, ch, "providers.modelark")
        _set(blk, "key_env", ARK_KEY_ENV, ch, "providers.modelark")
        if not blk.get("default_model"):
            _set(blk, "default_model", ark[0]["id"], ch, "providers.modelark")
        mods = CommentedMap()
        for mm in ark:
            e = CommentedMap(); e["context_length"] = int(mm.get("context") or 1048576); mods[mm["id"]] = e
        _set(blk, "models", mods, ch, "providers.modelark")

    # 8. inert keys that mislead a reader (nothing in Hermes reads a top-level `vision:`)
    v = d.get("vision")
    if isinstance(v, dict) and set(v.keys()) <= {"provider", "model", "base_url", "extra_body"}:
        _pop(d, "vision", ch, "")
    return ch


# ── effective view: what Hermes will actually do with a config ───────────────────────────────
def _fb_chain(cfg: dict) -> List[Tuple[str, str]]:
    """Mirror of hermes_cli.fallback_config.get_fallback_chain (fallback_providers, then legacy)."""
    out, seen = [], set()
    for key in ("fallback_providers", "fallback_model"):
        raw = cfg.get(key)
        for e in ([raw] if isinstance(raw, dict) else raw if isinstance(raw, list) else []):
            if isinstance(e, dict) and e.get("provider") and e.get("model"):
                ident = (str(e["provider"]).lower(), str(e["model"]).lower())
                if ident not in seen:
                    seen.add(ident); out.append((str(e["provider"]), str(e["model"])))
    return out


def _routing_for(cfg: dict, model_id: str) -> dict:
    pr = cfg.get("provider_routing") or {}
    flat = {k: pr.get(k) for k in ("only", "ignore", "order", "sort", "require_parameters", "data_collection")}
    per = (pr.get("models") or {}).get(model_id) or {}
    merged = {**flat, **{k: v for k, v in per.items() if k in flat}}
    return {k: v for k, v in merged.items() if v}


def effective(cfg: dict) -> dict:
    """Per slot: [(provider, model)] as Hermes resolves it, + OpenRouter prefs and reasoning per model."""
    m = cfg.get("model") or {}
    main = [(str(m.get("provider") or ""), str(m.get("default") or m.get("model") or ""))] + _fb_chain(cfg)
    dl = cfg.get("delegation") or {}
    if dl.get("model"):
        sub = [(str(dl.get("provider") or ""), str(dl["model"]))] + _fb_chain({"fallback_providers": dl.get("fallback_providers")})
    else:
        sub = None  # inherits the parent's route and chain
    cr = cfg.get("cron") or {}
    if cr.get("model"):
        own = cr.get("fallback_providers")
        cron = [(str(cr.get("model_provider") or ""), str(cr["model"]))] + (
            _fb_chain({"fallback_providers": own}) if own is not None else _fb_chain(cfg))
    else:
        cron = None
    aux = {}
    for task, blk in (cfg.get("auxiliary") or {}).items():
        if isinstance(blk, dict) and blk.get("model") and blk.get("provider") not in (None, "", "auto"):
            aux[task] = {"chain": [(str(blk["provider"]), str(blk["model"]))] + _fb_chain({"fallback_providers": blk.get("fallback_chain")}),
                         "reasoning": blk.get("reasoning_effort")}
    ag = cfg.get("agent") or {}
    ids = {mid for prov, mid in main + (sub or []) + (cron or []) + [x for a in aux.values() for x in a["chain"]] if prov == "openrouter"}
    return {"main": main, "subagents": sub, "cron": cron, "aux": aux,
            "routing": {mid: _routing_for(cfg, mid) for mid in sorted(ids)},
            "reasoning": ag.get("reasoning_effort"), "reasoning_overrides": dict(ag.get("reasoning_overrides") or {})}


def expected_effective(doc: dict, p: str) -> dict:
    """What effective(config) must equal after a compile — the verification target."""
    M = doc["models"]; A = doc["agents"][p]
    route = lambda ch: [(M[a]["provider"], M[a]["id"]) for a in ch]
    main = route(chain_of(A["main"]))
    sub = route(chain_of(A["subagents"])) if A.get("subagents") else None
    cron = None
    if A.get("cron"):
        c = chain_of(A["cron"]); cron = route(c)
    aux = {t: {"chain": route(chain_of(s)), "reasoning": reasoning_of(s)} for t, s in (A.get("aux") or {}).items()}
    return {"main": main, "subagents": sub, "cron": cron, "aux": aux}


def verify_profile(doc: dict, p: str, cfg: dict) -> List[str]:
    """Problems between what the doc says and what Hermes will do with *cfg* ([] = verified)."""
    bad = []
    eff = effective(cfg); exp = expected_effective(doc, p)
    for k in ("main", "subagents", "cron"):
        if eff[k] != exp[k]:
            bad.append(f"{p}.{k}: config resolves {eff[k]} but models.yaml says {exp[k]}")
    for t, spec in exp["aux"].items():
        got = eff["aux"].get(t)
        if not got or got["chain"] != spec["chain"] or got["reasoning"] != spec["reasoning"]:
            bad.append(f"{p}.aux.{t}: config resolves {got} but models.yaml says {spec}")
    for t in eff["aux"]:
        if t not in exp["aux"]:
            bad.append(f"{p}.aux.{t}: set in config but not in models.yaml")
    M = doc["models"]
    for alias in models_used(doc, p):
        mm = M[alias]
        if mm["provider"] == "openrouter":
            r = eff["routing"].get(mm["id"], {})
            if r.get("data_collection") != "deny":
                bad.append(f"{p}: {mm['id']} would be served WITHOUT data_collection: deny")
            for k in ("only", "order"):
                want = (mm.get("hosts") or {}).get(k)
                if want and r.get(k) != want:
                    bad.append(f"{p}: {mm['id']} {k} resolves {r.get(k)} not {want}")
        if mm.get("reasoning") and str(eff["reasoning_overrides"].get(mm["id"])) != str(mm["reasoning"]):
            bad.append(f"{p}: {mm['id']} reasoning pin missing (wants {mm['reasoning']})")
    return bad


# ── import: live configs -> doc (drift detection + the first write) ──────────────────────────
def import_configs(root: Optional[Path] = None, base: Optional[dict] = None) -> dict:
    """Describe the live configs as a models.yaml document. Registry entries come from *base*
    (models not in it are added with the facts the configs carry)."""
    doc = copy.deepcopy(base) if base else {"version": 1, "policy": {"data_collection": "deny", "min_host_uptime": 95}, "models": {}, "agents": {}}
    M = doc.setdefault("models", {})
    by_id = {(m["provider"], m["id"]): a for a, m in M.items()}

    def alias_for(provider: str, mid: str) -> str:
        key = (provider, mid)
        if key in by_id:
            return by_id[key]
        a = re.sub(r"[^a-z0-9]+", "-", mid.split("/")[-1].lower()).strip("-")
        while a in M:
            a += "-x"
        M[a] = {"id": mid, "provider": provider, "vendor": mid.split("/")[0] if "/" in mid else "unknown",
                "label": mid, "billing": "subscription" if provider == "modelark" else "metered"}
        by_id[key] = a
        return a

    agents = doc.setdefault("agents", {})
    for p in PROFILES:
        cfg = _load_plain(cfg_path(p, root))
        eff = effective(cfg)
        prev = agents.get(p) or {}
        a = {k: prev[k] for k in ("name", "role", "locked") if k in prev}
        a["main"] = [alias_for(*x) for x in eff["main"]]
        a["subagents"] = [alias_for(*x) for x in eff["subagents"]] if eff["subagents"] else None
        a["cron"] = [alias_for(*x) for x in eff["cron"]] if eff["cron"] else None
        aux = {}
        for t, s in eff["aux"].items():
            ch = [alias_for(*x) for x in s["chain"]]
            aux[t] = {"chain": ch, "reasoning": s["reasoning"]} if s["reasoning"] else ch
        a["aux"] = aux
        a["reasoning"] = eff["reasoning"]
        pr = cfg.get("provider_routing") or {}
        a["routing"] = {k: pr[k] for k in ROUTING_KEYS if pr.get(k) not in (None, [], "")}
        agents[p] = a
    return doc


# ── SOUL blocks: what each agent reads about its own models, generated from the doc ─────────
SOUL_LINE = "<!-- fleet-models:model -->"
TABLE_BEGIN = "<!-- fleet-models:table -->"
TABLE_END = "<!-- /fleet-models:table -->"


def soul_path(p: str, root: Optional[Path] = None) -> Path:
    r = root or fleet_root()
    return r / "SOUL.md" if p == "root" else r / "profiles" / p / "SOUL.md"


def _short(doc: dict, alias: str) -> str:
    m = doc["models"].get(alias) or {}
    return str(m.get("short") or alias)


def _chain_text(doc: dict, spec) -> str:
    return " → ".join(_short(doc, a) for a in chain_of(spec))


def soul_model_line(doc: dict, p: str) -> str:
    a = doc["agents"][p]
    parts = [f"main {_chain_text(doc, a['main'])}"]
    if a.get("subagents"):
        parts.append(f"subagents {_chain_text(doc, a['subagents'])}")
    if a.get("cron"):
        parts.append(f"cron {_chain_text(doc, a['cron'])}")
    aux = a.get("aux") or {}
    if "vision" in aux:
        parts.append(f"vision (the `vision` toolset) {_chain_text(doc, aux['vision'])}")
    helpers = {t: _chain_text(doc, s) for t, s in aux.items() if t != "vision"}
    if helpers:
        same = set(helpers.values())
        parts.append(("helpers " + next(iter(same))) if len(same) == 1 else
                     "helpers " + "; ".join(f"{t} {c}" for t, c in helpers.items()))
    return (f"- **Model:** {' · '.join(parts)}. ModelArk is the flat Coding Plan subscription (recorded as $0 "
            f"\"modelark subscription\"); OpenRouter calls are billed and always carry data_collection deny. "
            f"Source: `~/.hermes/fleet/models.yaml` — changed only on the dashboard's Models tab. {SOUL_LINE}")


def soul_table(doc: dict) -> str:
    rows = [TABLE_BEGIN,
            "*Generated from `~/.hermes/fleet/models.yaml` — the single model file. Change models on the dashboard's "
            "Models tab (or `plugins/fleet-models/core.py`), never by hand in config.yaml: a hand edit is drift and "
            "the next apply overwrites it.*", "",
            "| Profile | Main loop | Subagents | Vision | Why |", "|---|---|---|---|---|"]
    for p in PROFILES:
        a = doc["agents"][p]
        name = "`default` (you)" if p == "root" else f"`{p}`"
        aux = a.get("aux") or {}
        rows.append(f"| {name} | {_chain_text(doc, a['main'])} | {_chain_text(doc, a['subagents']) if a.get('subagents') else 'inherits main'} | "
                    f"{_chain_text(doc, aux['vision']) if 'vision' in aux else 'main model'} | {a.get('why') or ''} |")
    root_aux = (doc["agents"].get("root") or {}).get("aux") or {}
    helpers = sorted({f"{t}: {_chain_text(doc, sp)}" for t, sp in root_aux.items() if t != "vision"})
    rows += ["", "Your helpers — " + "; ".join(helpers) + "." if helpers else ""]
    for alias, m in doc["models"].items():
        bits = []
        if (m.get("hosts") or {}).get("order") or (m.get("hosts") or {}).get("only"):
            bits.append("hosts " + " → ".join((m["hosts"].get("order") or m["hosts"].get("only"))))
        if m.get("reasoning"):
            bits.append(f"reasoning {m['reasoning']}")
        if bits and any(alias in chain_of(sp) for a in doc["agents"].values()
                        for sp in [a.get("main"), a.get("subagents"), a.get("cron"), *(a.get("aux") or {}).values()]):
            rows.append(f"`{_short(doc, alias)}` = `{m['id']}` on OpenRouter — {'; '.join(bits)}.  ")
    rows.append(TABLE_END)
    return "\n".join(rows)


def write_souls(doc: dict, root: Optional[Path] = None) -> List[str]:
    """Refresh the generated SOUL blocks. Only text between/on the markers is touched; a SOUL without a
    marker is left alone. Returns the SOULs changed."""
    root = root or fleet_root()
    changed = []
    for p in PROFILES:
        f = soul_path(p, root)
        if not f.exists():
            continue
        txt = f.read_text()
        new = txt
        if TABLE_BEGIN in new and TABLE_END in new:
            a, rest = new.split(TABLE_BEGIN, 1)
            _, b = rest.split(TABLE_END, 1)
            new = a + soul_table(doc) + b
        if SOUL_LINE in new:
            new = "\n".join(soul_model_line(doc, p) if SOUL_LINE in ln else ln for ln in new.split("\n"))
        if new != txt:
            _atomic_write(f, new)
            changed.append(p)
    return changed


# ── plan / apply / verify / history / revert ─────────────────────────────────────────────────
def _lock(root: Path):
    lp = root / "fleet" / ".lock"
    lp.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lp, "a+")
    fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


def plan(doc: dict, root: Optional[Path] = None) -> dict:
    """Compile *doc* against the live configs without writing. {profile: {changes, diff, text}}."""
    root = root or fleet_root()
    out = {}
    for p in PROFILES:
        f = cfg_path(p, root)
        src = f.read_text()
        y, d = _load_rt(src)
        ch = compile_profile(doc, p, d)
        new = _dump_rt(y, d) if ch else src
        diff = "".join(difflib.unified_diff(src.splitlines(True), new.splitlines(True), f"{p}/config.yaml", f"{p}/config.yaml (compiled)", n=2))
        out[p] = {"changes": ch, "diff": diff, "text": new, "src": src}
    return out


def dump_doc(doc: dict) -> str:
    import yaml
    head = ("# WeRoll Hermes fleet — model settings. THE source of truth for every agent's waterfall.\n"
            "# Edit through the dashboard's Models tab (or plugins/fleet-models/core.py); `apply` compiles this\n"
            "# into the nine config.yaml files, verifies them, and keeps a pre-image for revert.\n"
            "# Hand edits here are fine too — run `core.py apply` afterwards. Hand edits to the compiled keys\n"
            "# in config.yaml are drift: the dashboard flags them and the next apply overwrites them.\n")
    return head + yaml.safe_dump(doc, sort_keys=False, width=110, allow_unicode=True)


def write_doc(doc: dict, root: Optional[Path] = None) -> None:
    """models.yaml + models.json (the same document for stdlib-only readers: cost scripts, pre-flight)."""
    root = root or fleet_root()
    _atomic_write(doc_path(root), dump_doc(doc))
    _atomic_write(doc_path(root).with_suffix(".json"), json.dumps(doc, indent=1, default=str) + "\n")
    _doc_cache.pop(str(doc_path(root)), None)


def _history_dir(root: Path) -> Path:
    return root / "fleet" / "history"


def apply(doc: dict, *, by: str = "cli", summary: str = "", root: Optional[Path] = None,
          base_revision: Optional[int] = None, unlock: Tuple[str, ...] = (), dry_run: bool = False,
          snapshot: Optional[dict] = None) -> dict:
    """validate -> snapshot -> write models.yaml + configs -> verify (restore the snapshot on failure)."""
    root = root or fleet_root()
    fh = _lock(root)
    try:
        dp = doc_path(root)
        prev = _load_plain(dp) if dp.exists() else None
        if base_revision is not None and prev is not None and int(prev.get("revision") or 0) != int(base_revision):
            return {"ok": False, "errors": [f"models.yaml moved on (revision {prev.get('revision')} vs your {base_revision}) — reload and redo the edit"]}
        errors, warnings = validate(doc, previous=prev, unlock=unlock, snapshot=snapshot)
        if errors:
            return {"ok": False, "errors": errors, "warnings": warnings}
        pl = plan(doc, root)
        changed = [p for p in PROFILES if pl[p]["changes"]]
        new_doc = copy.deepcopy(doc)
        new_doc["revision"] = int((prev or {}).get("revision") or 0) + 1
        new_doc["updated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        new_doc["updated_by"] = by
        if dry_run:
            return {"ok": True, "dry_run": True, "changed": changed, "warnings": warnings,
                    "plan": {p: {"changes": pl[p]["changes"], "diff": pl[p]["diff"]} for p in PROFILES}}
        hid = _dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        hd = _history_dir(root) / hid
        hd.mkdir(parents=True)
        if dp.exists():
            shutil.copy2(dp, hd / "models.yaml")
        for p in PROFILES:
            shutil.copy2(cfg_path(p, root), hd / f"{p}.config.yaml")
            if soul_path(p, root).exists():
                shutil.copy2(soul_path(p, root), hd / f"{p}.SOUL.md")
        try:
            for p in changed:
                _atomic_write(cfg_path(p, root), pl[p]["text"])
            write_doc(new_doc, root)
            problems = []
            for p in PROFILES:
                problems += verify_profile(new_doc, p, _load_plain(cfg_path(p, root)))
            if problems:
                raise RuntimeError("verify failed: " + "; ".join(problems[:6]))
            souls = write_souls(new_doc, root)
        except Exception as exc:
            _restore(hd, root)
            return {"ok": False, "errors": [f"{type(exc).__name__}: {exc} — every file restored from {hid}"], "warnings": warnings}
        rec = {"id": hid, "ts": new_doc["updated_at"], "by": by, "revision": new_doc["revision"],
               "summary": summary or f"{len(changed)} config(s) changed", "changed": changed, "souls": souls,
               "changes": {p: pl[p]["changes"] for p in changed}}
        with open(root / "fleet" / "history.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        _doc_cache.pop(str(dp), None)
        return {"ok": True, "id": hid, "revision": new_doc["revision"], "changed": changed, "warnings": warnings,
                "changes": rec["changes"], "souls": souls}
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN); fh.close()


def _restore(hd: Path, root: Path) -> None:
    for p in PROFILES:
        src = hd / f"{p}.config.yaml"
        if src.exists():
            _atomic_write(cfg_path(p, root), src.read_text())
        sp = hd / f"{p}.SOUL.md"
        if sp.exists():
            _atomic_write(soul_path(p, root), sp.read_text())
    if (hd / "models.yaml").exists():
        write_doc(_load_plain(hd / "models.yaml"), root)


def history(root: Optional[Path] = None, limit: int = 50) -> List[dict]:
    f = (root or fleet_root()) / "fleet" / "history.jsonl"
    if not f.exists():
        return []
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return rows[-limit:][::-1]


def revert(hid: str, *, by: str = "cli", root: Optional[Path] = None) -> dict:
    """Put every file back as it was before apply *hid* (itself recorded, so it can be undone)."""
    root = root or fleet_root()
    hd = _history_dir(root) / hid
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", hid) or not hd.is_dir():
        return {"ok": False, "errors": [f"no snapshot {hid}"]}
    fh = _lock(root)
    try:
        nid = _dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        nd = _history_dir(root) / nid
        nd.mkdir(parents=True)
        if doc_path(root).exists():
            shutil.copy2(doc_path(root), nd / "models.yaml")
        for p in PROFILES:
            shutil.copy2(cfg_path(p, root), nd / f"{p}.config.yaml")
            if soul_path(p, root).exists():
                shutil.copy2(soul_path(p, root), nd / f"{p}.SOUL.md")
        _restore(hd, root)
        # keep revision monotonic so a stale dashboard can't write over the revert
        if doc_path(root).exists():
            d = _load_plain(doc_path(root))
            cur = _load_plain(nd / "models.yaml") if (nd / "models.yaml").exists() else {}
            d["revision"] = int(cur.get("revision") or 0) + 1
            d["updated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
            d["updated_by"] = f"{by} (revert of {hid})"
            write_doc(d, root)
        rec = {"id": nid, "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "by": by,
               "summary": f"revert of {hid}", "reverts": hid, "changed": PROFILES}
        with open(root / "fleet" / "history.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        _doc_cache.pop(str(doc_path(root)), None)
        return {"ok": True, "id": nid}
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN); fh.close()


def drift(root: Optional[Path] = None) -> Dict[str, List[str]]:
    """Profiles whose live config no longer matches models.yaml (someone edited config.yaml by hand)."""
    root = root or fleet_root()
    doc = load_doc(root)
    return {p: v for p in PROFILES if (v := verify_profile(doc, p, _load_plain(cfg_path(p, root))))}


# ── CLI ──────────────────────────────────────────────────────────────────────────────────────
def _main(argv: List[str]) -> int:
    cmd = argv[0] if argv else "show"
    root = fleet_root()
    if cmd == "import":
        base = load_doc(root) if doc_path(root).exists() else None
        doc = import_configs(root, base)
        if "--write" in argv:
            write_doc(doc, root); print("wrote", doc_path(root))
        else:
            print(dump_doc(doc))
        return 0
    if cmd == "show":
        doc = load_doc(root)
        for p in PROFILES:
            a = doc["agents"][p]
            print(f"{a.get('name') or p:8} main: {' > '.join(chain_of(a['main']))}")
            for s in ("subagents", "cron"):
                if a.get(s):
                    print(f"{'':8} {s}: {' > '.join(chain_of(a[s]))}")
            for t, s in (a.get("aux") or {}).items():
                print(f"{'':8} aux.{t}: {' > '.join(chain_of(s))}" + (f"  (reasoning {reasoning_of(s)})" if reasoning_of(s) else ""))
        return 0
    if cmd in ("plan", "apply"):
        doc = load_doc(root, fresh=True)
        by = argv[argv.index("--by") + 1] if "--by" in argv else "cli"
        unlock = tuple(x for x in PROFILES if f"--unlock={x}" in argv)
        r = apply(doc, by=by, root=root, dry_run=(cmd == "plan"), unlock=unlock,
                  summary=argv[argv.index("--summary") + 1] if "--summary" in argv else "")
        if cmd == "plan" and r.get("ok"):
            for p in PROFILES:
                if r["plan"][p]["changes"]:
                    print(f"== {p}"); [print("   ", c) for c in r["plan"][p]["changes"]]
            if "--diff" in argv:
                for p in PROFILES:
                    sys.stdout.write(r["plan"][p]["diff"])
            print("changed:", r["changed"] or "nothing")
        else:
            print(json.dumps({k: v for k, v in r.items() if k != "plan"}, indent=1))
        for w in r.get("warnings") or []:
            print("WARN", w)
        return 0 if r.get("ok") else 1
    if cmd == "verify":
        doc = load_doc(root, fresh=True); bad = 0
        for p in PROFILES:
            probs = verify_profile(doc, p, _load_plain(cfg_path(p, root)))
            print(f"{p:8} {'OK' if not probs else 'DRIFT'}"); [print("   ", x) for x in probs]; bad += bool(probs)
        return 1 if bad else 0
    if cmd == "souls":
        doc = load_doc(root, fresh=True)
        if "--write" in argv:
            print("SOULs refreshed:", write_souls(doc, root) or "none changed")
        else:
            print(soul_table(doc)); [print(soul_model_line(doc, p)) for p in PROFILES]
        return 0
    if cmd == "history":
        for h in history(root):
            print(h["id"], h.get("by"), h.get("summary"))
        return 0
    if cmd == "revert":
        r = revert(argv[1], root=root); print(json.dumps(r)); return 0 if r.get("ok") else 1
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
