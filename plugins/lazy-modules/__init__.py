#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lazy-modules — trigger-based module loading for oversized AGENTS.md files.

Background (design validated upstream in #110868, spill trap found by @KeyArgo):

  * An aggregated context file (AGENTS.md et al.) gets truncated twice on the
    way into the prompt: once at load (``context_file_max_chars``) and again by
    the subdirectory-hint layer (hard 32K), with a head/tail middle-drop that
    silently loses whatever sits in the middle.
  * This plugin splits that file into manifest groups (``splitter.py``) and
    injects a group's full text at most ONCE per session via ``pre_llm_call``:
    core applies hook context to the CURRENT turn's user message at API time
    only, stamps the exact bytes into the row's ``api_content`` sidecar, and
    replays them verbatim for later turns — so the group is paid once, rides
    along in history, and the system prompt stays byte-stable (prompt caching
    intact, no core changes).
  * Spill trap: each hook return is passed through
    ``tools.hook_output_spill.spill_if_oversized`` (default max_chars=10,000).
    Oversized text is NOT injected — it becomes a ``[... saved to <path>]``
    pointer. Budgeting therefore happens here, in CHARS, against the live
    ``hooks.output_spill.max_chars`` value: groups that do not fit the per-turn
    budget are DEFERRED to a later turn (never recorded as loaded, never
    silently pointerized). A deferred oversized group is reported by ``probe``.

Everything runs inside try/except: this is a hot-path hook and must never take
down the agent loop.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_STATE_LOCK = threading.Lock()
_LOADED: Dict[str, Dict[str, float]] = {}  # session_id -> {group: ts}
_MAX_SESSIONS = 200
_MAX_GROUPS_PER_TURN = 6
_FALLBACK_SPILL_CAP = 9_500  # core DEFAULT_MAX_CHARS is 10,000
_SPILL_MARGIN = 512
_MANIFEST_CACHE: Optional[dict] = None
_MANIFEST_TS: float = 0.0
_SPILL_CFG_CACHE: Tuple[float, int] = (0.0, _FALLBACK_SPILL_CAP)  # (ts, cap)

_CMD_IMPORT = re.compile(r"(?:^|\s)\$import\s+([\w\-,.]+)", re.IGNORECASE)


def _data_root() -> Path:
    import os

    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / "lazy-modules"


# ------------------------------------------------------------------ manifest


def _manifest(force: bool = False) -> Optional[dict]:
    global _MANIFEST_CACHE, _MANIFEST_TS
    mf = _data_root() / "manifest.json"
    if not mf.exists():
        return None
    try:
        mtime = mf.stat().st_mtime
    except OSError:
        return None
    if force or _MANIFEST_CACHE is None or mtime != _MANIFEST_TS:
        try:
            _MANIFEST_CACHE = json.loads(mf.read_text(encoding="utf-8"))
            _MANIFEST_TS = mtime
        except Exception as exc:  # corrupt manifest -> off, never crash
            logger.warning("lazy-modules: manifest unreadable: %s", exc)
            return None
    return _MANIFEST_CACHE


def _module_text(name: str) -> str:
    p = _data_root() / "modules" / Path(name).name
    if not p.is_file():
        return ""
    return p.read_text(encoding="utf-8", errors="replace").strip()


def _spill_cap() -> int:
    """Live wire budget in CHARS: core's hooks.output_spill.max_chars minus a
    margin (cached 30s; falls back to 9,500 if the config layer is absent)."""
    global _SPILL_CFG_CACHE
    now = time.time()
    if now - _SPILL_CFG_CACHE[0] < 30:
        return _SPILL_CFG_CACHE[1]
    cap = _FALLBACK_SPILL_CAP
    try:
        from tools.hook_output_spill import get_spill_config

        cfg = get_spill_config()
        if not cfg.get("enabled", True):
            cap = 10 * _FALLBACK_SPILL_CAP  # spill disabled -> generous, still bounded
        else:
            cap = max(4_000, int(cfg.get("max_chars", 10_000)) - _SPILL_MARGIN)
    except Exception:
        logger.debug("lazy-modules: spill config unavailable", exc_info=True)
    _SPILL_CFG_CACHE = (now, cap)
    return cap


def _match_groups(text: str, manifest: dict) -> List[str]:
    low = text.lower()
    hit: List[str] = []
    for g in manifest.get("groups", []):
        if g.get("always") or g["name"] in hit:
            continue  # always-groups live in the static slim layer, not on-demand
        for trig in g.get("triggers", []):
            if trig and trig.lower() in low:
                hit.append(g["name"])
                break
    return hit


def _explicit_groups(text: str, manifest: dict) -> List[str]:
    names = {g["name"] for g in manifest.get("groups", [])}
    out: List[str] = []
    for m in _CMD_IMPORT.finditer(text):
        for tok in m.group(1).split(","):
            tok = tok.strip()
            if tok in names and tok not in out:
                out.append(tok)
    return out


def _extract_text(user_message: Any) -> str:
    if isinstance(user_message, str):
        return user_message
    if isinstance(user_message, list):  # multimodal parts
        return " ".join(
            str(p.get("text", "")) for p in user_message if isinstance(p, dict)
        )
    return str(user_message or "")


def _load_group_bodies(
    groups: List[str], manifest: dict, cap: Optional[int] = None
) -> Tuple[str, List[str], int]:
    """Pack whole groups into ONE hook return under the wire budget (chars —
    core spills each hook's merged return, so staying under cap for the joined
    blob is the invariant). Only groups actually present in the blob are
    reported as loaded; the rest defer to a later turn (never recorded, never
    silently dropped)."""
    cap = cap if cap is not None else _spill_cap()
    by_name = {g["name"]: g for g in manifest.get("groups", [])}
    parts: List[str] = []
    loaded: List[str] = []
    total = 0
    for gname in groups:
        g = by_name.get(gname)
        if not g:
            continue
        body = "\n\n".join(filter(None, (_module_text(m) for m in g["modules"])))
        if not body:
            continue
        blob = f"[lazy-modules] loaded group '{gname}':\n{body}"
        n = len(blob)
        sep = 2 if parts else 0
        if total + sep + n > cap:
            continue  # defer; try later groups that may be smaller
        parts.append(blob)
        loaded.append(gname)
        total += sep + n
        if len(loaded) >= _MAX_GROUPS_PER_TURN:
            break
    return "\n\n".join(parts), loaded, total


def _record(session_id: str, groups: List[str], nchars: int, why: str) -> None:
    try:
        with _STATE_LOCK:
            seen = _LOADED.setdefault(session_id, {})
            for g in groups:
                seen[g] = time.time()
            if len(_LOADED) > _MAX_SESSIONS:  # bounded: evict oldest sessions
                oldest = sorted(
                    _LOADED.items(), key=lambda kv: max(kv[1].values() or [0])
                )[:10]
                for k, _ in oldest:
                    _LOADED.pop(k, None)
        log = _data_root() / "injections.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "session": session_id,
                        "groups": groups,
                        "chars": nchars,
                        "via": why,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        logger.debug("lazy-modules: record failed", exc_info=True)


# ------------------------------------------------------------------ hooks


def _pre_llm_call(**kwargs: Any) -> Optional[str]:
    try:
        manifest = _manifest()
        if not manifest:
            return None
        session_id = str(kwargs.get("session_id") or "anon")
        text = _extract_text(kwargs.get("user_message"))
        if not text.strip():
            return None
        already = _LOADED.get(session_id, {})
        groups = _match_groups(text, manifest) + _explicit_groups(text, manifest)
        groups = [g for g in dict.fromkeys(groups) if g not in already]
        if not groups:
            return None
        blob, loaded, n = _load_group_bodies(groups, manifest)
        if not blob:
            return None
        _record(session_id, loaded, n, "pre_llm_call")
        logger.info(
            "lazy-modules: injected %s (%d chars) deferred=%s session=%s",
            loaded,
            n,
            [g for g in groups if g not in loaded],
            session_id[:24],
        )
        return blob
    except Exception:
        logger.debug("lazy-modules: pre_llm_call swallowed", exc_info=True)
        return None


def _section(session_info: Any) -> str:
    try:
        manifest = _manifest()
        if not manifest:
            return ""
        lines = ["## Lazy prompt modules (lazy-modules)"]
        lines.append(
            "The user's context file is split into module groups stored by the "
            "lazy-modules plugin; a group whose trigger keywords appear in the "
            "conversation is injected in full once per session, or load one "
            "explicitly with `$import <group>`."
        )
        cap = _spill_cap()
        for g in manifest.get("groups", []):
            tag = "always" if g.get("always") else ", ".join(g.get("triggers", [])[:6])
            note = ""
            if not g.get("always"):
                body = "\n\n".join(
                    filter(None, (_module_text(m) for m in g["modules"]))
                )
                if len(body) > cap:
                    note = "  [over wire budget — split it in groups.json]"
            lines.append(f"- {g['name']}: {len(g['modules'])} module(s) [{tag}]{note}")
        return "\n".join(lines)[:3_600]
    except Exception:
        return ""


# ------------------------------------------------------------------ CLI + selftest


def _cli_status(ns: Any) -> int:
    mf = _manifest(force=True) or {}
    print(
        f"manifest: {mf.get('version', '-')} source={mf.get('source', '-')} "
        f"groups={len(mf.get('groups', []))}"
    )
    for g in mf.get("groups", []):
        size = sum(len(_module_text(m)) for m in g["modules"])
        print(
            f"  {g['name']:<20} {'always' if g['always'] else 'lazy':<7} "
            f"{size:>7} chars  triggers={', '.join(g['triggers'][:5]) or '-'}"
        )
    try:
        rows = (
            (_data_root() / "injections.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[-8:]
        )
    except OSError:
        rows = []
    print("recent injections:")
    for r in rows or ["  (none)"]:
        print(" ", r[:160])
    return 0


def _cli_split(ns: Any) -> int:
    from . import splitter

    rc = splitter.main(["--source", ns.source, "--root", str(_data_root())])
    if rc == 0:
        print(
            "next: hermes lazy-modules probe  (fix any FAIL wire-budget groups by "
            "editing groups.json — rerun split after writing it — then swap "
            "<source>.slim in for the original context file and restart)"
        )
    return rc


def _cli_probe(ns: Any) -> int:
    """Offline probe: trigger matching, dedupe, and the wire-budget invariant —
    every lazy group must fit the spill budget ALONE, or it starves silently."""
    mf = _manifest(force=True)
    if not mf:
        print("FAIL: no manifest — run: hermes lazy-modules split --source <AGENTS.md>")
        return 1
    rc = 0
    cap = _spill_cap()
    for g in mf.get("groups", []):
        if g.get("always"):
            continue
        body = "\n\n".join(filter(None, (_module_text(m) for m in g["modules"])))
        blob, loaded, _n = _load_group_bodies([g["name"]], mf, cap=cap)
        size = len(body)
        if g["name"] not in loaded or size > cap:
            print(
                f"FAIL wire-budget group '{g['name']}' = {size} chars (cap {cap}) "
                f"— split it smaller in groups.json"
            )
            rc = 1
    if rc == 0:
        print(
            f"OK  wire-budget: every lazy group fits under {cap} chars (wire, not just match)"
        )
    return rc


def _setup(sub: Any) -> None:
    sp = sub.add_subparsers(dest="action")
    p_split = sp.add_parser(
        "split", help="split an aggregated context file into modules"
    )
    p_split.add_argument(
        "--source", required=True, help="path to AGENTS.md (or similar) to split"
    )
    sp.add_parser("status", help="groups + sizes + recent injections")
    sp.add_parser("probe", help="offline wire-budget + invariant selftest")
    sp.add_parser("reload", help="drop the manifest cache (next match re-reads)")


def _handler(ns: Any) -> int:
    action = getattr(ns, "action", None)
    if action == "split":
        return _cli_split(ns)
    if action == "probe":
        return _cli_probe(ns)
    if action == "reload":
        _manifest(force=True)
        print("manifest cache dropped")
        return 0
    return _cli_status(ns)


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", _pre_llm_call)
    try:
        ctx.register_system_prompt_section(
            "lazy-modules.index", _section, position="after_memory", max_chars=4_000
        )
    except Exception:  # older core without the section API -> hooks still work
        logger.debug("lazy-modules: no system_prompt_section API", exc_info=True)
    ctx.register_cli_command(
        "lazy-modules",
        "Trigger-based module loading for oversized AGENTS.md",
        _setup,
        _handler,
        description="Split an aggregated context file into groups injected on demand, "
        "cache-safely, within the hook spill budget.",
    )
