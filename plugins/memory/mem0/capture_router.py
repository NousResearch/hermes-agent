"""Arm-B two-pass capture router for the Track A-lite drain worker (Phase 2.5).

FLAG-GATED, default OFF (`mem0_capture_router.enabled` in mem0.json). When OFF the drain worker's
behavior is byte-identical to today — this module is never invoked.

When ON, the router runs ADDITIVELY on top of the mem0 write path:

  drain_once():
    router.route_turn(stage=False)       <- best-effort preflight, never breaks the turn: two DEDICATED
                                            extraction passes run CONCURRENTLY (armB-prefs + armB-world),
                                            then deterministic correction signals are computed.
    self._add(messages, kwargs)          <- mem0 server-side extraction writes preference/ops_state facts.
                                            Normal turns still use the certified salience gate. Correction
                                            turns (marker OR contradiction signal) omit that prompt so a
                                            below-threshold correction is degraded-not-lost.
    router.stage_route_result(...)       <- after mem0 write+scrub is proven clean, world_entity/event facts
                                            are DEDUPED against the prefs-pass output (the benchmark's leak
                                            fix), then written as STAGED markdown to the staging dir with
                                            frontmatter. The router does NOT write to mem0.

STAGING (the Phase 2.5 gate): while `staging_mode` is true (default), world/event facts are written to
~/.hermes/state/capture-router-staged/<date>/<turn_id>.md — NOT to mem0, NOT to the brain repo/inbox.
Flipping `staging_mode` false (a config flip, not a code change) redirects those same writes to the
gbrain capture inbox (~/gbrain/brain/inbox) which the nightly sync ingests. Go-live is a knob, not a diff.

The benchmark (benchmark-report.md) chose Arm B: two dedicated passes beat one merged prompt on the
world domain (+16 F1). Its wiring notes: run the two passes CONCURRENTLY, and dedup the world pass
against the prefs pass (B's world pass leaks user-ops junk as low-value world candidates).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# The four capture classes plus the "nothing worth capturing" sentinel.
PREFS_CLASSES = ("preference", "ops_state")
WORLD_CLASSES = ("world_entity", "event")
ALL_CLASSES = PREFS_CLASSES + WORLD_CLASSES + ("none",)

# RC1 correction-bypass markers are DATA, not hidden control flow. Operators can override this
# whole list via mem0_capture_router.correction_markers in mem0.json; no env vars are introduced.
# Patterns are intentionally deterministic and local: marker false-negatives degrade to the normal
# salience gate (not lost), while positive matches bypass that gate in the drain worker.
DEFAULT_CORRECTION_MARKERS = (
    r"\bno\s*[,!:;-]",
    r"\bactually\b",
    r"\bwrong\b",
    r"\bi\s+corrected\b",
    r"\b(?:correction|correcting)\b",
    r"\b(?:it\s+is|it'?s|that\s+is|that's)\s+.+?\s+not\s+.+",
    r"\bexplicit\s+fact\s+edit\b",
)

_DEFAULT_STAGING_DIR = "~/.hermes/state/capture-router-staged"
_DEFAULT_BRAIN_INBOX = "~/gbrain/brain/inbox"

# Prompt assets live alongside the plugin (copied from the benchmark harness so the live wiring does
# not depend on a path under ~/.hermes/plans, which is not shipped with the plugin).
_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "assets")
_PREFS_PROMPT_FILE = "capture_router_armB_prefs.md"
_WORLD_PROMPT_FILE = "capture_router_armB_world.md"


# ---------------------------------------------------------------------------
# Candidate parsing (shared shape with bench/run_arms.py)
# ---------------------------------------------------------------------------

def parse_candidates(text: str) -> Optional[List[Dict[str, Any]]]:
    """Best-effort parse of a model extraction response into a candidate list, or None if unparseable.
    Mirrors bench/run_arms.py so the live path and the benchmark agree on output shape."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    try:
        d = json.loads(t)
    except Exception:
        try:
            d = json.loads(t[t.index("{"):t.rindex("}") + 1])
        except Exception:
            return None
    c = d.get("candidates")
    if not isinstance(c, list):
        return None
    out: List[Dict[str, Any]] = []
    for x in c:
        if isinstance(x, dict) and x.get("content"):
            out.append({
                "content": str(x["content"]),
                "class": str(x.get("class", "")).strip(),
                "confidence": x.get("confidence"),
            })
    return out


# ---------------------------------------------------------------------------
# Dedup (the benchmark's leak fix)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_COMPOUND_RE = re.compile(r"\b[a-z0-9]+(?:[._-][a-z0-9]+)+\b", re.IGNORECASE)
_PORT_RE = re.compile(r"\b(?:port|tcp|udp)\s*[:#-]?\s*(\d{2,5})\b", re.IGNORECASE)
_BOOL_RE = re.compile(r"\b(?:enabled|disabled|true|false)\b", re.IGNORECASE)
_FACT_STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "but", "for", "from", "has", "have",
    "into", "its", "not", "now", "off", "the", "then", "this", "that", "their", "there",
    "user", "uses", "using", "was", "were", "with", "without", "runs", "run", "is", "at",
    "to", "of", "in", "on", "ip", "address",
}


def _tokens(text: str) -> set:
    return set(_WORD_RE.findall((text or "").lower()))


def _normalise_correction_markers(markers: Optional[Any]) -> List[str]:
    """Return the configured regex marker list. None means defaults; [] intentionally disables
    marker matches for tests/operators who want contradiction-only correction detection."""
    if markers is None:
        return list(DEFAULT_CORRECTION_MARKERS)
    if isinstance(markers, str):
        return [markers] if markers.strip() else []
    if isinstance(markers, (list, tuple)):
        return [str(m) for m in markers if str(m).strip()]
    return list(DEFAULT_CORRECTION_MARKERS)


def correction_marker_match(user: str, assistant: str = "",
                            markers: Optional[Any] = None) -> Optional[str]:
    """Return the matched correction marker regex, or None. Bad operator-supplied regexes degrade
    to literal substring checks instead of disabling capture routing."""
    text = f"{user or ''}\n{assistant or ''}"
    text_l = text.lower()
    for marker in _normalise_correction_markers(markers):
        try:
            if re.search(marker, text, re.IGNORECASE | re.DOTALL):
                return marker
        except re.error:
            if marker.lower() in text_l:
                return marker
    return None


def _anchor_tokens(text: str) -> set:
    """Meaningful tokens used only to decide whether two facts talk about the same subject."""
    lowered = _IPV4_RE.sub(" ", (text or "").lower())
    compounds = set(_COMPOUND_RE.findall(lowered))
    words = {
        w for w in _WORD_RE.findall(lowered)
        if len(w) > 2 and not w.isdigit() and w not in _FACT_STOPWORDS
    }
    return compounds | words


def _fact_value_sets(text: str) -> Dict[str, set]:
    lowered = text or ""
    return {
        "ipv4": set(_IPV4_RE.findall(lowered)),
        "port": set(_PORT_RE.findall(lowered)),
        "bool": {m.group(0).lower() for m in _BOOL_RE.finditer(lowered)},
    }


def facts_contradict(new_fact: str, existing_fact: str) -> bool:
    """Deterministic, conservative contradiction detector for router dedup.

    It only fires when facts share a subject anchor AND carry conflicting exact values (IP/port/boolean).
    Misses degrade to the normal salience gate; this avoids pretending to solve open-ended semantics.
    """
    shared = _anchor_tokens(new_fact) & _anchor_tokens(existing_fact)
    shared_compound = any(("-" in t or "." in t or "_" in t) for t in shared)
    if not shared_compound and len(shared) < 2:
        return False
    new_values = _fact_value_sets(new_fact)
    old_values = _fact_value_sets(existing_fact)
    for key in ("ipv4", "port", "bool"):
        if new_values[key] and old_values[key] and new_values[key].isdisjoint(old_values[key]):
            return True
    return False


def _existing_fact_texts(rows: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        if isinstance(row, str) and row.strip():
            out.append(row)
        elif isinstance(row, dict):
            text = row.get("memory") or row.get("content") or row.get("text") or row.get("title") or row.get("file")
            if text:
                out.append(str(text))
    return out


def dedup_world_against_prefs(
    world_cands: List[Dict[str, Any]],
    prefs_cands: List[Dict[str, Any]],
    *,
    overlap_threshold: float = 0.6,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Drop world-pass candidates that are really the SAME fact as a prefs-pass candidate — B's world
    pass leaks the user's own config/ops as low-value world_entity/event candidates (benchmark
    secondary finding). Deterministic Jaccard-style token overlap: a world candidate whose token set
    overlaps a prefs candidate at >= threshold (relative to the smaller set) is a duplicate.

    Returns (kept, dropped). Order-preserving and side-effect free (pure) so it is trivially testable.
    """
    prefs_token_sets = [_tokens(c.get("content", "")) for c in prefs_cands]
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for wc in world_cands:
        wt = _tokens(wc.get("content", ""))
        is_dup = False
        if wt:
            for pt in prefs_token_sets:
                if not pt:
                    continue
                inter = len(wt & pt)
                denom = min(len(wt), len(pt))
                if denom and (inter / denom) >= overlap_threshold:
                    is_dup = True
                    break
        (dropped if is_dup else kept).append(wc)
    return kept, dropped


# ---------------------------------------------------------------------------
# Two-pass extraction with primary/fallback provider
# ---------------------------------------------------------------------------

class BridgeExtractor:
    """Runs one extraction pass against codex-bridge (PRIMARY); on ANY error/timeout falls back to
    gemini-bridge. Both are OpenAI-compatible /v1/chat/completions endpoints behind a bearer secret.

    The secret is resolved the same way the bridges' own launchers do — `op read` from 1Password
    (fleet service-account token), no new env var, no secret on disk. A caller may inject an
    `auth_fn`/`http_fn` for tests so no network or 1Password access is needed.
    """

    def __init__(
        self,
        *,
        primary_url: str = "http://127.0.0.1:18812/v1/chat/completions",
        fallback_url: str = "http://192.168.1.216:18813/v1/chat/completions",
        model: str = "gpt-5.4-mini",
        fallback_model: Optional[str] = None,
        primary_secret_ref: str = "op://Engineering/codex-bridge/secret",
        fallback_secret_ref: str = "op://Engineering/gemini-bridge/secret",
        timeout_s: float = 180.0,
        http_fn: Optional[Callable[[str, bytes, Dict[str, str], float], str]] = None,
        auth_fn: Optional[Callable[[str], str]] = None,
    ):
        self._primary_url = primary_url
        self._fallback_url = fallback_url
        self._model = model
        # gemini-bridge advertises different model ids; a caller can pin one. Default: let the bridge
        # pick its default model by omitting an id it doesn't know when the passthrough model is unknown.
        self._fallback_model = fallback_model or model
        self._primary_ref = primary_secret_ref
        self._fallback_ref = fallback_secret_ref
        self._timeout_s = timeout_s
        self._http = http_fn or self._default_http
        self._auth = auth_fn or self._op_read
        # ref -> (secret, fetched_at). TTL'd so a rotated 1Password token recovers without a
        # process restart (Greptile #250 P2); per-ref locks so concurrent first-turn passes
        # don't spawn duplicate `op read` subprocesses (Greptile #250 P2).
        self._secret_cache: Dict[str, Tuple[str, float]] = {}
        self._secret_ttl_s = 3600.0
        self._secret_locks: Dict[str, threading.Lock] = {}
        self._secret_locks_guard = threading.Lock()

    # -- provider plumbing --------------------------------------------------
    @staticmethod
    def _op_read(ref: str) -> str:
        import subprocess
        try:
            out = subprocess.run(
                ["op", "read", ref],
                capture_output=True,
                text=True,
                timeout=20,
                stdin=subprocess.DEVNULL,
            )
            if out.returncode == 0:
                return out.stdout.strip()
            logger.warning("capture-router: op read %s failed rc=%s", ref, out.returncode)
        except Exception as e:
            logger.warning("capture-router: op read %s error: %s", ref, e)
        return ""

    def _secret(self, ref: str) -> str:
        now = time.time()
        hit = self._secret_cache.get(ref)
        if hit is not None and (now - hit[1]) < self._secret_ttl_s:
            return hit[0]
        with self._secret_locks_guard:
            lock = self._secret_locks.setdefault(ref, threading.Lock())
        with lock:
            # double-check under the lock — the other pass may have minted while we waited
            hit = self._secret_cache.get(ref)
            if hit is not None and (time.time() - hit[1]) < self._secret_ttl_s:
                return hit[0]
            secret = self._auth(ref) or ""
            if secret:
                self._secret_cache[ref] = (secret, time.time())
            else:
                # failed fetch: cache briefly (60s) so a down `op` doesn't stampede,
                # but recover quickly once it's back
                self._secret_cache[ref] = ("", time.time() - self._secret_ttl_s + 60.0)
            return secret

    def invalidate_secret(self, ref: str) -> None:
        """Drop a cached secret (called on auth-shaped errors so rotation heals mid-process)."""
        self._secret_cache.pop(ref, None)

    @staticmethod
    def _default_http(url: str, body: bytes, headers: Dict[str, str], timeout: float) -> str:
        req = urllib.request.Request(url, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8")

    def _call(self, url: str, secret_ref: str, model: str, system_prompt: str,
              user: str, assistant: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
        user_content = f"USER MESSAGE:\n{user}\n\nASSISTANT REPLY:\n{assistant}"
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }).encode("utf-8")
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self._secret(secret_ref)}"}
        t0 = time.time()
        raw = self._http(url, body, headers, self._timeout_s)
        latency = time.time() - t0
        resp = json.loads(raw)
        text = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {}) or {}
        cands = parse_candidates(text)
        if cands is None:
            raise ValueError(f"unparseable extraction output: {text[:200]}")
        return cands, usage, latency

    def _call_with_auth_retry(self, url: str, secret_ref: str, model: str, system_prompt: str,
                              user: str, assistant: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
        """_call, but on an auth-shaped failure (401/403) drop the cached secret and retry once —
        so a rotated 1Password token heals mid-process instead of failing until restart."""
        try:
            return self._call(url, secret_ref, model, system_prompt, user, assistant)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                logger.warning("capture-router: auth-shaped %s from %s — refreshing secret and retrying",
                               e.code, url)
                self.invalidate_secret(secret_ref)
                return self._call(url, secret_ref, model, system_prompt, user, assistant)
            raise

    def extract(self, system_prompt: str, user: str, assistant: str) -> Dict[str, Any]:
        """One pass. Returns {candidates, usage, latency, provider} or {error, ...}. codex PRIMARY,
        gemini FALLBACK on any exception/timeout. Never raises (fail-soft — a pass failure yields no
        candidates rather than breaking the turn)."""
        try:
            cands, usage, latency = self._call_with_auth_retry(
                self._primary_url, self._primary_ref, self._model, system_prompt, user, assistant)
            return {"candidates": cands, "usage": usage, "latency": latency, "provider": "codex-bridge"}
        except Exception as primary_err:
            logger.warning("capture-router: primary (codex-bridge) pass failed, trying fallback: %s",
                           primary_err)
            try:
                cands, usage, latency = self._call_with_auth_retry(
                    self._fallback_url, self._fallback_ref, self._fallback_model,
                    system_prompt, user, assistant)
                return {"candidates": cands, "usage": usage, "latency": latency,
                        "provider": "gemini-bridge", "primary_error": str(primary_err)[:200]}
            except Exception as fallback_err:
                logger.warning("capture-router: fallback (gemini-bridge) pass ALSO failed: %s",
                               fallback_err)
                return {"error": f"primary={primary_err}; fallback={fallback_err}",
                        "candidates": [], "usage": {}, "latency": 0.0, "provider": "none"}


# ---------------------------------------------------------------------------
# The router
# ---------------------------------------------------------------------------

def _load_prompt(name: str) -> str:
    path = os.path.join(_PROMPT_DIR, name)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class CaptureRouter:
    """Deterministic class router around the two-pass extractor. Pure routing logic + staged writes;
    the extractor is injected so tests never touch the network."""

    def __init__(
        self,
        *,
        extractor: Optional[BridgeExtractor] = None,
        prefs_prompt: Optional[str] = None,
        world_prompt: Optional[str] = None,
        staging_dir: str = _DEFAULT_STAGING_DIR,
        brain_inbox_dir: str = _DEFAULT_BRAIN_INBOX,
        staging_mode: bool = True,
        confidence_floor: float = 0.0,
        now_fn: Optional[Callable[[], datetime]] = None,
        write_fn: Optional[Callable[[str, str], None]] = None,
        correction_markers: Optional[Any] = None,
        existing_fact_lookup_fn: Optional[Callable[[str], Any]] = None,
    ):
        self._extractor = extractor or BridgeExtractor()
        self._prefs_prompt = prefs_prompt if prefs_prompt is not None else _load_prompt(_PREFS_PROMPT_FILE)
        self._world_prompt = world_prompt if world_prompt is not None else _load_prompt(_WORLD_PROMPT_FILE)
        self._staging_dir = os.path.expanduser(staging_dir)
        self._brain_inbox = os.path.expanduser(brain_inbox_dir)
        self._staging_mode = bool(staging_mode)
        self._confidence_floor = float(confidence_floor)
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._write = write_fn or self._default_write
        self._correction_markers = _normalise_correction_markers(correction_markers)
        self._existing_fact_lookup = existing_fact_lookup_fn
        self.stats = {"turns_routed": 0, "world_staged": 0, "world_deduped": 0,
                      "prefs_seen": 0, "extract_errors": 0, "fallback_passes": 0,
                      "corrections_detected": 0}

    def correction_marker(self, user: str, assistant: str = "") -> Optional[str]:
        return correction_marker_match(user, assistant, self._correction_markers)

    # -- extraction ---------------------------------------------------------
    def two_pass_extract(self, user: str, assistant: str) -> Dict[str, Any]:
        """Run the prefs pass and the world pass CONCURRENTLY (benchmark wiring note: collapse the
        2x sequential latency). Returns a dict with both pass results."""
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_prefs = ex.submit(self._extractor.extract, self._prefs_prompt, user, assistant)
            f_world = ex.submit(self._extractor.extract, self._world_prompt, user, assistant)
            prefs = f_prefs.result()
            world = f_world.result()
        return {"prefs": prefs, "world": world}

    # -- routing ------------------------------------------------------------
    def _classify(self, cands: List[Dict[str, Any]], allowed: tuple) -> List[Dict[str, Any]]:
        """Deterministic class filter: keep only candidates whose class label is in `allowed` and
        which clear the confidence floor. A candidate with an out-of-domain label is dropped (the
        prefs pass should never emit world classes and vice versa; if it does, it is misrouted noise)."""
        out = []
        for c in cands:
            cls = (c.get("class") or "").strip()
            if cls not in allowed:
                continue
            conf = c.get("confidence")
            if isinstance(conf, (int, float)) and conf < self._confidence_floor:
                continue
            out.append(c)
        return out

    def _contradiction_signals(self, world_facts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Detect world facts that contradict retrieved existing facts during the dedup phase.

        The lookup is optional and degraded-safe: if it is absent or fails, the router returns no
        contradiction signal and the drain worker keeps the normal salience path.
        """
        if not self._existing_fact_lookup:
            return []
        signals: List[Dict[str, str]] = []
        for fact in world_facts:
            content = str(fact.get("content") or "")
            if not content:
                continue
            try:
                existing_rows = self._existing_fact_lookup(content)
            except Exception as e:
                logger.debug("capture-router: existing-fact lookup degraded for contradiction check: %s", e)
                continue
            for existing in _existing_fact_texts(existing_rows):
                if facts_contradict(content, existing):
                    signals.append({
                        "type": "contradiction",
                        "fact": content[:240],
                        "existing": existing[:240],
                    })
                    break
        return signals

    def route_turn(self, user: str, assistant: str, *, turn_id: str, session: str,
                   ts: Optional[str] = None, stage: bool = True) -> Dict[str, Any]:
        """Full router pass for ONE turn. Runs the two concurrent extractions, applies the
        deterministic class router + dedup, and STAGES world/event facts. Returns a structured
        result for observability/replay. Never raises (fail-soft)."""
        result: Dict[str, Any] = {
            "turn_id": turn_id, "session": session, "ts": ts,
            "prefs_facts": [], "world_facts": [], "world_dropped": [],
            "destination": None, "usage": {}, "latency": 0.0,
            "providers": {}, "error": None,
            "correction_detected": False, "correction_signals": [],
        }
        signals: List[Dict[str, str]] = []
        correction_counted = False
        marker = self.correction_marker(user, assistant)
        if marker:
            signals.append({"type": "marker", "marker": marker})
            result["correction_detected"] = True
            result["correction_signals"] = signals
            self.stats["corrections_detected"] += 1
            correction_counted = True
        try:
            passes = self.two_pass_extract(user, assistant)
        except Exception as e:  # ThreadPool/executor level failure — should be rare (extract is soft)
            self.stats["extract_errors"] += 1
            result["error"] = f"two_pass_extract failed: {e}"
            return result

        prefs_res, world_res = passes["prefs"], passes["world"]
        result["providers"] = {"prefs": prefs_res.get("provider"), "world": world_res.get("provider")}
        for r in (prefs_res, world_res):
            if r.get("provider") == "gemini-bridge":
                self.stats["fallback_passes"] += 1
            if r.get("error"):
                self.stats["extract_errors"] += 1
        # combined tokens + latency (concurrent passes: latency is the MAX, tokens SUM)
        usage: Dict[str, Any] = {}
        for r in (prefs_res, world_res):
            for k, v in (r.get("usage") or {}).items():
                if isinstance(v, (int, float)):
                    usage[k] = usage.get(k, 0) + v
        result["usage"] = usage
        result["latency"] = max(prefs_res.get("latency") or 0.0, world_res.get("latency") or 0.0)

        prefs_cands = self._classify(prefs_res.get("candidates") or [], PREFS_CLASSES)
        world_raw = self._classify(world_res.get("candidates") or [], WORLD_CLASSES)
        # DEDUP world against prefs (the leak fix). RC1 also checks the kept world facts against
        # retrieved existing facts here: a same-subject conflicting exact value is a correction signal.
        world_kept, world_dropped = dedup_world_against_prefs(world_raw, prefs_cands)
        signals.extend(self._contradiction_signals(world_kept))
        if signals:
            result["correction_detected"] = True
            result["correction_signals"] = signals
            if not correction_counted:
                self.stats["corrections_detected"] += 1

        self.stats["prefs_seen"] += len(prefs_cands)
        self.stats["world_deduped"] += len(world_dropped)

        result["prefs_facts"] = prefs_cands       # destination: mem0 (written by the unchanged add path)
        result["world_facts"] = world_kept
        result["world_dropped"] = world_dropped

        # Deterministic destination for world/event facts.
        result["destination"] = "staging" if self._staging_mode else "brain-inbox"
        if stage:
            self.stage_route_result(result)

        self.stats["turns_routed"] += 1
        return result

    # -- staged write -------------------------------------------------------
    @staticmethod
    def _default_write(path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

    def stage_route_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Write a previously computed route result. Used by the drain worker to compute correction
        signals before add(), then stage world facts only after the mem0 write+scrub boundary is clean."""
        if result.get("staged_path"):
            return result
        world_facts = result.get("world_facts") or []
        if not world_facts:
            return result
        dest_dir = self._staging_dir if self._staging_mode else self._brain_inbox
        result["destination"] = "staging" if self._staging_mode else "brain-inbox"
        path = self._stage_world_facts(
            world_facts, dest_dir,
            turn_id=str(result.get("turn_id") or "unknown"),
            session=str(result.get("session") or "default"),
            ts=result.get("ts"),
        )
        result["staged_path"] = path
        self.stats["world_staged"] += len(world_facts)
        return result

    def _stage_world_facts(self, facts: List[Dict[str, Any]], dest_dir: str, *,
                           turn_id: str, session: str, ts: Optional[str]) -> str:
        """Write world/event facts for one turn as a single markdown file with YAML frontmatter.
        Path: <dest_dir>/<date>/<turn_id>.md. The frontmatter carries class/session/ts/source_turn
        so the nightly gbrain sync (once staging_mode is flipped off) can ingest with provenance."""
        now = self._now()
        date = now.strftime("%Y-%m-%d")
        # the dominant class in the file (for the frontmatter `class`); per-fact class is inline.
        classes = [f.get("class", "") for f in facts]
        top_class = max(set(classes), key=classes.count) if classes else "world_entity"
        fm_ts = ts or now.isoformat()
        lines = [
            "---",
            f"class: {top_class}",
            f"session: {session}",
            f"ts: {fm_ts}",
            f"source_turn: {turn_id}",
            f"routed_by: capture-router-armb",
            f"staged_at: {now.isoformat()}",
            "---",
            "",
            f"# Captured world knowledge — turn {turn_id}",
            "",
        ]
        for f in facts:
            conf = f.get("confidence")
            conf_str = f" _(confidence: {conf})_" if conf is not None else ""
            lines.append(f"- **[{f.get('class','')}]** {f.get('content','')}{conf_str}")
        content = "\n".join(lines) + "\n"
        path = os.path.join(dest_dir, date, f"{turn_id}.md")
        self._write(path, content)
        return path


def build_router_from_config(cfg: Dict[str, Any],
                             existing_fact_lookup_fn: Optional[Callable[[str], Any]] = None
                             ) -> Optional[CaptureRouter]:
    """Construct a CaptureRouter from the `mem0_capture_router` sub-block of mem0.json, or None if
    the flag is absent/off. Degrade-safe: any construction error -> None (router disabled, drain
    worker keeps its unchanged behavior)."""
    router_cfg = cfg.get("mem0_capture_router") or {}
    if not isinstance(router_cfg, dict) or not router_cfg.get("enabled"):
        return None
    try:
        extractor = BridgeExtractor(
            primary_url=str(router_cfg.get(
                "primary_url", "http://127.0.0.1:18812/v1/chat/completions")),
            fallback_url=str(router_cfg.get(
                "fallback_url", "http://192.168.1.216:18813/v1/chat/completions")),
            primary_secret_ref=str(router_cfg.get(
                "primary_secret_ref", "op://Engineering/codex-bridge/secret")),
            fallback_secret_ref=str(router_cfg.get(
                "fallback_secret_ref", "op://Engineering/gemini-bridge/secret")),
            model=str(router_cfg.get("model", "gpt-5.4-mini")),
            fallback_model=router_cfg.get("fallback_model") or None,
            timeout_s=float(router_cfg.get("timeout_s", 180.0)),
        )
        return CaptureRouter(
            extractor=extractor,
            staging_dir=str(router_cfg.get("staging_dir", _DEFAULT_STAGING_DIR)),
            brain_inbox_dir=str(router_cfg.get("brain_inbox_dir", _DEFAULT_BRAIN_INBOX)),
            staging_mode=bool(router_cfg.get("staging_mode", True)),
            confidence_floor=float(router_cfg.get("confidence_floor", 0.0)),
            correction_markers=router_cfg.get("correction_markers"),
            existing_fact_lookup_fn=(
                existing_fact_lookup_fn
                if bool(router_cfg.get("contradiction_lookup_enabled", True)) else None
            ),
        )
    except Exception as e:
        logger.warning("capture-router: build failed (router disabled): %s", e)
        return None
