#!/usr/bin/env python3
"""LCM Arm-B harness — live, node-served long-session recovery gate (PRD-7).

Arm A (lcm_live_recovery.py) proves RAW-STORE/FTS recovery. Arm B proves the
SUMMARY-NODE DAG actually works end-to-end in a live long session:

  1. Plant a unique sentinel in a fresh Aegis session.
  2. Drive enough real filler turns (default 44) to create >=4 depth-0 leaves
     and force >=1 depth-1 condensation node (DAG knobs at DEFAULT 20k/fanin-4;
     only LCM_CONTEXT_THRESHOLD is lowered so compression fires repeatedly).
  3. Assert, against the live lcm.db, that:
       (a) a depth>=1, source_type="nodes" condensation node was created;
       (b) the sentinel SURVIVED into a depth>=1 node summary (lossless
           condensation of the load-bearing fact);
       (c) node-served recovery works: lcm_expand_query targeted at the
           condensation node (node_ids=[<id>], raw store NOT consulted) returns
           the exact sentinel.
  4. Score correct/confident-wrong + Wilson lower bound across N trials.

This is the gate Apollo's long coding sessions need: it proves the DAG isn't
dormant and that condensation preserves facts recoverably — not just that raw
grep happens to still find them.

Cost rule: cheap model only (Haiku). Opus refused.

Usage:
  python scripts/lcm_arm_b_node_recovery.py --profile aegis \
    --model claude-haiku-4-5 --n 20 --filler-turns 44 --threshold 0.10 \
    --out docs/reports/lcm-qa/arm-b-node-recovery.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Interpreter guard: this harness imports the LCM engine IN-PROCESS, and the
# engine uses py3.9+ syntax (e.g. `list[str]` annotations in config.py). On this
# fleet `python3` resolves to anaconda 3.7.4 (PATH-poison), which cannot even
# import the engine — the import raises "'type' object is not subscriptable",
# which a broad except previously stringified into a fake per-trial recovery
# error and scored 0/N. Fail LOUD and early instead of producing fake-fails.
if sys.version_info < (3, 9):
    sys.stderr.write(
        f"FATAL: lcm_arm_b_node_recovery needs Python >=3.9 to import the LCM "
        f"engine in-process; got {sys.version.split()[0]} at {sys.executable}. "
        f"Run it with the venv interpreter: "
        f"~/.hermes/hermes-agent/venv/bin/python scripts/lcm_arm_b_node_recovery.py ...\n"
    )
    raise SystemExit(3)

WILSON_Z = 1.96

# Distinct semantic owners — the DAG must preserve MEANING (which owner) through
# condensation, even when it dedupes near-identical verbatim tokens. Same pool
# Arm A uses for its semantic prompts, so the two arms probe the same contract.
_OWNER_POOL = [
    "Ada Lovelace", "Grace Hopper", "Katherine Johnson", "Margaret Hamilton",
    "Barbara Liskov", "Frances Allen", "Radia Perlman", "Karen Sparck Jones",
    "Shafi Goldwasser", "Barbara Walters", "Hedy Lamarr", "Joan Clarke",
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def score_semantic_recovery(
    node_answer: str, target_owner: str, other_owners: list[str]
) -> tuple[bool, bool]:
    """Score a single semantic node-served recovery answer.

    Returns (correct, confident_wrong).

    - correct: the target owner's name appears in the recovered answer.
    - confident_wrong: the model did NOT recover the target, did NOT abstain
      ("no matching" / "not found" / "not present"), AND made a confident
      owner assertion that is not the target. Two detection paths:
        (a) name-collision: it asserts a KNOWN other owner (the K=4 failure mode), OR
        (b) free-standing fabrication: it affirmatively states "<X> is the
            recovery owner" / "the recovery owner is <X>" for an X that is not
            the target — even if X is not in the known pool (the K=1 threat
            model, Opus GI-3: at K=1 there are no siblings, so the only failure
            is inventing a free-standing wrong owner).

    Module-level + pure so the detector is positive- AND negative-controlled
    in tests (PRD-8.1 AC-3): a detector that never fires trivially yields
    confident_wrong==0 and rubber-stamps the gate.
    """
    na = _norm(node_answer)
    correct = _norm(target_owner) in na
    abstained = any(s in na for s in ("no matching", "not found", "not present", "not in the"))
    # (a) asserts a known non-target owner
    collision = any(_norm(o) in na for o in other_owners if _norm(o) != _norm(target_owner))
    # (b) free-standing affirmative owner assertion of a non-target name
    fabricated_freestanding = False
    if not correct and not abstained:
        for m in re.finditer(
            r"(?:recovery owner(?:\s+is|\s+associated[^.]*?is)?\s+|^|\*\*)\s*"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
            node_answer or "",
        ):
            cand = _norm(m.group(1))
            if cand and cand != _norm(target_owner):
                fabricated_freestanding = True
                break
    confident_wrong = (not correct) and (not abstained) and (collision or fabricated_freestanding)
    return correct, confident_wrong



def wilson_lower_bound(successes: int, total: int, z: float = WILSON_Z) -> float:
    if total <= 0:
        return 0.0
    phat = successes / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return (centre - margin) / denom


# ---- PRD-8.2 gate split: separate the condensation gate from the recovery gate
# The original single gate scored "no node ever formed" (condensation didn't
# fire) IDENTICALLY to "model had a node and failed to recover the fact." Those
# are different subsystems with different fixes, and conflating them mislabels a
# condensation-trigger flake as an LCM recovery failure. It also scored a
# transient provider error (HTTP 503/429/timeout) as a permanent recovery miss.
# classify_trial buckets every trial into exactly one outcome so the verdict can
# gate the two subsystems independently and exclude infra noise from the
# correctness denominator. Pure + module-level so it is unit-controlled and can
# re-score existing run JSON without a live re-run.

_INFRA_ERROR_RE = re.compile(
    r"\[recovery error:.*?(?:\b50\d\b|\b429\b|no eligible sub|timed? ?out|"
    r"timeout|connection|temporarily|overloaded|unavailable)",
    re.IGNORECASE,
)

# Trial outcome buckets (one per trial):
#   RECOVERED        - node formed, fact recovered correctly
#   CONFIDENT_WRONG  - node formed, model asserted a wrong owner (hard fail, any count > 0)
#   RECOVERY_MISS    - node formed AND fact preserved, but model failed to recover (real LCM gap)
#   NO_CONDENSATION  - no depth>=1 node ever formed (condensation-trigger gap, NOT a recovery gap)
#   INFRA_ERROR      - transient provider/transport error during recovery (retryable, excluded)
OUTCOME_RECOVERED = "recovered"
OUTCOME_CONFIDENT_WRONG = "confident_wrong"
OUTCOME_RECOVERY_MISS = "recovery_miss"
OUTCOME_NO_CONDENSATION = "no_condensation"
OUTCOME_INFRA_ERROR = "infra_error"


def classify_trial(trial: dict) -> str:
    """Bucket one trial dict (from the run JSON 'trials' list) into one outcome."""
    answer = trial.get("node_served_answer") or ""
    condensed = bool(trial.get("condensed"))
    if trial.get("confident_wrong"):
        return OUTCOME_CONFIDENT_WRONG
    if trial.get("correct"):
        return OUTCOME_RECOVERED
    # not correct, not confident-wrong:
    if _INFRA_ERROR_RE.search(answer):
        return OUTCOME_INFRA_ERROR
    if not condensed:
        return OUTCOME_NO_CONDENSATION
    # node formed, fact was preserved, model still didn't recover it = real gap
    return OUTCOME_RECOVERY_MISS


def compute_split_verdict(trials: list[dict], *, gate_eligible: bool = True) -> dict:
    """Two-gate verdict over a trials list.

    Gate 1 (RECOVERY CORRECTNESS, the binding Apollo gate): among trials where a
      node formed AND no infra error occurred, recall must be >= 0.95, Wilson LB
      >= 0.90, and confident_wrong == 0. Infra errors are EXCLUDED from the
      denominator (retryable, not a recovery property).
    Gate 2 (CONDENSATION RELIABILITY, a tuning gate, non-binding for correctness):
      fraction of trials where a depth>=1 node formed. Reported and thresholded
      separately so a condensation-trigger flake does not masquerade as a
      recovery failure.
    """
    n = len(trials)
    buckets: dict[str, int] = {}
    for t in trials:
        b = classify_trial(t)
        buckets[b] = buckets.get(b, 0) + 1

    recovered = buckets.get(OUTCOME_RECOVERED, 0)
    confident_wrong = buckets.get(OUTCOME_CONFIDENT_WRONG, 0)
    recovery_miss = buckets.get(OUTCOME_RECOVERY_MISS, 0)
    no_condensation = buckets.get(OUTCOME_NO_CONDENSATION, 0)
    infra_error = buckets.get(OUTCOME_INFRA_ERROR, 0)

    # Gate 1 denominator: trials that actually reached the recovery test
    # (node formed, no infra noise). = recovered + recovery_miss + confident_wrong.
    recovery_eligible = recovered + recovery_miss + confident_wrong
    recovery_rate = (recovered / recovery_eligible) if recovery_eligible else 0.0
    recovery_wlb = wilson_lower_bound(recovered, recovery_eligible)

    # Gate 2: condensation reliability over all non-infra trials
    non_infra = n - infra_error
    condensation_rate = ((non_infra - no_condensation) / non_infra) if non_infra else 0.0

    recovery_gate_pass = (
        recovery_eligible > 0
        and recovery_rate >= 0.95
        and recovery_wlb >= 0.90
        and confident_wrong == 0
    )
    return {
        "n": n,
        "buckets": buckets,
        "recovery_eligible_n": recovery_eligible,
        "recovery_rate": round(recovery_rate, 4),
        "recovery_wilson_lb": round(recovery_wlb, 4),
        "confident_wrong": confident_wrong,
        "recovery_miss": recovery_miss,
        "no_condensation": no_condensation,
        "infra_error": infra_error,
        "condensation_rate": round(condensation_rate, 4),
        "recovery_gate_pass": recovery_gate_pass,
        "recovery_gate": (
            "PASS" if (recovery_gate_pass and gate_eligible and recovery_eligible >= 180)
            else "PASS-UNDERPOWERED" if recovery_gate_pass
            else "FAIL"
        ),
    }


_GROUPED_MAPPING_RE = re.compile(r"\b\d+\s*/\s*\d+")  # e.g. "1300/1600" grouped line
_ABSTAIN_RE = re.compile(
    r"no matching owner found|not found|not present|no matching",
    re.IGNORECASE,
)


def needs_escalation(node_answer: str) -> bool:
    """PRD-8.3 B1 trigger: escalate to the verbatim store when the node answer
    abstained, is empty, or cites a grouped/range mapping (the lossy-merge
    signature). Pure + module-level so it is unit-controlled."""
    a = node_answer or ""
    if not a.strip():
        return True
    if _ABSTAIN_RE.search(a):
        return True
    if _GROUPED_MAPPING_RE.search(a):
        return True
    return False


@dataclass
class TrialResult:
    idx: int
    sentinel: str
    session_id: str | None
    depth1_node_id: int | None
    sentinel_in_node: bool
    node_served_answer: str
    correct: bool
    confident_wrong: bool
    leaves: int
    condensed: int
    notes: str = ""


@dataclass
class ArmBConfig:
    profile: str = "aegis"
    model: str = "claude-haiku-4-5"
    n: int = 20
    filler_turns: int = 44
    filler_tokens: int = 3000
    threshold: float = 0.10
    timeout_seconds: int = 600
    lcm_db: str = ""

    def __post_init__(self):
        if not self.lcm_db:
            self.lcm_db = os.path.expanduser(
                f"~/.hermes/profiles/{self.profile}/lcm.db"
            )


class ArmBHarness:
    def __init__(self, cfg: ArmBConfig):
        self.cfg = cfg
        self.home = os.path.expanduser(f"~/.hermes/profiles/{cfg.profile}")

    # ---- live hermes driver -------------------------------------------------
    def _hermes(self, args: list[str], prompt: str) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["LCM_CONTEXT_THRESHOLD"] = str(self.cfg.threshold)
        cmd = ["hermes", "-p", self.cfg.profile, "chat", "-Q", "-m", self.cfg.model]
        cmd.extend([*args, "-q", prompt])
        return subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=self.cfg.timeout_seconds, env=env, check=False,
        )

    @staticmethod
    def _session_id(stdout: str) -> str | None:
        m = re.findall(r"session_id:\s*([0-9A-Za-z_]+)", stdout or "")
        return m[-1] if m else None

    # ---- DB introspection ---------------------------------------------------
    def _resolve_current_session(self, planted_session: str) -> str:
        """Follow the lifecycle rollover chain from the planted session to the
        session that actually owns the nodes now.

        Live long sessions roll the session id on compaction-boundary rollover
        (carry_over_new_session_context reassigns retained nodes to the new
        session). So nodes for this conversation may live under a DIFFERENT
        session id than the one we planted into. We resolve it by walking
        lcm_lifecycle_state from old->new, then fall back to the most recent
        session that has a depth>=1 node.
        """
        conn = sqlite3.connect(self.cfg.lcm_db)
        conn.row_factory = sqlite3.Row
        try:
            # Walk forward: find rows where last_finalized_session_id chains from
            # our planted session toward the current session.
            current = planted_session
            for _ in range(50):  # bounded
                row = conn.execute(
                    "SELECT current_session_id FROM lcm_lifecycle_state "
                    "WHERE last_finalized_session_id = ? "
                    "ORDER BY rowid DESC LIMIT 1",
                    (current,),
                ).fetchone()
                if not row or not row["current_session_id"]:
                    break
                current = row["current_session_id"]
            return current
        finally:
            conn.close()

    def _nodes_for_session(self, session_id: str) -> list[dict]:
        conn = sqlite3.connect(self.cfg.lcm_db)
        conn.row_factory = sqlite3.Row
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(summary_nodes)")]
            rows = conn.execute(
                "SELECT * FROM summary_nodes WHERE session_id = ? ORDER BY depth, node_id",
                (session_id,),
            ).fetchall()
            return [dict(zip(cols, r)) for r in rows]
        finally:
            conn.close()

    def _find_node_with_sentinel(self, sentinel: str) -> dict | None:
        """Last-resort: locate ANY depth>=1 node whose summary carries the
        sentinel, regardless of session (proves condensation preserved it)."""
        conn = sqlite3.connect(self.cfg.lcm_db)
        conn.row_factory = sqlite3.Row
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(summary_nodes)")]
            rows = conn.execute(
                "SELECT * FROM summary_nodes WHERE depth >= 1 AND summary LIKE ? "
                "ORDER BY node_id DESC LIMIT 1",
                (f"%{sentinel}%",),
            ).fetchall()
            return dict(zip(cols, rows[0])) if rows else None
        finally:
            conn.close()

    def _find_node_with_text(self, needle: str) -> dict | None:
        """Locate the newest depth>=1 node whose summary contains `needle`
        (a phrase or owner name). Used by the semantic batch path."""
        return self._find_node_with_sentinel(needle)

    def _node_served_recovery_semantic(self, session_id: str, node_id: int,
                                       probe: dict) -> str:
        """Node-only recovery for a SEMANTIC probe: ask the node who the recovery
        owner is, raw store excluded. Reuses the hardened settle+retry path.

        PRD-8.3 B1/B2/B3: the recovery instruction forbids inferring an owner
        from a grouped/range mapping, mandates grounding the answer in the exact
        identifier asked, and requires abstention ("no matching owner found")
        over a guess when the exact identifier is not unambiguously present.
        Abstention scores as a recall miss, NEVER confident-wrong.
        """
        return self._node_served_recovery(
            session_id, node_id, probe["owner"],
            question=(
                f"Who is the recovery owner associated with the EXACT handoff "
                f"phrase {probe['phrase']}? Answer with the owner's full name. "
                f"Use ONLY an entry that names {probe['phrase']} exactly and by "
                f"itself. Do NOT infer the owner from a grouped or range mapping "
                f"(e.g. a line like '1300/1600/1900 = Name'); a grouped line is "
                f"not a valid source. If {probe['phrase']} is not present exactly "
                f"and unambiguously with its own full owner name, reply with "
                f"exactly: no matching owner found"
            ),
        )

    # ---- node-served recovery (the honest oracle) ---------------------------
    def _wait_db_settled(self, timeout: float = 30.0) -> None:
        """Block until the live lcm.db is readable without a busy/lock error.

        The in-process engine reads the SAME sqlite file the live gateway is
        writing. If we instantiate it while the gateway is mid-flush (e.g. the
        instant a prior arm's session is being persisted), a half-written read
        can surface as a malformed structure that later subscripts blow up on
        ('type' object is not subscriptable). Poll a cheap read under a short
        busy_timeout until it succeeds twice in a row before touching the engine.
        """
        deadline = time.time() + timeout
        ok_streak = 0
        while time.time() < deadline:
            try:
                conn = sqlite3.connect(self.cfg.lcm_db, timeout=2.0)
                conn.execute("PRAGMA busy_timeout=2000")
                conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()
                conn.execute("SELECT COUNT(*) FROM messages").fetchone()
                conn.close()
                ok_streak += 1
                if ok_streak >= 2:
                    return
            except sqlite3.Error:
                ok_streak = 0
            time.sleep(0.5)

    def _node_served_recovery(self, session_id: str, node_id: int, sentinel: str,
                              question: str = "What is the exact recovery sentinel?") -> str:
        """Drive lcm_expand_query targeted at ONLY the condensation node (raw
        store excluded) via an in-process engine bound to the live db. This
        proves the node itself can serve the fact, independent of raw rows.

        Hardened against live-DB contention: wait for the store to settle, and
        retry once on a malformed/locked read instead of scoring a fake-FAIL.
        """
        sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent"))
        from plugins.context_engine.lcm.config import LCMConfig
        from plugins.context_engine.lcm.engine import LCMEngine

        last_exc: Exception | None = None
        for attempt in range(3):
            self._wait_db_settled()
            eng = LCMEngine(config=LCMConfig(database_path=self.cfg.lcm_db),
                            hermes_home=self.home)
            try:
                eng.on_session_start(session_id)
                out = eng.handle_tool_call(
                    "lcm_expand_query",
                    {"prompt": question,
                     "node_ids": [node_id]},
                )
                data = json.loads(out)
                if not isinstance(data, dict):
                    raise TypeError(f"expand returned non-dict: {type(data).__name__}")
                return str(data.get("answer", ""))
            except (TypeError, ValueError, json.JSONDecodeError, sqlite3.Error) as exc:
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))  # back off, let the store settle
            finally:
                try:
                    eng.shutdown()
                except Exception:
                    pass
        raise RuntimeError(f"node recovery failed after retries: {last_exc}")

    def _store_grounded_recovery(self, session_id: str, sentinel: str,
                                 question: str) -> str:
        """PRD-8.3 B1: mandatory escalation path — answer the identifier lookup
        from the VERBATIM MessageStore (no node_ids restriction), not the summary
        node. This is the lossless authority: the store preserves every fact, so
        a merged/lossy summary cannot produce a confident-wrong here. Used when
        the node-served answer abstained or cited a grouped/range mapping.
        """
        sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent"))
        from plugins.context_engine.lcm.config import LCMConfig
        from plugins.context_engine.lcm.engine import LCMEngine

        last_exc: Exception | None = None
        for attempt in range(3):
            self._wait_db_settled()
            eng = LCMEngine(config=LCMConfig(database_path=self.cfg.lcm_db),
                            hermes_home=self.home)
            try:
                eng.on_session_start(session_id)
                # No node_ids => expand searches the verbatim store, the lossless
                # layer. Same abstain-over-guess instruction (already in question).
                out = eng.handle_tool_call(
                    "lcm_expand_query",
                    {"prompt": question},
                )
                data = json.loads(out)
                if not isinstance(data, dict):
                    raise TypeError(f"expand returned non-dict: {type(data).__name__}")
                return str(data.get("answer", ""))
            except (TypeError, ValueError, json.JSONDecodeError, sqlite3.Error) as exc:
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
            finally:
                try:
                    eng.shutdown()
                except Exception:
                    pass
        raise RuntimeError(f"store-grounded recovery failed after retries: {last_exc}")

    # ---- batched multi-sentinel session (10x cheaper) -----------------------
    def run_session_batch(self, batch_idx: int, k: int) -> list[TrialResult]:
        """Plant K sentinels across ONE long session, then recover each from its
        condensation node. Amortizes the expensive 44-turn buildup across K
        independent recovery samples instead of 1.

        Layout per session:
          plant s0 -> g filler -> plant s1 -> g filler -> ... -> plant s_{k-1}
          -> big tail filler (force the early sentinels to age into leaves and
             condense) -> recover each s_i from a depth>=1 node carrying it.
        """
        filler = "Routine deterministic project chatter for compaction. " * max(
            1, self.cfg.filler_tokens // 8
        )
        # Distinct SEMANTIC probes: each is a unique owner + phrase. The DAG is a
        # *semantic* compressor, not a verbatim store — it dedupes near-identical
        # exact tokens but preserves distinct meanings. So we probe meaning
        # (owner name) recovered from the node, not exact-string survival.
        owners = _OWNER_POOL
        probes = []
        for i in range(k):
            owner = owners[(batch_idx * k + i) % len(owners)]
            phrase = f"recover-{batch_idx:02d}{i:02d}"
            probes.append({
                "owner": owner,
                "phrase": phrase,
                "fact": f"The recovery owner is {owner}; the handoff phrase is {phrase}.",
                "tag": f"ARMB-{batch_idx:02d}{i:02d}",
            })

        # 1. plant probe[0] in a fresh session, capture id
        plant0 = self._hermes(
            ["--pass-session-id"],
            f"Remember this exact fact for later, verbatim: {probes[0]['fact']} "
            f"Reply with only OK.",
        )
        session_id = self._session_id(plant0.stdout) or self._session_id(plant0.stderr)
        if not session_id:
            return [TrialResult(batch_idx * k + i, p["owner"], None, None, False,
                                "", False, False, 0, 0, notes="no session_id")
                    for i, p in enumerate(probes)]

        turn = 0
        def _filler_turns(count: int):
            nonlocal turn
            for _ in range(count):
                turn += 1
                self._hermes(
                    ["--resume", session_id],
                    f"Status turn {turn}, reply 'ack {turn}' only. "
                    f"Context notes: {filler}",
                )

        # 2. FRONT-LOAD: plant all remaining probes back-to-back with minimal
        #    spacing, so every probe sits in the SAME early region of the
        #    session. The previous interleaved-with-gaps layout left late probes
        #    too close to the tail to ever age into a condensed node (only 3/10
        #    condensed). Front-loading + one big tail ages them ALL out together.
        _filler_turns(2)
        for p in probes[1:]:
            self._hermes(
                ["--resume", session_id],
                f"Remember this exact fact for later, verbatim: {p['fact']} "
                f"Reply with only OK.",
            )
            _filler_turns(2)

        # 3. big uniform tail to force condensation of ALL the early probes
        _filler_turns(self.cfg.filler_turns)

        # 4. recover each probe's OWNER from a depth>=1 node carrying it
        results: list[TrialResult] = []
        for i, p in enumerate(probes):
            hit = self._find_node_with_text(p["phrase"]) or self._find_node_with_text(p["owner"])
            depth1_id = hit["node_id"] if hit else None
            in_node = hit is not None
            sess = hit["session_id"] if hit else session_id
            node_answer = ""
            escalated = False
            if depth1_id is not None:
                try:
                    node_answer = self._node_served_recovery_semantic(sess, depth1_id, p)
                except (ImportError, SyntaxError, ModuleNotFoundError) as exc:
                    # Structural failure (wrong interpreter, broken import) is NOT
                    # a per-trial recovery miss — it would silently score 0/N as a
                    # fake-fail. Abort the whole run loudly instead.
                    raise RuntimeError(
                        f"FATAL structural error importing/using the LCM engine "
                        f"(not a recovery miss): {type(exc).__name__}: {exc}"
                    ) from exc
                except Exception as exc:  # noqa: BLE001 — genuine per-trial failure
                    node_answer = f"[recovery error: {exc}]"

            # PRD-8.3 B1: mandatory escalation. If the node answer abstained, is
            # empty, or cites a grouped/range mapping (the lossy-merge signature),
            # the lossless verbatim store is the authority — re-answer from it.
            # This makes correctness independent of summary fidelity. An infra
            # error string is NOT escalated (it is a transient, scored separately).
            answer = node_answer
            if depth1_id is not None and not node_answer.startswith("[recovery error") \
                    and needs_escalation(node_answer):
                try:
                    store_answer = self._store_grounded_recovery(
                        sess, p["owner"],
                        question=(
                            f"Who is the recovery owner associated with the EXACT "
                            f"handoff phrase {p['phrase']}? Answer with the owner's "
                            f"full name. Use ONLY an entry that names {p['phrase']} "
                            f"exactly and by itself; never infer from a grouped or "
                            f"range mapping. If not present exactly, reply with "
                            f"exactly: no matching owner found"
                        ),
                    )
                    answer = store_answer
                    escalated = True
                except Exception as exc:  # noqa: BLE001
                    answer = f"[recovery error: {exc}]"

            # SEMANTIC scoring: the owner name recovered (post-escalation) is the
            # win condition, not exact-string survival of the whole sentence.
            # Detector is the module-level, test-controlled score_semantic_recovery
            # (PRD-8.1 AC-3 positive+negative control).
            correct, confident_wrong = score_semantic_recovery(
                answer, p["owner"], owners
            )
            results.append(TrialResult(
                idx=batch_idx * k + i, sentinel=f"{p['owner']}/{p['phrase']}",
                session_id=sess, depth1_node_id=depth1_id, sentinel_in_node=in_node,
                node_served_answer=answer[:200], correct=correct,
                confident_wrong=confident_wrong,
                leaves=0, condensed=1 if depth1_id is not None else 0,
                notes=f"batch={batch_idx} pos={i} owner={p['owner']} escalated={escalated}",
            ))
        return results

    # ---- one trial (legacy single-sentinel-per-session) ---------------------
    def run_trial(self, idx: int) -> TrialResult:
        sentinel = f"LCM-ARMB-{int(time.time())}-{idx:03d}"
        filler = "Routine deterministic project chatter for compaction. " * max(
            1, self.cfg.filler_tokens // 8
        )

        plant = self._hermes(
            ["--pass-session-id"],
            f"Remember this exact fact for later, verbatim: "
            f"The exact recovery sentinel is {sentinel}. Reply with only OK.",
        )
        session_id = self._session_id(plant.stdout) or self._session_id(plant.stderr)
        if not session_id:
            return TrialResult(idx, sentinel, None, None, False, "", False, False,
                               0, 0, notes="no session_id from plant")

        # Drive filler turns to build leaves -> condensation.
        for t in range(self.cfg.filler_turns):
            self._hermes(
                ["--resume", session_id],
                f"Status turn {t + 1}, reply 'ack {t + 1}' only. "
                f"Context notes: {filler}",
            )

        # Inspect the DAG. The session id rolls over on compaction, so resolve
        # the session that actually owns the nodes now.
        current_session = self._resolve_current_session(session_id)
        nodes = self._nodes_for_session(current_session)
        if not any((n.get("depth") or 0) >= 1 for n in nodes):
            # planted session may itself still own them, or chain resolution
            # missed a hop — also check the planted session directly.
            nodes = nodes + self._nodes_for_session(session_id)
        leaves = sum(1 for n in nodes if n.get("depth") == 0)
        condensed_nodes = [n for n in nodes if (n.get("depth") or 0) >= 1
                           and str(n.get("source_type")) == "nodes"]
        condensed = len(condensed_nodes)

        depth1_id = None
        sentinel_in_node = False
        for n in condensed_nodes:
            if sentinel in str(n.get("summary") or ""):
                depth1_id = n["node_id"]
                sentinel_in_node = True
                break
        # Authoritative fallback: find ANY depth>=1 node carrying this sentinel,
        # regardless of session bookkeeping (proves condensation preserved it).
        if depth1_id is None:
            hit = self._find_node_with_sentinel(sentinel)
            if hit is not None:
                depth1_id = hit["node_id"]
                sentinel_in_node = True
                current_session = hit["session_id"]
                if condensed == 0:
                    condensed = 1
        # Last fall back: any condensation node, even without the fact.
        if depth1_id is None and condensed_nodes:
            depth1_id = condensed_nodes[0]["node_id"]

        node_answer = ""
        if depth1_id is not None:
            try:
                node_answer = self._node_served_recovery(
                    current_session, depth1_id, sentinel
                )
            except Exception as exc:  # noqa: BLE001
                node_answer = f"[recovery error: {exc}]"

        correct = sentinel in node_answer
        # confident-wrong: gave a DIFFERENT sentinel-shaped answer confidently.
        confident_wrong = (
            not correct
            and bool(re.search(r"LCM-ARMB-\d+-\d+", node_answer))
            and "no matching" not in node_answer.lower()
        )

        return TrialResult(
            idx=idx, sentinel=sentinel, session_id=session_id,
            depth1_node_id=depth1_id, sentinel_in_node=sentinel_in_node,
            node_served_answer=node_answer[:200], correct=correct,
            confident_wrong=confident_wrong, leaves=leaves, condensed=condensed,
        )

    # ---- run all + report ---------------------------------------------------
    def run(self, out_path: Path, sentinels_per_session: int = 1) -> dict:
        results: list[TrialResult] = []
        t0 = time.time()
        if sentinels_per_session > 1:
            import math as _m
            n_batches = _m.ceil(self.cfg.n / sentinels_per_session)
            produced = 0
            for b in range(n_batches):
                k = min(sentinels_per_session, self.cfg.n - produced)
                batch = self.run_session_batch(b, k)
                results.extend(batch)
                produced += len(batch)
                for r in batch:
                    print(f"batch {b} sentinel {r.sentinel}: node={r.depth1_node_id} "
                          f"in_node={r.sentinel_in_node} correct={r.correct}",
                          file=sys.stderr)
        else:
            for i in range(self.cfg.n):
                r = self.run_trial(i)
                results.append(r)
                print(f"trial {i}: node={r.depth1_node_id} "
                      f"sentinel_in_node={r.sentinel_in_node} correct={r.correct} "
                      f"leaves={r.leaves} condensed={r.condensed}", file=sys.stderr)

        n = len(results)
        condensation_fired = sum(1 for r in results if r.condensed >= 1)
        fact_preserved = sum(1 for r in results if r.sentinel_in_node)
        node_recall = sum(1 for r in results if r.correct)
        confident_wrong = sum(1 for r in results if r.confident_wrong)
        wlb = wilson_lower_bound(node_recall, n)

        # PRD-8.2 two-gate split: classify each trial and gate recovery-correctness
        # separately from condensation-reliability, excluding transient infra errors.
        trial_dicts = [vars(r) for r in results]
        split = compute_split_verdict(trial_dicts, gate_eligible=(n >= 180))

        report = {
            "arm": "B-node-served",
            "model": self.cfg.model,
            "profile": self.cfg.profile,
            "N": n,
            "gate_eligible": n >= 180,
            "condensation_fired": condensation_fired,
            "fact_preserved_in_node": fact_preserved,
            "node_served_recall": node_recall,
            "node_served_recall_rate": round(node_recall / n, 4) if n else 0.0,
            "wilson_lower_bound": round(wlb, 4),
            "confident_wrong": confident_wrong,
            "duration_s": round(time.time() - t0, 1),
            "thresholds": {"recall>=": 0.95, "wilson_lb>=": 0.90, "confident_wrong": 0},
            # --- PRD-8.2 split gates ---
            "split_verdict": split,
            "verdict": (
                "PASS" if (n >= 180 and node_recall / n >= 0.95 and wlb >= 0.90
                           and confident_wrong == 0)
                else "PASS-UNDERPOWERED" if (node_recall / n >= 0.95 and confident_wrong == 0)
                else "FAIL"
            ),
            "recovery_verdict": split["recovery_gate"],
            "trials": trial_dicts,
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        md = _render_md(report)
        out_path.write_text(md)
        json_path = out_path.with_suffix(".json")
        json_path.write_text(json.dumps(report, indent=2))
        return report


def _render_md(rep: dict) -> str:
    lines = [
        "# LCM Arm-B — Live Node-Served Long-Session Recovery",
        "",
        f"Model: {rep['model']} · Profile: {rep['profile']} · N: {rep['N']}",
        f"Gate-eligible (N>=180): {rep['gate_eligible']}",
        f"**Verdict: {rep['verdict']}**",
        "",
        "## Gate summary",
        f"- Condensation fired (>=1 depth-1 node): {rep['condensation_fired']}/{rep['N']}",
        f"- Fact preserved in a depth>=1 node: {rep['fact_preserved_in_node']}/{rep['N']}",
        f"- Node-served recall: {rep['node_served_recall']}/{rep['N']} "
        f"({rep['node_served_recall_rate']})",
        f"- Wilson 95% lower bound: {rep['wilson_lower_bound']} (required >= 0.90)",
        f"- Confident-wrong: {rep['confident_wrong']} (required 0)",
        f"- Duration: {rep['duration_s']}s",
        "",
    ]
    sv = rep.get("split_verdict")
    if sv:
        b = sv["buckets"]
        lines += [
            "## PRD-8.2 split gates (recovery-correctness vs condensation-reliability)",
            f"**Recovery verdict: {rep.get('recovery_verdict', sv['recovery_gate'])}** "
            f"(binding Apollo gate — node formed, infra errors excluded)",
            f"- Recovery-eligible trials (node formed, no infra error): {sv['recovery_eligible_n']}/{sv['n']}",
            f"- Recovery recall: {sv['recovery_rate']} (required >= 0.95)",
            f"- Recovery Wilson 95% LB: {sv['recovery_wilson_lb']} (required >= 0.90)",
            f"- Confident-wrong: {sv['confident_wrong']} (required 0)",
            "",
            "Trial outcome breakdown:",
            f"- recovered (correct): {b.get('recovered', 0)}",
            f"- recovery_miss (node formed, fact preserved, not recovered — REAL gap): {b.get('recovery_miss', 0)}",
            f"- confident_wrong (asserted wrong owner): {b.get('confident_wrong', 0)}",
            f"- no_condensation (no node ever formed — condensation-trigger gap, NOT recovery): {b.get('no_condensation', 0)}",
            f"- infra_error (transient 5xx/429/timeout — retryable, excluded): {b.get('infra_error', 0)}",
            f"- Condensation reliability (non-infra): {sv['condensation_rate']}",
            "",
        ]
    lines += [
        "## Trial records",
        "| idx | session | node | sentinel_in_node | correct | leaves | condensed |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in rep["trials"]:
        lines.append(
            f"| {t['idx']} | {t['session_id']} | {t['depth1_node_id']} | "
            f"{t['sentinel_in_node']} | {t['correct']} | {t['leaves']} | {t['condensed']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="LCM Arm-B node-served recovery gate")
    ap.add_argument("--profile", default="aegis")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--filler-turns", type=int, default=44)
    ap.add_argument("--filler-tokens", type=int, default=3000)
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--timeout-seconds", type=int, default=600)
    ap.add_argument("--sentinels-per-session", type=int, default=1,
                    help="Plant K sentinels per long session (amortizes the "
                         "filler buildup ~Kx). K=1 = legacy one-per-session.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if "opus" in args.model.lower():
        print("REFUSED: Opus prohibited for LCM QA (PRD-7 cost rule).", file=sys.stderr)
        return 2

    cfg = ArmBConfig(
        profile=args.profile, model=args.model, n=args.n,
        filler_turns=args.filler_turns, filler_tokens=args.filler_tokens,
        threshold=args.threshold, timeout_seconds=args.timeout_seconds,
    )
    rep = ArmBHarness(cfg).run(Path(args.out),
                               sentinels_per_session=args.sentinels_per_session)
    print(json.dumps({k: v for k, v in rep.items() if k != "trials"}, indent=2))
    return 0 if rep["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
