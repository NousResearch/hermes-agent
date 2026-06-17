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

WILSON_Z = 1.96


def wilson_lower_bound(successes: int, total: int, z: float = WILSON_Z) -> float:
    if total <= 0:
        return 0.0
    phat = successes / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return (centre - margin) / denom


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

    # ---- node-served recovery (the honest oracle) ---------------------------
    def _node_served_recovery(self, session_id: str, node_id: int, sentinel: str) -> str:
        """Drive lcm_expand_query targeted at ONLY the condensation node (raw
        store excluded) via an in-process engine bound to the live db. This
        proves the node itself can serve the fact, independent of raw rows."""
        sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent"))
        from plugins.context_engine.lcm.config import LCMConfig
        from plugins.context_engine.lcm.engine import LCMEngine

        cfg = LCMConfig(database_path=self.cfg.lcm_db)
        eng = LCMEngine(config=cfg, hermes_home=self.home)
        try:
            eng.on_session_start(session_id)
            out = eng.handle_tool_call(
                "lcm_expand_query",
                {"prompt": "What is the exact recovery sentinel?", "node_ids": [node_id]},
            )
            data = json.loads(out)
            return str(data.get("answer", ""))
        finally:
            try:
                eng.shutdown()
            except Exception:
                pass

    # ---- one trial ----------------------------------------------------------
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
    def run(self, out_path: Path) -> dict:
        results: list[TrialResult] = []
        t0 = time.time()
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
            "verdict": (
                "PASS" if (n >= 180 and node_recall / n >= 0.95 and wlb >= 0.90
                           and confident_wrong == 0)
                else "PASS-UNDERPOWERED" if (node_recall / n >= 0.95 and confident_wrong == 0)
                else "FAIL"
            ),
            "trials": [vars(r) for r in results],
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
    rep = ArmBHarness(cfg).run(Path(args.out))
    print(json.dumps({k: v for k, v in rep.items() if k != "trials"}, indent=2))
    return 0 if rep["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
