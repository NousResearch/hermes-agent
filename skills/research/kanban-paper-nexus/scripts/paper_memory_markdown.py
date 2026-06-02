#!/usr/bin/env python3
"""Emit a store_memory_markdown entry for kanban-paper-nexus handoffs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from paper_doc_registry import canonical_paper_id  # noqa: E402

_STAGE_SCORE = {
    "orchestrator": 0.55,
    "T0": 0.6,
    "T1": 0.75,
    "T2": 0.7,
    "T3": 0.65,
    "T4": 0.7,
    "T5": 0.85,
    "T6": 0.9,
}


def _workflow_id(handoff: dict) -> str:
    cid = handoff.get("canonical_id") or canonical_paper_id(
        handoff.get("paper_id", "")
    )
    return f"paper-nexus:{cid}"


def _body_for_stage(stage: str, handoff: dict) -> str:
    cid = handoff.get("canonical_id") or canonical_paper_id(
        handoff.get("paper_id", "unknown")
    )
    lines = [
        f"# [{stage}] paper-nexus {cid}",
        f"- paper_id: {handoff.get('paper_id', '')}",
        f"- canonical_id: {cid}",
    ]
    if handoff.get("thesis_one_liner"):
        lines.append(f"- thesis: {handoff['thesis_one_liner']}")
    if handoff.get("reading_map"):
        lines.append(f"- reading_map: {handoff['reading_map']}")
    if handoff.get("feishu_doc_url"):
        lines.append(f"- feishu_doc: {handoff['feishu_doc_url']}")
    if handoff.get("feishu_doc_action"):
        lines.append(f"- feishu_doc_action: {handoff['feishu_doc_action']}")
    if handoff.get("arxiv_abs"):
        lines.append(f"- arxiv: {handoff['arxiv_abs']}")
    if handoff.get("kanban_task_ids"):
        lines.append(f"- task_ids: {', '.join(handoff['kanban_task_ids'])}")
    if handoff.get("kanban_task"):
        lines.append(f"- kanban_task: {handoff['kanban_task']}")
    if handoff.get("claims"):
        lines.append("- claims:")
        lines.append("```json")
        lines.append(json.dumps(handoff["claims"], ensure_ascii=False, indent=2))
        lines.append("```")
    if handoff.get("experiment_audit"):
        lines.append("- experiment_audit:")
        lines.append("```json")
        lines.append(
            json.dumps(handoff["experiment_audit"], ensure_ascii=False, indent=2)
        )
        lines.append("```")
    if handoff.get("qa_pass") is not None:
        lines.append(f"- qa_pass: {handoff['qa_pass']}")
    if handoff.get("recommendation_zh"):
        lines.append(f"- recommendation: {handoff['recommendation_zh']}")
    if handoff.get("summary_zh"):
        lines.append(f"- summary_zh: {handoff['summary_zh']}")
    if handoff.get("notes"):
        lines.append(f"- notes: {handoff['notes']}")
    return "\n".join(lines) + "\n"


def build_entry(
    stage: str,
    handoff: dict,
    *,
    session_id: str | None = None,
    task_id: str | None = None,
    agent_name: str = "hermes-agent",
) -> str:
    stage_key = stage.upper() if stage.upper().startswith("T") else stage
    cid = handoff.get("canonical_id") or canonical_paper_id(
        handoff.get("paper_id", "unknown")
    )
    score = _STAGE_SCORE.get(stage_key, _STAGE_SCORE.get(stage, 0.65))
    sid = (
        session_id
        or os.environ.get("HERMES_SESSION_ID")
        or f"paper-nexus-{cid}"
    )
    title = handoff.get("memory_title") or f"[paper-nexus] {cid} {stage_key}"
    tags = [
        "paper",
        "arxiv",
        "kanban-paper-nexus",
        cid,
        stage_key.lower(),
    ]
    header = [
        f"session_id: {sid}",
        f"agent_name: {agent_name}",
        f"title: {title}",
        f"workflow_id: {_workflow_id(handoff)}",
        f"tags: {', '.join(tags)}",
        f"importance_score: {score}",
    ]
    if task_id:
        header.append(f"request_id: {task_id}")
    return "\n".join(header) + "\n\n" + _body_for_stage(stage_key, handoff)


def main() -> int:
    p = argparse.ArgumentParser(description="Build store_memory_markdown entry")
    p.add_argument("--stage", required=True, help="orchestrator|T0..T6")
    p.add_argument("--handoff", required=True, help="path to handoff.json")
    p.add_argument("--session-id", default=None)
    p.add_argument("--task-id", default=None)
    p.add_argument("--agent-name", default="hermes-agent")
    args = p.parse_args()
    handoff = json.loads(Path(args.handoff).read_text(encoding="utf-8"))
    sys.stdout.write(
        build_entry(
            args.stage,
            handoff,
            session_id=args.session_id,
            task_id=args.task_id,
            agent_name=args.agent_name,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
