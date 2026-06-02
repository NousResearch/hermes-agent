#!/usr/bin/env python3
"""Create or update the bilingual Feishu doc for one paper (registry-aware)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from build_bilingual_doc_md import build  # noqa: E402
from paper_doc_registry import canonical_paper_id, lookup, register, resolve  # noqa: E402
from paper_doc_title import feishu_doc_title, load_handoff, resolve_title_zh  # noqa: E402
from paper_nexus_metadata import resolve_and_fetch  # noqa: E402


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def workspace_root(board: str = "paper-nexus") -> Path:
    return hermes_home() / "kanban" / "boards" / board / "workspaces"


def _stage_key_from_handoff(data: dict) -> str | None:
    stage = str(data.get("stage") or "").strip().upper()
    if data.get("audit_scores") or data.get("recommended_actions") or data.get("verdict"):
        return "T4"
    if stage in {"T0", "T1", "T2", "T3", "T4"}:
        return stage
    if data.get("thesis") and (data.get("reading_map_sections") or data.get("sections")):
        return "T0"
    if (
        data.get("claims_summary")
        or data.get("claims")
        or data.get("cel_row_count")
        or data.get("claims_covered")
        or data.get("key_figures_cited")
    ):
        return "T1"
    if (
        data.get("task") == "method-and-reproduction"
        or data.get("reproduction_bottlenecks")
        or data.get("architecture")
        or data.get("training")
        or data.get("inference")
        or data.get("key_formulas")
        or data.get("artifact") == "method-and-reproduction.md"
    ):
        return "T2"
    if (
        data.get("deliverable") == "benchmark-and-open-source-map.md"
        or data.get("model_family")
        or data.get("benchmark_map")
        or data.get("open_source_resources")
        or data.get("top_3_recommendations")
        or data.get("comparison_models")
        or data.get("top_3_relevant_works")
        or data.get("open_source_status")
    ):
        return "T3"
    if data.get("audit_type"):
        return "T4"
    return None


def _candidate_ids(meta: dict) -> set[str]:
    def _norm(text: str) -> str:
        value = str(text or "").strip()
        if value.lower().startswith("arxiv:"):
            value = value.split(":", 1)[1].strip()
        return canonical_paper_id(value)

    ids = {
        str(meta.get("canonical_id") or "").strip(),
        _norm(str(meta.get("canonical_id") or "").strip()),
        str(meta.get("paper_id") or "").strip(),
        _norm(str(meta.get("paper_id") or "").strip()),
    }
    return {item for item in ids if item}


def collect_stage_handoffs(
    meta: dict,
    *,
    board: str = "paper-nexus",
    handoff_path: str | None = None,
) -> dict:
    roots = []
    if handoff_path:
        hp = Path(handoff_path).expanduser().resolve()
        if hp.parent.name:
            roots.append(hp.parent.parent if hp.parent.parent.name == "workspaces" else hp.parent)
    roots.append(workspace_root(board))

    candidates = _candidate_ids(meta)
    found: dict[str, dict] = {}
    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.glob("*/handoff.json"):
            if path in seen:
                continue
            seen.add(path)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            stage = _stage_key_from_handoff(data)
            if not stage or stage in found:
                continue
            pid = str(data.get("canonical_id") or data.get("paper_id") or "").strip()
            if not pid:
                continue
            norm_pid = pid.split(":", 1)[1].strip() if pid.lower().startswith("arxiv:") else pid
            if pid in candidates or canonical_paper_id(norm_pid) in candidates:
                found[stage] = data
    return found


def _patch_drive_doc_title(document_id: str, online_title: str) -> None:
    """Set Feishu Drive listing name (and doc <title> when still Untitled)."""
    if not document_id or not online_title:
        return
    _run_lark(
        [
            "drive",
            "files",
            "patch",
            "--as",
            "bot",
            "--params",
            json.dumps({"file_token": document_id, "type": "docx"}),
            "--data",
            json.dumps({"new_title": online_title[:200]}),
        ],
    )


def _created_doc_fields(out: dict) -> tuple[str, str | None]:
    data = out.get("data", {}) if isinstance(out, dict) else {}
    doc = data.get("document", {}) if isinstance(data, dict) else {}
    doc_url = (
        doc.get("url")
        or data.get("doc_url")
        or data.get("url")
        or ""
    )
    document_id = (
        doc.get("document_id")
        or data.get("doc_id")
        or data.get("document_id")
    )
    return doc_url, document_id


def _run_lark(args: list[str], content: str | None = None) -> dict:
    cmd = ["lark-cli", *args]
    proc = subprocess.run(
        cmd,
        input=content,
        capture_output=True,
        text=True,
        timeout=120,
    )
    raw = proc.stdout or proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"lark-cli failed ({proc.returncode}): {raw[:2000]}")
    start = raw.find("{")
    if start < 0:
        raise RuntimeError(f"lark-cli non-JSON output: {raw[:500]}")
    return json.loads(raw[start:])


def sync_paper_doc(
    paper_id: str,
    marker: str = "",
    board: str = "paper-nexus",
    *,
    handoff_path: str | None = None,
    title_zh: str | None = None,
) -> dict:
    meta = resolve_and_fetch(paper_id)
    meta["canonical_id"] = canonical_paper_id(
        meta.get("canonical_id") or meta["paper_id"]
    )
    handoff = load_handoff(handoff_path)
    stage_handoffs = collect_stage_handoffs(
        meta,
        board=board,
        handoff_path=handoff_path,
    )
    reg_entry = lookup(meta["canonical_id"], board)
    meta["title_zh"] = resolve_title_zh(
        meta,
        handoff=handoff,
        title_zh=title_zh,
        registry_entry=reg_entry,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as tf:
        tf.write(build(meta, marker, handoffs=stage_handoffs))
        md_path = tf.name

    body = Path(md_path).read_text(encoding="utf-8")
    Path(md_path).unlink(missing_ok=True)

    online_title = feishu_doc_title(meta["canonical_id"], meta["title_zh"])
    plan = resolve(meta["canonical_id"], board)
    if plan["action"] == "update":
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        body = f"\n\n---\n\n## 流水线更新 / Pipeline refresh · {stamp}\n\n{body}"
        _run_lark(
            [
                "docs",
                "+update",
                "--api-version",
                "v2",
                "--as",
                "bot",
                "--doc",
                plan["doc_url"],
                "--command",
                "append",
                "--doc-format",
                "markdown",
                "--new-title",
                online_title,
                "--content",
                body,
            ],
        )
        doc_url = plan["doc_url"]
        document_id = plan.get("document_id")
        action = "update"
    else:
        out = _run_lark(
            [
                "docs",
                "+create",
                "--api-version",
                "v2",
                "--as",
                "bot",
                "--title",
                online_title,
                "--markdown",
                body,
            ],
        )
        doc_url, document_id = _created_doc_fields(out)
        if not doc_url:
            raise RuntimeError(f"create returned no url: {out}")
        action = "create"

    if document_id:
        _patch_drive_doc_title(document_id, online_title)

    register(
        meta["canonical_id"],
        doc_url,
        document_id=document_id,
        title=meta["title"],
        title_zh=meta["title_zh"],
        board=board,
    )
    wf = f"paper-nexus:{meta['canonical_id']}"
    return {
        "action": action,
        "canonical_id": meta["canonical_id"],
        "paper_id": meta["paper_id"],
        "doc_url": doc_url,
        "document_id": document_id,
        "title": meta["title"],
        "title_zh": meta["title_zh"],
        "feishu_doc_title": feishu_doc_title(meta["canonical_id"], meta["title_zh"]),
        "memory_os": {
            "workflow_id": wf,
            "store_hint": (
                f"After T5/T6 call store_memory_markdown with workflow_id={wf}; "
                "generate entry via paper_memory_markdown.py"
            ),
        },
        "stage_handoffs": sorted(stage_handoffs.keys()),
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: paper_feishu_doc_sync.py <arxiv_or_s2_url> [marker]", file=sys.stderr)
        return 2
    import argparse as _ap

    ap = _ap.ArgumentParser()
    ap.add_argument("paper_id")
    ap.add_argument("marker", nargs="?", default="")
    ap.add_argument("--handoff", default="")
    ap.add_argument("--title-zh", default="")
    ns = ap.parse_args()
    result = sync_paper_doc(
        ns.paper_id,
        marker=ns.marker,
        handoff_path=ns.handoff or None,
        title_zh=ns.title_zh or None,
    )
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
