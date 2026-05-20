#!/usr/bin/env python3
"""Create or update the bilingual Feishu doc for one paper (registry-aware)."""

from __future__ import annotations

import json
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
    reg_entry = lookup(meta["canonical_id"], board)
    meta["title_zh"] = resolve_title_zh(
        meta,
        handoff=handoff,
        title_zh=title_zh,
        registry_entry=reg_entry,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as tf:
        tf.write(build(meta, marker))
        md_path = tf.name

    body = Path(md_path).read_text(encoding="utf-8")
    Path(md_path).unlink(missing_ok=True)

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
                "--content",
                body,
            ],
        )
        doc_url = plan["doc_url"]
        document_id = plan.get("document_id")
        action = "update"
    else:
        online_title = feishu_doc_title(meta["canonical_id"], meta["title_zh"])
        wrapped = f"# {online_title}\n\n{body}"
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
                "--doc-format",
                "markdown",
                "--markdown",
                wrapped,
            ],
        )
        doc = out.get("data", {}).get("document", {})
        doc_url = doc.get("url", "")
        document_id = doc.get("document_id")
        if not doc_url:
            raise RuntimeError(f"create returned no url: {out}")
        action = "create"

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
