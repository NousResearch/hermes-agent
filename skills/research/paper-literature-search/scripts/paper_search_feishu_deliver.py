#!/usr/bin/env python3
"""Format ranked papers and send Feishu IM via lark-cli (no online doc)."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))
_LIVE = _DIR.parents[2] / "devops" / "kanban-feishu-live" / "scripts"
if str(_LIVE) not in sys.path:
    sys.path.insert(0, str(_LIVE))

from paper_search_rank import run_search  # noqa: E402

_EN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "into", "is", "of", "on", "or", "over", "the", "their", "this",
    "to", "toward", "under", "using", "via", "with",
}

_TITLE_GLOSSARY = [
    ("social recommender systems", "社交推荐系统"),
    ("recommender systems", "推荐系统"),
    ("intelligent transportation systems", "智能交通系统"),
    ("traffic forecasting", "交通预测"),
    ("time series", "时间序列"),
    ("medical licensing examination", "医学执照考试"),
    ("health information seeking", "健康信息获取"),
    ("large language models", "大语言模型"),
    ("large language model", "大语言模型"),
    ("graph neural networks", "图神经网络"),
    ("graph neural network", "图神经网络"),
    ("multimodal", "多模态"),
    ("healthcare", "医疗健康"),
    ("medical", "医疗"),
    ("medicine", "医学"),
    ("education", "教育"),
    ("research", "研究"),
    ("practice", "实践"),
    ("survey", "综述"),
    ("review", "综述"),
    ("comprehensive", "全面"),
    ("systematic", "系统性"),
    ("benchmark", "基准"),
    ("evaluation", "评估"),
    ("performance", "性能"),
    ("recommendation", "推荐"),
    ("recommendations", "推荐"),
    ("therapy", "治疗"),
    ("clinical", "临床"),
    ("application", "应用"),
    ("applications", "应用"),
    ("online", "在线"),
    ("health", "健康"),
    ("information", "信息"),
    ("examination", "考试"),
    ("models", "模型"),
    ("model", "模型"),
]

_GENERIC_ZH_TERMS = {"综述", "全面", "系统性", "研究", "实践", "应用", "评估", "性能"}


def _slug(query: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "-", query.strip())[:40].strip("-")
    return s or "query"


def _trim(t: str, n: int = 100) -> str:
    t = " ".join((t or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def _replace_case_insensitive(text: str, src: str, dst: str) -> str:
    return re.sub(re.escape(src), dst, text, flags=re.I)


def _extract_keywords(title: str, abstract: str = "", limit: int = 3) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()
    for source in (title, abstract):
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+.\-]{2,}", source or ""):
            low = token.lower()
            if low in _EN_STOPWORDS or low in seen:
                continue
            seen.add(low)
            ranked.append(token)
            if len(ranked) >= limit:
                return ranked
    return ranked


def _paper_kind(title: str, abstract: str = "") -> tuple[str, str]:
    text = f"{title} {abstract}".lower()
    if any(x in text for x in ("survey", "review")):
        return "综述", "survey"
    if any(x in text for x in ("benchmark", "evaluation", "performance")):
        return "评测", "evaluation"
    if any(x in text for x in ("application", "utility", "practice")):
        return "应用", "application"
    if any(x in text for x in ("recommendation", "therapy", "clinical")):
        return "临床研究", "clinical study"
    return "研究", "study"


def _zh_keywords(title: str, abstract: str = "", limit: int = 3) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    lower_title = f"{title} {abstract}".lower()
    for src, dst in _TITLE_GLOSSARY:
        if dst in _GENERIC_ZH_TERMS:
            continue
        if src in lower_title and dst.lower() not in seen:
            seen.add(dst.lower())
            items.append(dst)
            if len(items) >= limit:
                return items
    raw = _extract_keywords(title, abstract, limit=limit + 3)
    for token in raw:
        zh = token
        if zh == token and re.fullmatch(r"(llm|gpt|bert|gnn|rlhf|ai|ml)", token, re.I):
            zh = token.upper()
        if zh == token and token.isupper():
            zh = token
        if zh == token:
            continue
        if zh.lower() in seen:
            continue
        seen.add(zh.lower())
        items.append(zh)
        if len(items) >= limit:
            break
    return items


def _cn_title_gloss(title: str, abstract: str = "") -> str:
    zh_terms = _zh_keywords(title, abstract, limit=3)
    kind_zh, _ = _paper_kind(title, abstract)
    if zh_terms:
        lead = " / ".join(zh_terms[:2])
        if kind_zh == "综述":
            return _trim(f"{lead}综述", 42)
        if kind_zh == "评测":
            return _trim(f"{lead}评测", 42)
        return _trim(f"{lead}{kind_zh}", 42)
    gloss = title
    for src, dst in _TITLE_GLOSSARY:
        gloss = _replace_case_insensitive(gloss, src, dst)
    gloss = re.sub(r"\s+", " ", gloss).strip(" .:-")
    gloss = re.sub(r"\b(A|An|The)\b\s*", "", gloss, flags=re.I).strip()
    if re.search(r"[\u4e00-\u9fff]", gloss):
        return _trim(gloss, 42)
    return _trim(title, 42)


def _signal_pair(paper: dict) -> tuple[str, str]:
    signals_zh: list[str] = []
    signals_en: list[str] = []
    year = int(paper.get("year") or 0)
    if (paper.get("citation_count") or 0) >= 500:
        signals_zh.append("高被引")
        signals_en.append("highly cited")
    elif (paper.get("citation_count") or 0) >= 100:
        signals_zh.append("较高被引")
        signals_en.append("well cited")
    if year >= 2024:
        signals_zh.append("近作")
        signals_en.append("recent")
    if paper.get("pdf_url") or paper.get("arxiv_abs"):
        signals_zh.append("可下载")
        signals_en.append("open access")
    return " / ".join(signals_zh) or "候选文献", ", ".join(signals_en) or "candidate paper"


def _cn_outline(paper: dict) -> str:
    kind_zh, _ = _paper_kind(paper.get("title", ""), paper.get("abstract", ""))
    focus = " / ".join(_zh_keywords(paper.get("title", ""), paper.get("abstract", ""), limit=3))
    signal_zh, _ = _signal_pair(paper)
    bits = [kind_zh]
    if focus:
        bits.append(f"主题：{focus}")
    bits.append(f"信号：{signal_zh}")
    return _trim("；".join(bits), 72)


def _en_outline(paper: dict) -> str:
    _, kind_en = _paper_kind(paper.get("title", ""), paper.get("abstract", ""))
    focus = ", ".join(_extract_keywords(paper.get("title", ""), paper.get("abstract", ""), limit=3))
    _, signal_en = _signal_pair(paper)
    bits = [kind_en]
    if focus:
        bits.append(f"focus: {focus}")
    bits.append(f"signal: {signal_en}")
    return _trim("; ".join(bits), 88)


def format_top_list(result: dict) -> str:
    papers = result["papers"]
    rich_papers = papers[:5]
    rest_papers = papers[5:]
    lines = [
        f"📚 文献检索结果 · {_trim(result['query'], 48)}",
        f"权重：相关性35% + 引用30% + 高影响引用15% + 时效15% + 可下载5%",
        f"候选 {result['candidate_count']} 篇 → Top {len(papers)}",
        f"IM 导读：前 {len(rich_papers)} 篇双语摘要，剩余候选简表",
        "",
    ]
    if not papers:
        lines.extend([
            "未检索到同时满足当前条件的候选论文。",
            "建议：先单搜核心别名（如 Yi-Pi / Pi-0.5 / Pi-1），再逐步加 LLM / VLM / edge deployment 限定。",
            "若你找的是厂商发布、产品页或技术博客，请改走 web / 新闻源，而不是论文库。",
            "",
            "精读某篇：/kanban-paper-nexus <arXiv id>",
        ])
        return "\n".join(lines)
    for i, p in enumerate(rich_papers, 1):
        sc = p.get("scores", {})
        cite = p.get("citation_count") or 0
        infl = p.get("influential_citation_count") or 0
        year = p.get("year") or "?"
        link = p.get("arxiv_abs") or p.get("url") or ""
        lines.append(f"{i}. [{sc.get('display', '?')}] {_cn_title_gloss(p.get('title', ''), p.get('abstract', ''))} ({year})")
        lines.append(f"   EN: {_trim(p.get('title', ''), 96)}")
        lines.append(f"   中纲: {_cn_outline(p)}")
        lines.append(f"   EN note: {_en_outline(p)}")
        lines.append(f"   引用 {cite} / 高影响 {infl} | {link}")
        if p.get("venue"):
            lines.append(f"   venue: {_trim(p['venue'], 40)}")
        lines.append("")
    if rest_papers:
        lines.append(f"延伸候选 {len(rest_papers)} 篇（简表）")
        for i, p in enumerate(rest_papers, len(rich_papers) + 1):
            sc = p.get("scores", {})
            year = p.get("year") or "?"
            lines.append(f"{i}. [{sc.get('display', '?')}] {_trim(p.get('title', ''), 84)} ({year})")
    lines.append("")
    lines.append("精读某篇：/kanban-paper-nexus <arXiv id>")
    return "\n".join(lines)


def build_delivery_summary(query: str, result: dict) -> dict:
    delivered = len(result.get("papers") or [])
    no_hit = delivered == 0
    return {
        "ok": True,
        "query": query,
        "candidate_count": int(result.get("candidate_count") or 0),
        "delivered": delivered,
        "no_hit": no_hit,
        "next_action": "stop_no_manual_fallback" if no_hit else "use_kanban_for_deep_read",
    }


def send_text(chat_id: str, text: str, *, dry_run: bool = False) -> dict:
    if dry_run:
        return {"dry_run": True, "chars": len(text)}
    proc = subprocess.run(
        [
            "lark-cli",
            "im",
            "+messages-send",
            "--as",
            "bot",
            "--chat-id",
            chat_id,
            "--text",
            text,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"lark-cli failed: {err}")
    raw = proc.stdout.strip()
    start = raw.find("{")
    if start >= 0:
        return json.loads(raw[start:])
    return {"ok": True}


def deliver_progress(
    chat_id: str,
    query: str,
    *,
    board: str = "paper-search",
    event: str,
    stage: str = "",
    summary: str = "",
    dry_run: bool = False,
) -> None:
    """Use kanban-feishu-live notify when session exists; else direct send."""
    slug = _slug(query)
    script = _LIVE / "kanban_feishu_stage_notify.py"
    if script.is_file():
        args = [
            sys.executable,
            str(script),
            "--board",
            board,
            "notify",
            "--entity-id",
            slug,
            "--event",
            event,
        ]
        if stage:
            args.extend(["--stage", stage])
        if summary:
            args.extend(["--summary", summary])
        if dry_run:
            args.append("--dry-run")
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            return
    # fallback one-liner
    msg = f"📚 paper-search · {query}\n{event} {stage}\n{summary}"
    send_text(chat_id, msg, dry_run=dry_run)


def init_session(
    chat_id: str,
    query: str,
    result: dict,
    *,
    board: str = "paper-search",
    dry_run: bool = False,
) -> None:
    slug = _slug(query)
    script = _LIVE / "kanban_feishu_stage_notify.py"
    if not script.is_file() or dry_run:
        return
    tasks = json.dumps({"T0": "search", "T1": "rank", "T2": "deliver"})
    # meta via env not supported; title carries query text
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--board",
            board,
            "init",
            slug,
            "--chat-id",
            chat_id,
            "--title",
            _trim(query, 60),
            "--tasks-inline",
            tasks,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--chat-id", default=os.environ.get("HERMES_SESSION_CHAT_ID", ""))
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--profile", default="ml")
    ap.add_argument("--boost-recency", action="store_true")
    ap.add_argument("--min-citations", type=int, default=0)
    ap.add_argument("--year-floor", type=int, default=0, help="exclude papers before this year")
    ap.add_argument("--json-in", help="skip search, use ranked json file")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    chat_id = (args.chat_id or "").strip()
    if not chat_id and not args.dry_run:
        print("chat-id required", file=sys.stderr)
        return 2

    if args.json_in:
        result = json.loads(Path(args.json_in).read_text(encoding="utf-8"))
    else:
        yf = args.year_floor if args.year_floor > 0 else None
        result = run_search(
            args.query,
            top=args.top,
            profile=args.profile,
            boost_recency=args.boost_recency,
            min_citations=args.min_citations,
            year_floor=yf,
        )

    if not args.dry_run:
        init_session(chat_id, args.query, result)

    deliver_progress(
        chat_id,
        args.query,
        event="pipeline_started",
        summary=f"检索式：{args.query}",
        dry_run=args.dry_run,
    )
    deliver_progress(
        chat_id,
        args.query,
        event="stage_done",
        stage="T0",
        summary=f"候选 {result['candidate_count']} 篇（S2+arXiv）",
        dry_run=args.dry_run,
    )
    deliver_progress(
        chat_id,
        args.query,
        event="stage_done",
        stage="T1",
        summary=f"已排序 Top {len(result['papers'])}",
        dry_run=args.dry_run,
    )

    body = format_top_list(result)
    if args.dry_run:
        print(body)
        return 0

    deliver_progress(
        chat_id,
        args.query,
        event="pipeline_done",
        summary=f"Top {len(result['papers'])} 已列出",
        dry_run=False,
    )
    send_text(chat_id, body)
    json.dump(build_delivery_summary(args.query, result), sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
