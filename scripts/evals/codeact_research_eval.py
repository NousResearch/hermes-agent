#!/usr/bin/env python3
"""Optional local eval for CodeAct research behavior.

This is intentionally not part of the default test suite. It runs the real
configured Hermes agent and attached model, so it depends on the user's local
LLM stack, provider config, web tooling, and network state.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import re
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_QUERIES: list[dict[str, str]] = [
    {
        "id": "medical_pharma_glp1_pipeline",
        "topic": "medical_pharma",
        "prompt": (
            "Search the web to find out information about the latest GLP-1/GIP "
            "drugs that are currently in development/testing as of May 6th, "
            "2026, and then give me a report on your findings."
        ),
    },
    {
        "id": "sports_current_starting_lineup",
        "topic": "sports",
        "prompt": (
            "Search the web to find out the current Detroit Pistons starting "
            "5 squad and cite your sources."
        ),
    },
    {
        "id": "finance_earnings_guidance",
        "topic": "finance",
        "prompt": (
            "Research Apple's latest earnings guidance and current analyst "
            "reaction. Use primary filings or investor materials where possible."
        ),
    },
    {
        "id": "technology_product_launch",
        "topic": "technology",
        "prompt": (
            "Research the latest public information about Nvidia Blackwell GPU "
            "availability and deployment status, then summarize with citations."
        ),
    },
    {
        "id": "geopolitics_current_conflict",
        "topic": "geopolitics",
        "prompt": (
            "Search for the latest reliable reporting and official statements "
            "about NATO policy discussions on Ukraine as of today. Separate "
            "official statements from analysis."
        ),
    },
    {
        "id": "shopping_product_comparison",
        "topic": "shopping",
        "prompt": (
            "Research the current best 27-inch OLED gaming monitors to buy, "
            "including pricing, professional reviews, and common complaints."
        ),
    },
    {
        "id": "gaming_patch_status",
        "topic": "gaming",
        "prompt": (
            "Research the latest patch notes and community issues for Elden "
            "Ring Nightreign. Prefer official patch notes first."
        ),
    },
    {
        "id": "social_trend_verification",
        "topic": "social_trends",
        "prompt": (
            "Research a currently trending TikTok or Reddit topic in tech, "
            "verify it with reliable sources, and explain what is known versus rumor."
        ),
    },
    {
        "id": "music_chart_report",
        "topic": "music",
        "prompt": (
            "Research the latest Billboard chart performance for Kendrick "
            "Lamar releases and summarize the source-backed chart facts."
        ),
    },
    {
        "id": "domestic_news_primary_sources",
        "topic": "domestic_news",
        "prompt": (
            "Research the latest White House domestic policy announcement and "
            "summarize it using primary documents plus independent reporting."
        ),
    },
]


@dataclass
class EvalEvent:
    kind: str
    timestamp: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    args: dict[str, Any] | None = None
    duration: float | None = None
    is_error: bool | None = None
    result_preview: str | None = None


class EvalTimeout(RuntimeError):
    pass


class Tee(io.TextIOBase):
    def __init__(self, stream, buffer: io.StringIO) -> None:
        self.stream = stream
        self.buffer = buffer

    def write(self, text: str) -> int:
        self.buffer.write(text)
        return self.stream.write(text)

    def flush(self) -> None:
        self.buffer.flush()
        self.stream.flush()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_loads_maybe(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _tool_call_name_args(tool_call: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(tool_call, dict):
        tool_call = getattr(tool_call, "__dict__", {}) or {}
    fn = tool_call.get("function") or {}
    name = fn.get("name") or tool_call.get("name") or ""
    args = _json_loads_maybe(fn.get("arguments") or tool_call.get("arguments") or {})
    return str(name or ""), args if isinstance(args, dict) else {}


def extract_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages or []):
        for tool_call in msg.get("tool_calls") or []:
            name, args = _tool_call_name_args(tool_call)
            calls.append(
                {
                    "message_index": idx,
                    "name": name,
                    "args": args,
                    "thoughts": args.get("thoughts") if isinstance(args, dict) else None,
                    "code": args.get("code") if isinstance(args, dict) else None,
                }
            )
    return calls


def extract_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages or []):
        if msg.get("role") != "tool":
            continue
        content = str(msg.get("content") or "")
        parsed = _json_loads_maybe(content)
        results.append(
            {
                "message_index": idx,
                "name": msg.get("name") or "",
                "content_preview": content[:2000],
                "parsed": parsed if isinstance(parsed, dict) else None,
            }
        )
    return results


def extract_event_tool_calls(events: list[Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for idx, event in enumerate(events or []):
        event_data = event if isinstance(event, dict) else getattr(event, "__dict__", {})
        if event_data.get("kind") != "tool_start":
            continue
        args = event_data.get("args") if isinstance(event_data.get("args"), dict) else {}
        calls.append(
            {
                "message_index": idx,
                "name": event_data.get("tool_name") or "",
                "args": args,
                "thoughts": args.get("thoughts") if isinstance(args, dict) else None,
                "code": args.get("code") if isinstance(args, dict) else None,
            }
        )
    return calls


def extract_event_tool_results(events: list[Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for idx, event in enumerate(events or []):
        event_data = event if isinstance(event, dict) else getattr(event, "__dict__", {})
        if event_data.get("kind") != "tool_complete":
            continue
        preview = str(event_data.get("result_preview") or "")
        parsed = _json_loads_maybe(preview)
        results.append(
            {
                "message_index": idx,
                "name": event_data.get("tool_name") or "",
                "content_preview": preview[:2000],
                "parsed": parsed if isinstance(parsed, dict) else None,
            }
        )
    return results


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(needle in low for needle in needles)


def analyze_run(
    *,
    query: dict[str, str],
    result: dict[str, Any],
    events: list[EvalEvent],
    stdout_text: str = "",
    log_text: str = "",
) -> dict[str, Any]:
    messages = result.get("messages") or []
    tool_calls = extract_tool_calls(messages)
    tool_results = extract_tool_results(messages)
    if not tool_calls:
        tool_calls = extract_event_tool_calls(events)
    if not tool_results:
        tool_results = extract_event_tool_results(events)
    run_code_calls = [call for call in tool_calls if call["name"] == "run_code"]
    code_blocks = [str(call.get("code") or "") for call in run_code_calls]
    first_code = code_blocks[0] if code_blocks else ""
    all_code = "\n".join(code_blocks)
    observed_code = all_code or "\n".join([stdout_text, log_text])
    final_response = str(result.get("final_response") or "")

    used_research_recipe = _contains_any(
        observed_code,
        ("research_web(", "medical_pharma_research("),
    )
    first_used_research_recipe = _contains_any(
        first_code,
        ("research_web(", "medical_pharma_research("),
    )
    used_web_search = "web_search(" in observed_code
    first_used_web_search = "web_search(" in first_code
    debugged_web_search = _contains_any(
        observed_code,
        (
            "help('web_search'",
            'help("web_search"',
            "sys.modules",
            "dir(",
            "__name__",
            "standalone function",
        ),
    )
    used_raw_curl_or_search_scrape = _contains_any(
        observed_code,
        (
            "curl",
            "duckduckgo.com/html",
            "google.com/search",
            "subprocess.run",
            "urllib.request.urlopen",
        ),
    )
    redirected_web_search = any(
        isinstance(item.get("parsed"), dict)
        and str(item["parsed"].get("redirected_from") or "") == "web_search"
        for item in tool_results
    )
    has_source_table = any(
        isinstance(item.get("parsed"), dict)
        and (
            item["parsed"].get("source_table")
            or item["parsed"].get("citation_metadata")
        )
        for item in tool_results
    )
    final_has_citations = bool(re.search(r"\[S\d+\]", final_response))
    codeact_used = bool(run_code_calls)

    score = 100
    findings: list[dict[str, str]] = []

    def add_finding(severity: str, text: str, penalty: int = 0) -> None:
        nonlocal score
        findings.append({"severity": severity, "text": text})
        score -= penalty

    if not codeact_used:
        add_finding("fail", "The run did not use CodeAct/run_code.", 35)
    if first_used_research_recipe:
        add_finding("pass", "The first CodeAct call used a research recipe.")
    elif first_used_web_search and redirected_web_search:
        add_finding(
            "warn",
            "The first CodeAct call used web_search, but it redirected to research_web.",
            8,
        )
    elif first_used_web_search:
        add_finding("warn", "The first CodeAct call used raw web_search.", 18)
    elif codeact_used:
        add_finding(
            "warn",
            "The first CodeAct call did not use research_web or web_search.",
            12,
        )

    if used_research_recipe:
        add_finding("pass", "The run used research_web/medical_pharma_research.")
    elif redirected_web_search:
        add_finding("pass", "A research-shaped web_search call redirected correctly.")
    else:
        add_finding("fail", "No research recipe or redirect was observed.", 30)

    if debugged_web_search:
        add_finding(
            "fail",
            "The model debugged web_search/namespace instead of using research tools.",
            25,
        )
    if used_raw_curl_or_search_scrape:
        add_finding(
            "fail",
            "The model fell back to raw curl/search scraping from CodeAct.",
            25,
        )
    if has_source_table:
        add_finding("pass", "A source_table/citation_metadata bundle was returned.")
    else:
        add_finding("warn", "No source_table/citation_metadata was found in tool results.", 10)
    if final_response and final_has_citations:
        add_finding("pass", "The final response used source-table citations.")
    elif final_response:
        add_finding("warn", "The final response did not include [S#] citations.", 8)

    score = max(0, min(100, score))
    verdict = "pass" if score >= 80 else "warn" if score >= 55 else "fail"

    return {
        "query": query,
        "verdict": verdict,
        "score": score,
        "findings": findings,
        "summary": {
            "completed": result.get("completed"),
            "api_calls": result.get("api_calls"),
            "message_count": len(messages),
            "tool_call_count": len(tool_calls),
            "run_code_call_count": len(run_code_calls),
            "first_codeact_code": first_code[:2000],
            "used_research_recipe": used_research_recipe,
            "first_used_research_recipe": first_used_research_recipe,
            "used_web_search": used_web_search,
            "first_used_web_search": first_used_web_search,
            "redirected_web_search": redirected_web_search,
            "debugged_web_search": debugged_web_search,
            "used_raw_curl_or_search_scrape": used_raw_curl_or_search_scrape,
            "has_source_table": has_source_table,
            "final_has_citations": final_has_citations,
            "stdout_chars": len(stdout_text),
            "log_chars": len(log_text),
        },
        "tool_sequence": [call["name"] for call in tool_calls],
        "codeact_calls": [
            {
                "index": idx + 1,
                "thoughts": call.get("thoughts"),
                "code": str(call.get("code") or "")[:4000],
            }
            for idx, call in enumerate(run_code_calls)
        ],
        "tool_results": [
            {
                "index": idx + 1,
                "name": item["name"],
                "content_preview": item["content_preview"][:1000],
            }
            for idx, item in enumerate(tool_results)
        ],
        "events": [asdict(event) for event in events],
        "final_response_preview": final_response[:4000],
    }


def choose_query(args: argparse.Namespace) -> dict[str, str]:
    if args.query:
        return {"id": "custom", "topic": args.topic or "custom", "prompt": args.query}
    if args.query_id:
        for item in DEFAULT_QUERIES:
            if item["id"] == args.query_id:
                return dict(item)
        raise SystemExit(f"Unknown --query-id {args.query_id!r}. Use --list-queries.")
    rng = random.Random(args.seed)
    return dict(rng.choice(DEFAULT_QUERIES))


def _read_positions(paths: list[Path]) -> dict[str, int]:
    positions: dict[str, int] = {}
    for path in paths:
        try:
            positions[str(path)] = path.stat().st_size
        except FileNotFoundError:
            positions[str(path)] = 0
    return positions


def _read_deltas(positions: dict[str, int], limit_chars: int = 80_000) -> dict[str, str]:
    deltas: dict[str, str] = {}
    for raw_path, pos in positions.items():
        path = Path(raw_path)
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(pos)
                text = handle.read()
        except FileNotFoundError:
            text = ""
        if len(text) > limit_chars:
            text = text[-limit_chars:]
        deltas[raw_path] = text
    return deltas


def _write_report_markdown(path: Path, analysis: dict[str, Any], artifacts: dict[str, str]) -> None:
    lines = [
        f"# CodeAct Research Local Eval: {analysis['query']['id']}",
        "",
        f"- Verdict: {analysis['verdict']}",
        f"- Score: {analysis['score']}/100",
        f"- Topic: {analysis['query'].get('topic')}",
        f"- Prompt: {analysis['query']['prompt']}",
        "",
        "## Findings",
        "",
    ]
    for item in analysis["findings"]:
        lines.append(f"- {item['severity'].upper()}: {item['text']}")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "```json",
            json.dumps(analysis["summary"], indent=2, ensure_ascii=False),
            "```",
            "",
            "## CodeAct Calls",
            "",
        ]
    )
    for call in analysis["codeact_calls"]:
        lines.extend(
            [
                f"### Call {call['index']}",
                "",
                f"Thoughts: {call.get('thoughts') or ''}",
                "",
                "```python",
                call.get("code") or "",
                "```",
                "",
            ]
        )
    lines.extend(["## Artifacts", ""])
    for label, artifact_path in artifacts.items():
        lines.append(f"- {label}: `{artifact_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_agent(args: argparse.Namespace, events: list[EvalEvent], lock: threading.Lock):
    from hermes_cli.config import load_config
    from hermes_cli.models import detect_provider_for_model
    from hermes_cli.oneshot import _normalize_toolsets
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.tools_config import _get_platform_tools
    from run_agent import AIAgent

    cfg = load_config()
    model_cfg = cfg.get("model") or {}
    if isinstance(model_cfg, str):
        cfg_model = model_cfg
        cfg_provider = ""
    else:
        cfg_model = model_cfg.get("default") or model_cfg.get("model") or ""
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()

    env_model = os.getenv("HERMES_INFERENCE_MODEL", "").strip()
    effective_model = (args.model or "").strip() or env_model or cfg_model
    effective_provider = (args.provider or "").strip() or None
    explicit_base_url_from_alias = None

    if effective_provider is None and (args.model or env_model):
        explicit_model = (args.model or "").strip() or env_model
        if explicit_model:
            try:
                from hermes_cli import model_switch as _ms

                _ms._ensure_direct_aliases()
                direct = _ms.DIRECT_ALIASES.get(explicit_model.strip().lower())
            except Exception:
                direct = None
            if direct is not None:
                effective_model = direct.model
                effective_provider = direct.provider
                if direct.base_url:
                    explicit_base_url_from_alias = direct.base_url.rstrip("/")
            else:
                current_provider = (
                    cfg_provider
                    or os.getenv("HERMES_INFERENCE_PROVIDER", "").strip().lower()
                    or "auto"
                )
                detected = detect_provider_for_model(explicit_model, current_provider)
                if detected:
                    effective_provider, effective_model = detected

    runtime = resolve_runtime_provider(
        requested=effective_provider,
        target_model=effective_model or None,
        explicit_base_url=explicit_base_url_from_alias,
    )

    toolsets_list = _normalize_toolsets(args.toolsets)
    if toolsets_list is None and not args.no_config_toolsets:
        toolsets_list = sorted(_get_platform_tools(cfg, "cli"))

    os.environ.setdefault("HERMES_YOLO_MODE", "1")
    os.environ.setdefault("HERMES_ACCEPT_HOOKS", "1")

    agent = AIAgent(
        api_key=runtime.get("api_key"),
        base_url=runtime.get("base_url"),
        provider=runtime.get("provider"),
        api_mode=runtime.get("api_mode"),
        model=effective_model,
        max_iterations=args.max_iterations,
        enabled_toolsets=toolsets_list,
        quiet_mode=args.quiet_agent,
        platform="cli",
        credential_pool=runtime.get("credential_pool"),
        provider_headers=runtime.get("headers"),
    )

    def on_start(tool_call_id: str, name: str, call_args: dict[str, Any]) -> None:
        with lock:
            events.append(
                EvalEvent(
                    kind="tool_start",
                    timestamp=_utc_now(),
                    tool_call_id=tool_call_id,
                    tool_name=name,
                    args=call_args,
                )
            )

    def on_complete(
        tool_call_id: str,
        name: str,
        call_args: dict[str, Any],
        result: str,
    ) -> None:
        with lock:
            events.append(
                EvalEvent(
                    kind="tool_complete",
                    timestamp=_utc_now(),
                    tool_call_id=tool_call_id,
                    tool_name=name,
                    args=call_args,
                    result_preview=str(result or "")[:2000],
                )
            )

    def on_progress(event: str, name: str, preview: Any, call_args: Any, **kwargs) -> None:
        with lock:
            events.append(
                EvalEvent(
                    kind=event,
                    timestamp=_utc_now(),
                    tool_name=name,
                    args=call_args if isinstance(call_args, dict) else None,
                    duration=kwargs.get("duration"),
                    is_error=kwargs.get("is_error"),
                    result_preview=str(preview or "")[:1000] if preview else None,
                )
            )

    agent.tool_start_callback = on_start
    agent.tool_complete_callback = on_complete
    agent.tool_progress_callback = on_progress
    return agent


def run_live_eval(args: argparse.Namespace) -> int:
    from hermes_constants import get_hermes_home
    from hermes_cli.env_loader import load_hermes_dotenv
    from hermes_logging import setup_logging

    load_hermes_dotenv(hermes_home=get_hermes_home(), project_env=REPO_ROOT / ".env")
    setup_logging(mode="cli")

    query = choose_query(args)
    if args.dry_run:
        print(json.dumps(query, indent=2, ensure_ascii=False))
        return 0

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or get_hermes_home() / "evals" / "codeact_research")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f"{run_id}_{query['id']}"
    run_dir.mkdir(parents=True, exist_ok=True)

    events: list[EvalEvent] = []
    lock = threading.Lock()
    agent = _build_agent(args, events, lock)

    log_paths = [
        get_hermes_home() / "logs" / "agent.log",
        get_hermes_home() / "logs" / "errors.log",
    ]
    log_positions = _read_positions(log_paths)

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    old_handler = None
    if args.timeout_seconds > 0 and hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(
            signal.SIGALRM,
            lambda _signum, _frame: (_ for _ in ()).throw(
                EvalTimeout(f"eval timed out after {args.timeout_seconds}s")
            ),
        )
        signal.alarm(args.timeout_seconds)

    started = time.monotonic()
    result: dict[str, Any]
    try:
        with contextlib.redirect_stdout(
            Tee(sys.stdout, stdout_buffer)
        ), contextlib.redirect_stderr(Tee(sys.stderr, stderr_buffer)):
            print(f"[eval] query_id={query['id']} topic={query.get('topic')}")
            print(f"[eval] prompt={query['prompt']}")
            result = agent.run_conversation(query["prompt"])
    except EvalTimeout as exc:
        result = {
            "completed": False,
            "api_calls": None,
            "messages": [],
            "final_response": "",
            "error": str(exc),
        }
    finally:
        if args.timeout_seconds > 0 and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    elapsed = time.monotonic() - started
    with lock:
        event_snapshot = list(events)
    log_deltas = _read_deltas(log_positions)
    stdout_text = stdout_buffer.getvalue()
    stderr_text = stderr_buffer.getvalue()
    log_text = "\n".join(log_deltas.values())

    analysis = analyze_run(
        query=query,
        result=result,
        events=event_snapshot,
        stdout_text=stdout_text + stderr_text,
        log_text=log_text,
    )
    analysis["summary"]["elapsed_seconds"] = round(elapsed, 2)
    if result.get("error"):
        analysis["error"] = result["error"]

    artifacts = {
        "analysis_json": str(run_dir / "analysis.json"),
        "report_md": str(run_dir / "report.md"),
        "stdout_log": str(run_dir / "stdout.log"),
        "stderr_log": str(run_dir / "stderr.log"),
        "agent_log_delta": str(run_dir / "agent.log.delta"),
        "errors_log_delta": str(run_dir / "errors.log.delta"),
        "messages_json": str(run_dir / "messages.json"),
    }

    Path(artifacts["analysis_json"]).write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    Path(artifacts["stdout_log"]).write_text(stdout_text, encoding="utf-8")
    Path(artifacts["stderr_log"]).write_text(stderr_text, encoding="utf-8")
    Path(artifacts["messages_json"]).write_text(
        json.dumps(
            result.get("messages") or [],
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    for raw_path, text in log_deltas.items():
        name = (
            "errors_log_delta"
            if raw_path.endswith("errors.log")
            else "agent_log_delta"
        )
        Path(artifacts[name]).write_text(text, encoding="utf-8")
    _write_report_markdown(Path(artifacts["report_md"]), analysis, artifacts)

    print("")
    print("CodeAct research local eval")
    print(f"  query:   {query['id']} ({query.get('topic')})")
    print(f"  verdict: {analysis['verdict']}  score: {analysis['score']}/100")
    print(f"  report:  {artifacts['report_md']}")
    print(f"  json:    {artifacts['analysis_json']}")
    for finding in analysis["findings"]:
        print(f"  - {finding['severity'].upper()}: {finding['text']}")

    if args.fail_on_bad and analysis["verdict"] == "fail":
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optional local eval for Hermes CodeAct research behavior.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query", help="Custom prompt to evaluate.")
    parser.add_argument("--topic", help="Topic label for --query.")
    parser.add_argument("--query-id", help="Use a specific built-in query id.")
    parser.add_argument(
        "--list-queries", action="store_true", help="List built-in query ids."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for query selection."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print selected query and exit.")
    parser.add_argument("--model", help="Model override.")
    parser.add_argument("--provider", help="Provider override.")
    parser.add_argument("--toolsets", help="Comma-separated toolset override.")
    parser.add_argument(
        "--no-config-toolsets",
        action="store_true",
        help="When --toolsets is omitted, do not load configured CLI toolsets.",
    )
    parser.add_argument("--max-iterations", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--quiet-agent", action="store_true", help="Suppress agent progress output."
    )
    parser.add_argument("--output-dir", help="Directory for eval artifacts.")
    parser.add_argument(
        "--fail-on-bad", action="store_true", help="Exit 1 when verdict is fail."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_queries:
        for item in DEFAULT_QUERIES:
            print(f"{item['id']}\t{item['topic']}\t{item['prompt']}")
        return 0
    return run_live_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
