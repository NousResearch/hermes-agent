"""
Hermes CLI trace command — list, show, analyze, export trace data.

Usage:
    hermes trace list [--limit N]
    hermes trace show <trace_id>
    hermes trace analyze <trace_id>
    hermes trace export <trace_id> [--format json|mermaid]
"""

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

# Load trace module from ~/.hermes/trace/ (not on Python path)
_HERMES_TRACE_PATH = Path.home() / ".hermes" / "trace"
_spec = importlib.util.spec_from_file_location("hermes_trace", _HERMES_TRACE_PATH / "__init__.py")
_hermes_trace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hermes_trace)

list_traces = _hermes_trace.list_traces
get_trace = _hermes_trace.get_trace
analyze_trace = _hermes_trace.analyze_trace


def _relative_time(ts: str) -> str:
    """Format ISO timestamp as relative time."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        now = datetime.now(dt.tzinfo)
        delta = (now - dt).total_seconds()
        if delta < 60:
            return "just now"
        if delta < 3600:
            return f"{int(delta / 60)}m ago"
        if delta < 86400:
            return f"{int(delta / 3600)}h ago"
        if delta < 172800:
            return "yesterday"
        if delta < 604800:
            return f"{int(delta / 86400)}d ago"
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ts


def _format_duration(ms: float) -> str:
    """Format milliseconds as human-readable duration."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    if ms < 60000:
        return f"{ms/1000:.1f}s"
    return f"{ms/60000:.1f}m"


def _resolve_trace_id(trace_id: str) -> str | None:
    """Resolve a trace ID (full or short prefix) to full ID."""
    traces = list_traces(limit=1000)
    # First try exact match
    for t in traces:
        if t["trace_id"] == trace_id:
            return trace_id
        if t["trace_id"].startswith(trace_id):
            return t["trace_id"]
    return None


def _print_trace_list(traces: list) -> None:
    """Print a table of traces."""
    if not traces:
        print("No traces found.")
        return

    print(f"{'TRACE ID':<12} {'LAST ACTIVE':<12} {'DURATION':<10} {'TOOLS':<6} {'STATUS'}")
    print("─" * 70)
    for t in traces:
        trace_id = t.get("trace_id_short", t["trace_id"][:8])
        last_active = _relative_time(t.get("last_timestamp", ""))
        # Calculate duration
        try:
            first = datetime.fromisoformat(t.get("first_timestamp", "").replace("Z", "+00:00"))
            last = datetime.fromisoformat(t.get("last_timestamp", "").replace("Z", "+00:00"))
            duration = (last - first).total_seconds() * 1000
            duration_str = _format_duration(duration)
        except Exception:
            duration_str = "?"
        tool_count = t.get("tool_count", 0)
        status = "✅ ok" if t.get("status") != "error" else "❌ error"
        print(f"{trace_id:<12} {last_active:<12} {duration_str:<10} {tool_count:<6} {status}")


def _print_trace_show(events: list) -> None:
    """Print detailed trace events."""
    if not events:
        print("Trace not found.")
        return

    first_ts = events[0].get("timestamp", "")
    last_ts = events[-1].get("timestamp", "")
    try:
        first_dt = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
        last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        total_ms = (last_dt - first_dt).total_seconds() * 1000
        total_str = _format_duration(total_ms)
    except Exception:
        total_str = "?"

    trace_id = events[0].get("trace_id", "?")[:12]
    print(f"Trace: {trace_id}")
    print(f"Time:  {first_ts} → {last_ts}")
    print(f"Duration: {total_str}")
    print(f"Events: {len(events)}")
    print()
    print("Events:")
    print("─" * 70)

    prev_ts = None
    for i, ev in enumerate(events, 1):
        ts = ev.get("timestamp", "")
        # Calculate delta from previous
        delta_str = ""
        if prev_ts:
            try:
                prev_dt = datetime.fromisoformat(prev_ts.replace("Z", "+00:00"))
                curr_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                delta_ms = (curr_dt - prev_dt).total_seconds() * 1000
                delta_str = f" [+{_format_duration(delta_ms)}]"
            except Exception:
                pass

        event_type = ev.get("event_type", "?")
        tool_name = ev.get("tool_name", "")

        if event_type == "llm_request":
            model = ev.get("extra", {}).get("model", "?")
            print(f" {i}. LLM Request  {ts}{delta_str}")
            print(f"    Model: {model}")
        elif event_type == "llm_response":
            preview = ev.get("extra", {}).get("response_preview", "")
            print(f" {i}. LLM Response {ts}{delta_str}")
            if preview:
                print(f"    Preview: {preview[:100]}{'...' if len(preview) > 100 else ''}")
        elif event_type == "tool_start":
            args = ev.get("tool_args", "")
            print(f" {i}. Tool Start   {ts}{delta_str}")
            print(f"    Tool: {tool_name}")
            if args:
                print(f"    Args: {args[:200]}{'...' if len(str(args)) > 200 else ''}")
        elif event_type == "tool_complete":
            result = ev.get("tool_result", "")
            error = ev.get("error", "")
            print(f" {i}. Tool Complete {ts}{delta_str}")
            print(f"    Tool: {tool_name}")
            if error:
                print(f"    Error: {error[:200]}")
            elif result:
                print(f"    Result: {result[:200]}{'...' if len(result) > 200 else ''}")
        else:
            print(f" {i}. {event_type} {ts}{delta_str}")
            if tool_name:
                print(f"    Tool: {tool_name}")

        prev_ts = ts


def _print_trace_analyze(analysis: dict) -> None:
    """Print trace analysis."""
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return

    trace_id = analysis.get("trace_id_short", "?")
    total_ms = analysis.get("total_duration_ms", 0)
    event_count = analysis.get("event_count", 0)
    status = analysis.get("status", "?")

    print(f"Trace: {trace_id}")
    print(f"Total Duration: {_format_duration(total_ms)}")
    print(f"Events: {event_count}")
    print(f"Status: {'❌ error' if status == 'error' else '✅ ok'}")
    print()

    # Tool calls analysis
    tool_calls = analysis.get("tool_calls", [])
    if tool_calls:
        print("Tool Calls:")
        print(f"{'TOOL':<30} {'DURATION':<12} {'STATUS'}")
        print("─" * 60)
        for call in tool_calls:
            name = call.get("tool_name", "?")[:28]
            dur = call.get("duration_ms")
            dur_str = _format_duration(dur) if dur else "?"
            call_status = "❌" if call.get("status") == "error" else "✅"
            print(f"{name:<30} {dur_str:<12} {call_status}")

    # Errors
    errors = analysis.get("errors", [])
    if errors:
        print()
        print("Errors:")
        print("─" * 60)
        for err in errors:
            ts = err.get("timestamp", "")
            tool = err.get("tool_name", "?")
            msg = err.get("error", "?")[:100]
            print(f"  [{ts}] {tool}: {msg}")


def _export_json(events: list) -> str:
    """Export events as formatted JSON."""
    return json.dumps(events, ensure_ascii=False, indent=2)


def _export_mermaid(events: list) -> str:
    """Export events as Mermaid sequence diagram."""
    lines = ["```mermaid", "sequenceDiagram"]

    # Track participants
    participants = ["User", "LLM"]
    tool_names = set()
    for ev in events:
        if ev.get("event_type") in ("tool_start", "tool_complete"):
            name = ev.get("tool_name", "?")
            if name not in tool_names:
                tool_names.add(name)
                participants.append(name)

    # Add participant declarations
    for p in participants:
        lines.append(f"    participant {p}")

    lines.append("")

    prev_ts = None
    for ev in events:
        ts = ev.get("timestamp", "")
        event_type = ev.get("event_type", "")
        tool_name = ev.get("tool_name", "")

        if event_type == "llm_request":
            lines.append("    User->>LLM: Request")
        elif event_type == "llm_response":
            lines.append("    LLM-->>User: Response")
        elif event_type == "tool_start":
            lines.append(f"    LLM->>{tool_name}: {tool_name}")
        elif event_type == "tool_complete":
            error = ev.get("error", "")
            if error:
                lines.append(f"    {tool_name}-->>LLM: Error")
            else:
                lines.append(f"    {tool_name}-->>LLM: Ok")

    lines.append("```")
    return "\n".join(lines)


def trace_command(args) -> None:
    """Main entry point for hermes trace command."""
    action = getattr(args, "trace_action", None)

    if action == "list":
        traces = list_traces(limit=args.limit)
        _print_trace_list(traces)

    elif action == "show":
        trace_id = _resolve_trace_id(args.trace_id)
        if not trace_id:
            print(f"Trace '{args.trace_id}' not found.")
            sys.exit(1)
        events = get_trace(trace_id)
        _print_trace_show(events)

    elif action == "analyze":
        trace_id = _resolve_trace_id(args.trace_id)
        if not trace_id:
            print(f"Trace '{args.trace_id}' not found.")
            sys.exit(1)
        analysis = analyze_trace(trace_id)
        _print_trace_analyze(analysis)

    elif action == "export":
        trace_id = _resolve_trace_id(args.trace_id)
        if not trace_id:
            print(f"Trace '{args.trace_id}' not found.")
            sys.exit(1)
        events = get_trace(trace_id)
        fmt = getattr(args, "format", "json")

        if fmt == "mermaid":
            output = _export_mermaid(events)
        else:
            output = _export_json(events)

        print(output)

    else:
        # No subcommand — show help
        print("Usage: hermes trace <list|show|analyze|export>")
        sys.exit(1)
