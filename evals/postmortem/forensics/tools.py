"""Lane 3: tool-layer friction — hardline false blocks, foreground refusals, whole-file rewrites, output
volume by tool. All OBSERVED from ``messages`` (role='tool') in the run tree.

    python -m evals.postmortem.forensics.tools --db state_copy.db
"""
from __future__ import annotations

import collections
import json
import re
from typing import Any, Dict

from evals.postmortem.forensics.common import Run

_HARDLINE = "BLOCKED (hardline)"
_MALFORMED = "command parser limit or malformed executable payload"
_FG_TIMEOUT = re.compile(r"Foreground timeout (\d+)s exceeds the maximum")
_FG_AMP = "Foreground command uses '&' backgrounding"
_FG_WRAP = "shell-level background wrappers"
_DEADLINE = re.compile(r"timed out after ([\d.]+)s")


def main(argv=None) -> int:
    run = Run.from_args(argv, (__doc__ or "").split("\n\n")[0])
    by_tool_bytes: Dict[str, int] = collections.Counter()
    by_tool_calls: Dict[str, int] = collections.Counter()
    hardline = malformed = fg_timeout = fg_amp = fg_wrap = tool_deadline = 0
    fg_timeout_requested: Dict[int, int] = collections.Counter()
    write_calls = big_writes = rewrite_of_existing = 0
    write_chars = 0
    read_paths_by_sess: Dict[str, set] = collections.defaultdict(set)
    for sid in run.in_run:
        for m in run.messages(sid, "role, tool_name, content, tool_calls"):
            if m["role"] == "assistant" and m.get("tool_calls"):
                try:
                    tcs = json.loads(m["tool_calls"])
                except ValueError:
                    continue
                for tc in tcs:
                    fn = (tc.get("function") or {})
                    name = fn.get("name") or ""
                    by_tool_calls[name] += 1
                    if name in ("read_file", "write_file", "patch"):
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except ValueError:
                            args = {}
                        if name == "read_file" and args.get("path"):
                            read_paths_by_sess[sid].add(args["path"])
                        if name == "write_file":
                            write_calls += 1
                            content = args.get("content") or ""
                            write_chars += len(content)
                            if len(content) > 20_000:
                                big_writes += 1
                                if args.get("path") in read_paths_by_sess[sid]:
                                    rewrite_of_existing += 1
                continue
            if m["role"] != "tool":
                continue
            c = m.get("content") or ""
            by_tool_bytes[m.get("tool_name") or "?"] += len(c)
            if _HARDLINE in c:
                hardline += 1
                if _MALFORMED in c:
                    malformed += 1
            mt = _FG_TIMEOUT.search(c)
            if mt:
                fg_timeout += 1; fg_timeout_requested[int(mt.group(1))] += 1
            if _FG_AMP in c:
                fg_amp += 1
            if _FG_WRAP in c:
                fg_wrap += 1
            if (m.get("tool_name") or "") == "terminal" and _DEADLINE.search(c):
                tool_deadline += 1
    top_bytes = sorted(by_tool_bytes.items(), key=lambda kv: -kv[1])[:8]
    report: Dict[str, Any] = {"observed": {
        "tool_results": sum(by_tool_calls.values()),
        "output_bytes_by_tool_top8": {k: v for k, v in top_bytes},
        "hardline_blocks": hardline, "hardline_blocks_malformed_class": malformed,
        "foreground_timeout_refusals": fg_timeout, "foreground_timeout_requested_top": dict(fg_timeout_requested.most_common(5)),
        "foreground_ampersand_refusals": fg_amp, "foreground_wrapper_refusals": fg_wrap,
        "terminal_deadline_kills": tool_deadline,
        "write_file": {"calls": write_calls, "chars": write_chars, "writes_over_20k_chars": big_writes,
                       "rewrites_of_a_file_read_this_session_over_20k": rewrite_of_existing,
                       "output_usd_at_fitted_price": round(write_chars / 3.5 * run.price_per_token["output_tokens"], 2)},
        "patch_calls": by_tool_calls.get("patch", 0),
    }}
    path = run.write("tools.json", report)
    o = report["observed"]
    print(f"[tools] {o['tool_results']:,} tool calls; hardline blocks {o['hardline_blocks']} ({o['hardline_blocks_malformed_class']} 'malformed' class); "
          f"foreground timeout refusals {o['foreground_timeout_refusals']} (asked: {o['foreground_timeout_requested_top']}); '&' {o['foreground_ampersand_refusals']}, nohup {o['foreground_wrapper_refusals']}")
    w = o["write_file"]
    print(f"[tools] write_file {w['calls']:,} calls, {w['chars']/1e6:.1f}M chars (~${w['output_usd_at_fitted_price']:,}); >20k: {w['writes_over_20k_chars']}, of which rewrites of a file read this session: {w['rewrites_of_a_file_read_this_session_over_20k']}; patch calls {o['patch_calls']:,}")
    print(f"[tools] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
