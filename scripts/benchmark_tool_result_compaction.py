"""Synthetic, local-only benchmark; no model requests or private session data.

Requires the optional ``tiktoken`` package. From the repository root:
    python -m scripts.benchmark_tool_result_compaction --encoding cl100k_base

Counts the entire tool-result body, not total conversation cost. Fixtures model
common payload shapes; they are not a production corpus or a task-quality eval.
"""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.tool_result_compaction import compact_tool_result


def fixtures():
    records = [{"id": i, "title": f"Planning meeting {i}", "start": "2026-09-14T09:00:00Z",
                "attendees": ["alex@example.test", "sam@example.test"], "cancelled": False}
               for i in range(50)]
    searches = [{"title": f"Reference {i}", "url": f"https://example.test/docs/{i}",
                 "snippet": "Reference text for a research question.", "rank": i}
                for i in range(40)]
    rows = [{"row": i, "quantity": i * 3, "unit": "items", "available": True,
             "note": None} for i in range(100)]
    prose = "A document paragraph with  intentional spacing.\n  Preserve indentation.\n" * 200
    return [
        ("calendar_records", "mcp_calendar", json.dumps({"events": records}, indent=2)),
        ("research_results", "web_search", json.dumps({"results": searches}, indent=2)),
        ("table_records", "mcp_table", json.dumps({"rows": rows}, indent=2)),
        ("document_text", "mcp_document", json.dumps({"text": prose}, indent=2)),
        ("already_compact", "mcp_table", json.dumps({"rows": rows}, separators=(",", ":"))),
        ("exact_file_read", "read_file", json.dumps({"content": prose}, indent=2)),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoding", default="cl100k_base", choices=["cl100k_base", "o200k_base"])
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    try:
        import tiktoken  # ty: ignore[unresolved-import]
    except ImportError:
        parser.error("Install tiktoken in your development environment to run this benchmark")
    encoding = tiktoken.get_encoding(args.encoding)
    print(f"Synthetic fixtures; encoding={args.encoding}; no model/API calls")
    print("fixture,original_chars,result_chars,original_tokens,result_tokens,saved_pct,median_ms")
    with tempfile.TemporaryDirectory(prefix="hermes-compaction-bench-") as directory:
        home = Path(directory)
        (home / "config.yaml").write_text(
            'tool_output:\n  json_compaction:\n    mode: "compact"\n', encoding="utf-8",
        )
        token = set_hermes_home_override(home)
        try:
            for name, tool, original in fixtures():
                result = compact_tool_result(original, tool)  # warm config cache
                timings = []
                for _ in range(args.iterations):
                    start = time.perf_counter()
                    compact_tool_result(original, tool)
                    timings.append((time.perf_counter() - start) * 1000)
                before = len(encoding.encode(original))
                after = len(encoding.encode(result))
                # Parsed data equality is necessary, but the tests separately
                # verify lexical fidelity for escapes, duplicate keys and numbers.
                assert json.loads(original) == json.loads(result)
                print(f"{name},{len(original)},{len(result)},{before},{after},"
                      f"{100 * (before - after) / before:.2f},{statistics.median(timings):.3f}")
        finally:
            reset_hermes_home_override(token)


if __name__ == "__main__":
    main()
