#!/usr/bin/env python3
"""Synthetic replay benchmark for retrieval-backed tool-result compaction."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shlex
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.model_metadata import estimate_request_tokens_rough
from agent.tool_executor import _allocated_tool_result_budget
from tools.budget_config import BudgetConfig
from tools.tool_result_storage import maybe_persist_tool_result


class _FixtureEnvironment:
    """Minimal local environment that exercises the production stdin write path."""

    def __init__(self, root: Path):
        self.root = root

    def get_temp_dir(self) -> str:
        return str(self.root)

    def execute(self, command: str, timeout: int, stdin_data: str) -> dict:
        del timeout
        match = re.search(r"cat > (.+)$", command)
        if not match:
            return {"output": "missing target", "returncode": 1}
        target = Path(shlex.split(match.group(1))[0])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(stdin_data, encoding="utf-8")
        target.chmod(0o600)
        return {"output": "", "returncode": 0}


def _synthetic_result(index: int, target_chars: int = 32_100) -> str:
    head = f"fixture tool {index} started\n"
    anomaly = f"WARNING: synthetic retry marker {index}\n"
    tail = f"fixture tool {index} completed\n"
    filler = (f"synthetic line {index:02d} -- no private session content\n" * 900)
    body = (head + filler + anomaly + filler + tail)[:target_chars]
    return json.dumps(
        {"output": body, "exit_code": 0, "error": None},
        ensure_ascii=False,
    )


def run_benchmark() -> dict:
    user_text = ("Synthetic Slack request context. " * 800)[:24_000]
    assistant_text = ("Synthetic routing note. " * 50)[:800]
    tool_names = ["terminal"] * 6
    tool_calls = [
        {
            "id": f"fixture_call_{index}",
            "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }
        for index, name in enumerate(tool_names)
    ]
    stable_prefix = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text, "tool_calls": tool_calls},
    ]
    raw_results = [_synthetic_result(index) for index in range(len(tool_names))]
    before = copy.deepcopy(stable_prefix) + [
        {
            "role": "tool",
            "tool_name": name,
            "tool_call_id": tool_calls[index]["id"],
            "content": raw_results[index],
        }
        for index, name in enumerate(tool_names)
    ]

    budget = BudgetConfig()
    with tempfile.TemporaryDirectory(prefix="hermes-context-tray-bench-") as temp_dir:
        env = _FixtureEnvironment(Path(temp_dir))
        after = copy.deepcopy(stable_prefix)
        artifact_paths: list[str] = []
        inline_hashes: list[str] = []
        for index, (name, raw) in enumerate(zip(tool_names, raw_results)):
            threshold = min(
                budget.resolve_threshold(name),
                _allocated_tool_result_budget(
                    after, tool_names[index:], budget
                ),
            )
            compacted = maybe_persist_tool_result(
                content=raw,
                tool_name=name,
                tool_use_id=tool_calls[index]["id"],
                env=env,
                config=budget,
                threshold=threshold,
            )
            path_match = re.search(r"Full output saved to: (.+)", compacted)
            if path_match:
                artifact_paths.append(path_match.group(1).strip())
            hash_match = re.search(r"sha256: ([0-9a-f]{64})", compacted)
            if hash_match:
                inline_hashes.append(hash_match.group(1))
            after.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "tool_call_id": tool_calls[index]["id"],
                    "content": compacted,
                }
            )

        retrieved_hashes = [
            hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in artifact_paths
        ]
        expected_hashes = [
            hashlib.sha256(raw.encode("utf-8")).hexdigest()
            for raw in raw_results
        ]
        retrieval_fidelity = (
            len(artifact_paths) == len(raw_results)
            and retrieved_hashes == expected_hashes
            and inline_hashes == expected_hashes
        )

    before_tokens = estimate_request_tokens_rough(before)
    after_tokens = estimate_request_tokens_rough(after)
    before_tool_chars = sum(
        len(message["content"]) for message in before if message["role"] == "tool"
    )
    after_tool_chars = sum(
        len(message["content"]) for message in after if message["role"] == "tool"
    )
    reduction_pct = round(
        (1 - (after_tool_chars / before_tool_chars)) * 100,
        2,
    )
    compression_threshold = 50_000
    stable_before = json.dumps(stable_prefix, ensure_ascii=False, sort_keys=True)
    stable_after = json.dumps(after[:2], ensure_ascii=False, sort_keys=True)
    pairing_before = [message.get("tool_call_id") for message in before[2:]]
    pairing_after = [message.get("tool_call_id") for message in after[2:]]

    return {
        "fixture": {
            "private_content": False,
            "user_chars": len(user_text),
            "assistant_chars": len(assistant_text),
            "tool_result_chars": before_tool_chars,
            "tool_results": len(raw_results),
        },
        "request_tokens": {"before": before_tokens, "after": after_tokens},
        "tool_result_chars": {
            "before": before_tool_chars,
            "after": after_tool_chars,
            "reduction_pct": reduction_pct,
        },
        "compression": {
            "threshold_tokens": compression_threshold,
            "count_before": int(before_tokens >= compression_threshold),
            "count_after": int(after_tokens >= compression_threshold),
            "progress_before_pct": round(before_tokens / compression_threshold * 100, 2),
            "progress_after_pct": round(after_tokens / compression_threshold * 100, 2),
        },
        "retrieval": {
            "artifacts": len(artifact_paths),
            "full_raw_sha256_identical": retrieval_fidelity,
        },
        "invariants": {
            "user_assistant_byte_identical": stable_before == stable_after,
            "already_sent_prefix_sha256_identical": (
                hashlib.sha256(stable_before.encode()).hexdigest()
                == hashlib.sha256(stable_after.encode()).hexdigest()
            ),
            "tool_call_result_pairing_identical": pairing_before == pairing_after,
        },
        "target_met": reduction_pct >= 60 and retrieval_fidelity,
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), ensure_ascii=False, indent=2, sort_keys=True))
