"""Anchor-index harvest surface (Compaction-v2 lean mode).

The ledger exists so exact identifiers cannot be paraphrased away by the summary LLM,
but it may only preserve what its scan can see and what its budget can fit. Covered here:

* identifiers that reach the transcript only as tool-call arguments
  (``read_file(path=...)`` / ``terminal(command=...)``) — invisible to a content-only scan;
* identifier classes the pattern table did not cover: spreadsheet / notebook / log paths,
  session ids, todo ids;
* budget starvation: one greedy section overflowing the ledger budget must not delete the
  cheap, high-signal sections behind it.

Assertions are relationships (what the produced index must contain), not snapshots of the
pattern table.
"""
import agent.context_compressor as cc


def test_anchor_index_harvests_tool_call_arguments_and_artifact_paths():
    """Paths reachable only through tool-call args, and non-code artifact paths, must be indexed."""
    turns = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": '{"path": "/srv/app/deploy/nightly.yaml"}'}},
                {"function": {"name": "terminal", "arguments": '{"command": "cat /srv/app/deploy/nightly.yaml"}'}},
            ],
        },
        {"role": "tool", "content": "ok"},
        {"role": "user", "content": "汇总 data/企业总表.csv 与 reports/2026/产业全景.xlsx 后回我"},
    ]

    index = cc._build_anchor_index(turns)

    assert "app/deploy/nightly.yaml" in index  # harvested verbatim from the args, not from content
    assert "data/企业总表.csv" in index
    assert "reports/2026/产业全景.xlsx" in index


def test_anchor_index_cheap_ids_survive_a_tight_budget(monkeypatch):
    """Ids must be emitted before the greedy file-path section, and the budget must hold."""
    monkeypatch.setattr(cc, "_LEAN_ANCHOR_BUDGET_CHARS", 900)
    turns = [
        {"role": "user", "content": "上一轮会话 20260920_220503_c78aff58 的口径，待办 [221791] 要复核"},
        {
            "role": "assistant",
            "content": "扫描以下文件。" + "、".join(f"src/pkg/module_{i}/handlers/deep_path_{i}.py" for i in range(40)),
        },
    ]

    index = cc._build_anchor_index(turns)

    assert "[221791]" in index
    assert "20260920_220503_c78aff58" in index
    assert len(index) <= cc._LEAN_ANCHOR_BUDGET_CHARS + 200  # heading + footer are outside the sections budget
