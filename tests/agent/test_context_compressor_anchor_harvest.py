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


# The production table, isolated to one class so the budget tests below assert labels, not counts.
_PATH_PATTERN = next(entry for entry in cc._ANCHOR_PATTERNS if entry[0] == "files")


def test_anchor_index_files_never_span_prose_or_quoting_punctuation():
    """A "path" value must be one path: an unpunctuated CJK/backtick list must not glue into one value."""
    turns = [{
        "role": "user",
        "content": "产物 tmp/`：`review_a.py`、`arch_b.py`、`wj_c.py`、`out.json` 已归档；另见 src/deep/real_module.py",
    }]

    index = cc._build_anchor_index(turns)
    line = next((ln for ln in index.splitlines() if ln.startswith("files:")), "")
    values = [v.split("(x")[0] for v in line[len("files: "):].split(", ")] if line else []

    assert "src/deep/real_module.py" in values, f"真实路径仍须采集: {values!r}"
    for value in values:
        assert not any(ch in value for ch in "`、：，。"), f"路径值吞入散文标点: {value!r}"
        assert len(value) <= 120, f"路径值长度失控: {len(value)}"
    assert not any("review_a.py" in v and "arch_b.py" in v for v in values), f"枚举被并成一个值: {values!r}"


def test_anchor_index_names_a_class_whose_values_cannot_fit(monkeypatch):
    """A scanned class that loses its values to the budget must still be named."""
    monkeypatch.setattr(cc, "_ANCHOR_PATTERNS", [_PATH_PATTERN])
    monkeypatch.setattr(cc, "_LEAN_ANCHOR_BUDGET_CHARS", 10)  # fits "files:", not the first value
    turns = [{"role": "user", "content": "见 src/very/deep/nested/module_name.py 与 src/other/second_file.py"}]

    index = cc._build_anchor_index(turns)

    assert "files:" in index  # label survives the cut
    assert "module_name.py" not in index  # ...without its values pretending to have fit


def test_anchor_index_omits_a_class_when_even_its_label_cannot_fit(monkeypatch):
    """No room for the label either: the section is dropped, not emitted as a stub."""
    monkeypatch.setattr(cc, "_ANCHOR_PATTERNS", [_PATH_PATTERN])
    monkeypatch.setattr(cc, "_LEAN_ANCHOR_BUDGET_CHARS", 0)
    turns = [{"role": "user", "content": "见 src/very/deep/nested/module_name.py"}]

    assert cc._build_anchor_index(turns) == ""
