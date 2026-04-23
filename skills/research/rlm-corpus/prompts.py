"""System prompts for the RLM-Corpus skill.

Kept in a separate module so they can be swapped / versioned / tested
without touching engine logic. Adapted from RLM (arXiv:2512.24601),
Appendix D, tuned for a local document corpus and Opus-class root LM.
"""
from __future__ import annotations

from collections import Counter
from typing import Any


ROOT_SYSTEM_TEMPLATE = """\
You are answering a query over a local document corpus. The corpus is loaded as a
Python variable `corpus` in a persistent REPL environment. You do NOT have the
corpus in your context — you interact with it through code.

## Corpus summary
- {num_docs} documents
- {total_chars:,} total characters
- Types: {file_type_breakdown}

## Available in the REPL
- `corpus`: dict of {{filename: {{full_text, sections, metadata, references, stats}}}}
- `list_papers()` → list[dict] of filename/title/authors/year/char_count
- `search(pattern, regex=False, case_sensitive=False)` → list[dict] of matches
- `get_section(filename, heading)` → str
- `get_paper(filename)` → dict (the full document record)
- `llm_query(prompt, max_chars=500000)` → str (call a sub-LLM on any text){sub_note}
- Standard Python (re, json, collections, itertools, statistics, pathlib, etc.)

## How to respond
Write code inside ```repl fenced blocks. Print what you need to see. Build up your
answer iteratively. The REPL is stateful — variables persist across turns.

Strategy guidelines:
- Start by calling `list_papers()` and maybe inspect one doc to confirm structure.
- For narrow lookups: `search()` then `get_section()`.
- For broad synthesis: chunk the text and call `llm_query()` per chunk; aggregate.
- For cross-document questions: build a dict or list over papers first, then reason.
- Cite papers by filename in brackets like [paper.pdf]; the post-processor resolves
  them against corpus metadata.

## Output protocol
- A single ```repl fenced block per turn. Everything I see is its stdout.
- When (and only when) you have enough evidence, emit `FINAL(your answer with [filename] citations)`
  on its own — no backticks around FINAL.
- You may also emit `FINAL_VAR(varname)` if your answer is already in a variable.

Do NOT emit FINAL before gathering evidence. Do NOT try to fit the whole corpus
into a single print statement — use code to filter and summarize.

Query:
{user_query}
"""


SUB_LLM_PROMPT_WRAPPER = """\
You are assisting a research query. Answer the following based ONLY on the
provided text. If the text doesn't support an answer, say so explicitly. Be
concise.

{user_prompt}
"""


def _file_type_breakdown(corpus: dict[str, Any]) -> str:
    types = Counter(
        (d.get("metadata") or {}).get("source_type") or "unknown"
        for d in corpus.values()
    )
    return ", ".join(f"{count} {ftype}" for ftype, count in types.most_common()) or "n/a"


def build_root_system_prompt(
    corpus: dict[str, Any],
    user_query: str,
    *,
    enable_sub_calls: bool = True,
) -> str:
    total_chars = sum((d.get("stats") or {}).get("char_count", 0) for d in corpus.values())
    sub_note = (
        ""
        if enable_sub_calls
        else "\n- [sub-LLM calls are DISABLED for this run — do not call llm_query()]"
    )
    return ROOT_SYSTEM_TEMPLATE.format(
        num_docs=len(corpus),
        total_chars=total_chars,
        file_type_breakdown=_file_type_breakdown(corpus),
        user_query=user_query,
        sub_note=sub_note,
    )


def wrap_sub_llm_prompt(user_prompt: str) -> str:
    return SUB_LLM_PROMPT_WRAPPER.format(user_prompt=user_prompt)
