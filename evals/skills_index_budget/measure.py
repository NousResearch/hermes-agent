#!/usr/bin/env python3
"""Render a library's skills index under several budgets and report what each one scrubs.

    PYTHONPATH=<tree> <hermes env python> measure.py <library.json> <out_dir> [window:percent ...]

Writes ``<out_dir>/renders.json`` (label -> index text, the shape a provider token counter takes)
and prints, per budget, the estimated tokens the gate saw and how many skills went names-only.
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
library_json, out_dir = sys.argv[1], sys.argv[2]
budgets = [tuple(float(x) for x in arg.split(":")) for arg in sys.argv[3:]] or [(200_000, 1), (1_000_000, 1)]

home = tempfile.mkdtemp(prefix="sib-measure-")
os.environ["HERMES_HOME"] = home
from library import build_home

rows = build_home(home, library_json)

from agent import prompt_builder as pb
from agent.model_metadata import estimate_tokens_rough
from agent.skills_index_delta import index_entries

TOOLS = {"skills_list", "skill_view", "skill_manage", "terminal", "web_search"}


def render(budget_tokens):
    return pb.build_skills_system_prompt(available_tools=TOOLS, available_toolsets={"skills", "terminal", "web"},
                                         index_budget_tokens=budget_tokens)


full = render(None)
renders = {"full": full}
full_names = set(index_entries(full))
print(f"skills listed: {len(full_names)}  full index: {len(full.encode())} bytes, gate estimate {estimate_tokens_rough(full)} tokens")
for window, percent in budgets:
    budget = int(window * percent / 100)
    text = render(budget)
    entries = index_entries(text)
    assert set(entries) == full_names, "a skill was dropped from the index"
    bare = sorted(n for n, row in entries.items() if ": " not in row)
    label = f"window{int(window)}_pct{percent:g}"
    renders[label] = text
    print(f"{label}: budget {budget}  estimate after {estimate_tokens_rough(text)}  names-only {len(bare)}/{len(entries)}")
    with open(os.path.join(out_dir, f"{label}.names_only.json"), "w", encoding="utf-8") as f:
        json.dump(bare, f, indent=1)
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "renders.json"), "w", encoding="utf-8") as f:
    json.dump(renders, f, indent=1)
