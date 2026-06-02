#!/usr/bin/env python3
"""Score paper-search / kanban Feishu messages against researcher A/B tiers."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass

_ARXIV = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
_BRACKET_ARXIV = re.compile(r"\[(\d{4}\.\d{4,5})\]")
_S2 = re.compile(r"\bs2:([0-9a-f]{40})\b", re.I)
_ARXIV_URL = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", re.I)


@dataclass(frozen=True)
class TierPaper:
    tier: str
    paper_id: str
    title_en: str
    title_zh: str


# Reference tiers from researcher brief (GNN classic track)
GNN_SEARCH_A = [
    TierPaper("A", "1609.02907", "Semi-Supervised Classification with Graph Convolutional Networks", "图卷积网络 GCN 半监督分类"),
    TierPaper("A", "1706.02216", "Inductive Representation Learning on Large Graphs (GraphSAGE)", "GraphSAGE 大图归纳表示学习"),
    TierPaper("A", "1710.10903", "Graph Attention Networks", "图注意力网络 GAT"),
]
GNN_SEARCH_B = [
    TierPaper("B", "2101.11174", "Graph Neural Network for Traffic Forecasting: A Survey", "GNN 交通预测综述"),
    TierPaper("B", "1901.00596", "A Comprehensive Survey on Graph Neural Networks", "图神经网络综合综述 Wu"),
    TierPaper("B", "2001.00405", "Benchmarking Graph Neural Networks", "GNN 基准评测"),
]

TRANSFORMER_A = [
    TierPaper("A", "1706.03762", "Attention Is All You Need", "Transformer 注意力即一切"),
]

S2_IDS_SURVEY = [
    TierPaper("A", "s2:ceced53f349f7e425352ecf4813b307667cd8aa6",
              "A survey on graph neural networks for intrusion detection systems",
              "图神经网络入侵检测综述"),
]


def extract_ids(text: str) -> set[str]:
    ids: set[str] = set()
    for m in _ARXIV.finditer(text):
        ids.add(re.sub(r"v\d+$", "", m.group(1), flags=re.I))
    for m in _BRACKET_ARXIV.finditer(text):
        ids.add(re.sub(r"v\d+$", "", m.group(1), flags=re.I))
    for m in _ARXIV_URL.finditer(text):
        ids.add(re.sub(r"v\d+$", "", m.group(1), flags=re.I))
    for m in _S2.finditer(text):
        ids.add(f"s2:{m.group(1).lower()}")
    return ids


def score_hits(found: set[str], tiers: list[TierPaper]) -> dict:
    by_id = {p.paper_id.lower(): p for p in tiers}
    hits = []
    for pid in sorted(found):
        key = pid.lower()
        if key in by_id:
            hits.append(by_id[key])
    tier_a = sum(1 for h in hits if h.tier == "A")
    tier_b = sum(1 for h in hits if h.tier == "B")
    return {
        "found_ids": sorted(found),
        "hits": [
            {"tier": h.tier, "id": h.paper_id, "title_zh": h.title_zh, "title_en": h.title_en}
            for h in hits
        ],
        "tier_a_hits": tier_a,
        "tier_b_hits": tier_b,
        "tier_a_expected": sum(1 for p in tiers if p.tier == "A"),
        "tier_b_expected": sum(1 for p in tiers if p.tier == "B"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gnn_search", "gcn_kanban", "s2_kanban", "transformer_kanban"], required=True)
    p.add_argument("--text", help="message body to score")
    p.add_argument("--text-file")
    p.add_argument("--expect-id", help="canonical id that must appear (kanban tests)")
    args = p.parse_args()

    text = args.text or ""
    if args.text_file:
        text = open(args.text_file, encoding="utf-8").read() + text

    found = extract_ids(text)
    if args.expect_id:
        exp = args.expect_id.strip().lower()
        if exp.startswith("s2:"):
            exp = exp.lower()
        else:
            exp = re.sub(r"v\d+$", "", exp, flags=re.I)
        if exp not in {x.lower() for x in found} and exp not in found:
            # also check without prefix
            found.add(f"__MISSING_EXPECT__{exp}")

    if args.mode == "gnn_search":
        tiers = GNN_SEARCH_A + GNN_SEARCH_B
        label = "paper-search · 图神经网络经典"
    elif args.mode == "gcn_kanban":
        tiers = [p for p in GNN_SEARCH_A if p.paper_id == "1609.02907"]
        label = "kanban-paper-nexus · GCN 1609.02907"
    elif args.mode == "transformer_kanban":
        tiers = TRANSFORMER_A
        label = "kanban-paper-nexus · Transformer 1706.03762"
    else:
        tiers = S2_IDS_SURVEY
        label = "kanban-paper-nexus · S2 IDS 综述"

    out = score_hits(found, tiers)
    out["label"] = label
    if args.expect_id:
        exp = args.expect_id.strip().lower()
        if not exp.startswith("s2:"):
            exp = re.sub(r"v\d+$", "", exp, flags=re.I)
        out["expect_id"] = exp
        out["expect_hit"] = any(
            exp == x.lower() or exp in x.lower() for x in found if not x.startswith("__MISSING")
        )
    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
