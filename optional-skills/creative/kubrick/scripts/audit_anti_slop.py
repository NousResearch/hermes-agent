#!/usr/bin/env python3
"""Deterministic heuristic audit for Kubrick symbolic Gates N–W."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

RULES = {
    "N": (r"\b(tarot|ouroboros|pentagram|sigil|alchemy|kabbalah|qabalah)\b.*\b(tarot|ouroboros|pentagram|sigil|alchemy|kabbalah|qabalah)\b", "Reduce mixed occult references to one governing observable operation."),
    "O": (r"\b(dialogue|image|music|lighting)\b.{0,80}\b(same meaning|all represent|all symbolize)\b", "Move one channel into counterpoint or remove redundant emphasis."),
    "P": (r"\b(red means danger|bird means freedom|mirror means identity|water means emotion)\b", "Replace fixed equivalence with contextual mutation and consequence."),
    "Q": (r"\b(repeats? unchanged|same motif again|identical recurrence)\b", "Mutate ownership, scale, state, rhythm, or function on recurrence."),
    "R": (r"\b(the archetype|the trickster|the shadow|the magician)\b", "Describe enacted behavior and relational function instead of naming an archetype."),
    "S": (r"\b(all traditions|universal symbol|same as (zen|hindu|african|indigenous|kabbalah))\b", "Restore provenance boundaries and remove unsupported equivalence."),
    "U": (r"\b(ignore causality|symbolism matters more|sacrifice clarity|because it is symbolic)\b", "Restore causality, agency, readability, and production feasibility."),
    "W": (r"\b(this proves|the true meaning is|the correct interpretation|it definitely symbolizes)\b", "Preserve interpretive openness; keep evidence concrete and avoid closure."),
}

def audit(text: str) -> dict:
    violations=[]
    for gate,(pattern,repair) in RULES.items():
        matches=[m.group(0) for m in re.finditer(pattern,text,re.I|re.S)]
        if matches: violations.append({"gate":gate,"evidence":matches[:3],"repair":repair})
    return {"status":"PASS" if not violations else "FAIL","violation_count":len(violations),"violations":violations}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input"); p.add_argument("--text"); p.add_argument("--json",action="store_true"); a=p.parse_args()
    text=a.text if a.text is not None else Path(a.input).read_text(encoding="utf-8")
    result=audit(text)
    print(json.dumps(result,indent=2) if a.json else ("PASS" if result["status"]=="PASS" else "\n".join(f"Gate {v['gate']}: {v['repair']}" for v in result["violations"])))
    raise SystemExit(0 if result["status"]=="PASS" else 1)
if __name__=="__main__": main()
