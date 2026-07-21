#!/usr/bin/env python3
"""Deep Research Orchestrator - Automates research workflow planning."""
import json, sys, argparse
from datetime import datetime

MODES = {
    "quick": {"max_sub_questions": 3, "searches_per_question": 2},
    "standard": {"max_sub_questions": 6, "searches_per_question": 3},
    "deep": {"max_sub_questions": 10, "searches_per_question": 4},
    "compare": {"max_sub_questions": 5, "searches_per_question": 3},
}

TEMPLATES = {
    "technology": [
        "Performance characteristics of {topic}?",
        "Ecosystem and tooling comparison?",
        "Adoption rate and case studies?",
        "Limitations and challenges?",
        "Migration path and cost?",
    ],
    "market": [
        "Market size and growth rate of {topic}?",
        "Key players and market share?",
        "Major trends and predictions?",
        "Regulatory landscape?",
        "Barriers to entry?",
    ],
    "competitive": [
        "Feature differences between {topic}?",
        "Pricing and value comparison?",
        "User reviews and satisfaction?",
        "Strengths and weaknesses of each?",
        "Best use cases for each?",
    ],
    "general": [
        "Current state of {topic}?",
        "Key developments and trends?",
        "Expert opinions and analysis?",
        "Pros and cons?",
        "Recommendations and best practices?",
    ],
}

def detect_type(q):
    ql = q.lower()
    if any(w in ql for w in ["compare","vs","versus","better than"]): return "competitive"
    if any(w in ql for w in ["market","industry","revenue","size"]): return "market"
    if any(w in ql for w in ["framework","library","language","tool","platform"]): return "technology"
    return "general"

def main():
    p = argparse.ArgumentParser(description="Deep Research Orchestrator")
    p.add_argument("-q","--question",required=True,help="Research question")
    p.add_argument("-m","--mode",default="standard",choices=MODES.keys())
    p.add_argument("-o","--output",default="research_report.md")
    args = p.parse_args()
    config = MODES[args.mode]
    q_type = detect_type(args.question)
    template = TEMPLATES.get(q_type, TEMPLATES["general"])
    subs = [t.format(topic=args.question) for t in template[:config["max_sub_questions"]]]
    plan = {
        "question": args.question,
        "mode": args.mode,
        "type": q_type,
        "sub_questions": subs,
        "config": config,
        "date": datetime.now().isoformat(),
    }
    print(json.dumps(plan, indent=2))

if __name__=="__main__": main()
