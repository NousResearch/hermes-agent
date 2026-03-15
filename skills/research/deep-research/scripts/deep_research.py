#!/usr/bin/env python3
"""
Deep Research — Research session manager and report generator.

Manages research sessions, tracks findings, generates reports.
Works with all available research tools (Google News, DDG, arxiv, etc.)

Usage:
  python3 deep_research.py init "Is AI replacing programmers?"     # New session
  python3 deep_research.py plan                                     # Show research plan
  python3 deep_research.py log "Found paper showing 40% increase"   # Log finding
  python3 deep_research.py source "https://..." "title" "type" --credibility 4
  python3 deep_research.py reflect                                  # Run reflection
  python3 deep_research.py report                                   # Generate report
  python3 deep_research.py activity                                 # Show research activity log
  python3 deep_research.py sessions                                 # List all sessions
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)


def get_state_dir():
    """Return a writable state directory for deep research sessions."""
    override = os.environ.get("HERMES_DEEP_RESEARCH_STATE_DIR")
    if override:
        return override

    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        return os.path.join(hermes_home, "state", "deep-research")

    return str(Path.home() / ".hermes" / "state" / "deep-research")


STATE_DIR = get_state_dir()
SESSIONS_DIR = os.path.join(STATE_DIR, "sessions")

def now_local():
    """Return current time in system's local timezone."""
    return datetime.now().astimezone()


def timestamp():
    return now_local().strftime("%Y-%m-%d %H:%M %Z")


def date_slug():
    return now_local().strftime("%Y-%m-%d")


def get_session_path(name=None):
    """Get path to session directory."""
    if name:
        return os.path.join(SESSIONS_DIR, name)
    # Find latest session
    if not os.path.exists(SESSIONS_DIR):
        return None
    sessions = sorted(os.listdir(SESSIONS_DIR), reverse=True)
    if sessions:
        return os.path.join(SESSIONS_DIR, sessions[0])
    return None


def load_session(session_path):
    """Load session data."""
    data_file = os.path.join(session_path, "session.json")
    if os.path.exists(data_file):
        with open(data_file, encoding="utf-8") as f:
            data = json.load(f)
            if "research_log" not in data and "chain_of_thought" in data:
                data["research_log"] = data.pop("chain_of_thought")
            return data
    return None


def save_session(session_path, data):
    """Save session data."""
    os.makedirs(session_path, exist_ok=True)
    data_file = os.path.join(session_path, "session.json")
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def cmd_init(args):
    """Initialize a new research session."""
    # Create a privacy-safe session name without embedding the topic in the path
    session_name = f"{date_slug()}_{uuid.uuid4().hex[:8]}"
    session_path = os.path.join(SESSIONS_DIR, session_name)

    if os.path.exists(session_path):
        print(f"Session already exists: {session_name}")
        return 1

    session = {
        "name": session_name,
        "topic": args.topic,
        "created": timestamp(),
        "status": "planning",
        "questions": [],
        "sub_questions": [],
        "sources": [],
        "findings": [],
        "contradictions": [],
        "gaps": [],
        "research_log": [],
        "confidence": None,
        "report": None,
    }

    # Parse initial questions if provided
    if args.questions:
        for q in args.questions.split("|"):
            session["sub_questions"].append(q.strip())

    save_session(session_path, session)

    # Add initial research log entry
    session["research_log"].append({
        "step": 1,
        "phase": "init",
        "timestamp": timestamp(),
        "content": f"Research session initialized.\nCore question: {args.topic}",
    })
    save_session(session_path, session)

    print(json.dumps({
        "session": session_name,
        "path": session_path,
        "topic": args.topic,
        "status": "planning",
    }, indent=2))
    return 0


def cmd_plan(args):
    """Show or update research plan."""
    session_path = get_session_path(args.session)
    if not session_path:
        print("No active session. Run: python3 deep_research.py init 'topic'")
        return 1

    session = load_session(session_path)
    if not session:
        print("Could not load session")
        return 1

    if args.add_question:
        session["sub_questions"].append(args.add_question)
        session["research_log"].append({
            "step": len(session["research_log"]) + 1,
            "phase": "planning",
            "timestamp": timestamp(),
            "content": f"Added sub-question: {args.add_question}",
        })
        save_session(session_path, session)
        print(f"Added: {args.add_question}")

    # Display current plan
    plan = {
        "session": session["name"],
        "topic": session["topic"],
        "status": session["status"],
        "sub_questions": session["sub_questions"],
        "sources_found": len(session["sources"]),
        "findings": len(session["findings"]),
        "confidence": session["confidence"],
    }
    print(json.dumps(plan, indent=2))
    return 0


def cmd_log(args):
    """Log a finding."""
    session_path = get_session_path(args.session)
    if not session_path:
        print("No active session")
        return 1

    session = load_session(session_path)

    finding = {
        "content": args.finding,
        "timestamp": timestamp(),
        "source_url": args.source or None,
        "credibility": args.credibility or None,
        "type": args.type or "observation",
    }
    session["findings"].append(finding)

    # Add to research log
    session["research_log"].append({
        "step": len(session["research_log"]) + 1,
        "phase": "evaluation",
        "timestamp": timestamp(),
        "content": f"Finding logged: {args.finding}",
    })

    save_session(session_path, session)
    print(json.dumps({"logged": True, "total_findings": len(session["findings"])}, indent=2))
    return 0


def cmd_source(args):
    """Add a source."""
    session_path = get_session_path(args.session)
    if not session_path:
        print("No active session")
        return 1

    session = load_session(session_path)

    source = {
        "url": args.url,
        "title": args.title,
        "type": args.source_type,
        "credibility": args.credibility or 3,
        "accessed": timestamp(),
    }
    session["sources"].append(source)

    session["research_log"].append({
        "step": len(session["research_log"]) + 1,
        "phase": "search",
        "timestamp": timestamp(),
        "content": f"Source added: {args.title} (credibility: {source['credibility']}/5)",
    })

    save_session(session_path, session)
    print(json.dumps({"added": True, "total_sources": len(session["sources"])}, indent=2))
    return 0


def cmd_reflect(args):
    """Run reflection phase."""
    session_path = get_session_path(args.session)
    if not session_path:
        print("No active session")
        return 1

    session = load_session(session_path)

    print("=" * 60)
    print("  REFLECTION PHASE")
    print("=" * 60)
    print()

    # Generate reflection prompts
    findings_count = len(session["findings"])
    sources_count = len(session["sources"])
    contradictions = len(session["contradictions"])
    gaps = len(session["gaps"])

    reflection = {
        "session": session["name"],
        "topic": session["topic"],
        "stats": {
            "findings": findings_count,
            "sources": sources_count,
            "contradictions": contradictions,
            "gaps": gaps,
        },
        "reflection_prompts": [
            f"1. What did you expect to find that you didn't? (Gaps: {gaps} identified)",
            f"2. What contradicts what you thought? (Contradictions: {contradictions} found)",
            f"3. What new questions arose from findings?",
            f"4. What are your biases? Are you confirming or investigating?",
            f"5. What would a skeptic say about your findings?",
            f"6. What's your confidence level? (Current: {session['confidence'] or 'not set'})",
        ],
        "credibility_breakdown": {},
    }

    # Calculate credibility breakdown
    for s in session["sources"]:
        cred = s.get("credibility", 3)
        key = f"{cred}/5"
        reflection["credibility_breakdown"][key] = reflection["credibility_breakdown"].get(key, 0) + 1

    print(json.dumps(reflection, indent=2))

    # Log reflection
    session["research_log"].append({
        "step": len(session["research_log"]) + 1,
        "phase": "reflection",
        "timestamp": timestamp(),
        "content": f"Reflection phase. Stats: {findings_count} findings, {sources_count} sources, {contradictions} contradictions, {gaps} gaps",
    })
    save_session(session_path, session)
    return 0


def cmd_report(args):
    """Generate research report."""
    session_path = get_session_path(args.session)
    if not session_path:
        print("No active session")
        return 1

    session = load_session(session_path)

    # Generate markdown report
    report = f"""# Research Report: {session['topic']}

**Date**: {timestamp()}
**Session**: {session['name']}
**Status**: {session['status']}
**Confidence**: {session['confidence'] or 'Not assessed'}

## Executive Summary

{session.get('summary', '[Summary not yet written. Add with: python3 deep_research.py summary "..."]')}

## Research Questions

"""
    for i, q in enumerate(session["sub_questions"], 1):
        report += f"{i}. {q}\n"

    report += f"\n## Key Findings ({len(session['findings'])})\n\n"
    for i, f in enumerate(session["findings"], 1):
        cred = f" (credibility: {f['credibility']}/5)" if f.get("credibility") else ""
        src = f" — [source]({f['source_url']})" if f.get("source_url") else ""
        report += f"{i}. {f['content']}{cred}{src}\n"

    if session["contradictions"]:
        report += f"\n## Contradictions Found ({len(session['contradictions'])})\n\n"
        for c in session["contradictions"]:
            report += f"- {c}\n"

    if session["gaps"]:
        report += f"\n## Gaps and Limitations ({len(session['gaps'])})\n\n"
        for g in session["gaps"]:
            report += f"- {g}\n"

    report += f"\n## Sources ({len(session['sources'])})\n\n"
    for i, s in enumerate(session["sources"], 1):
        report += f"{i}. [{s['title']}]({s['url']}) — {s['type']} — credibility {s['credibility']}/5 — accessed {s['accessed']}\n"

    # Save report
    report_path = os.path.join(session_path, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    session["report"] = report_path
    session["status"] = "complete"
    save_session(session_path, session)

    print(json.dumps({
        "report_generated": True,
        "path": report_path,
        "sections": ["summary", "findings", "contradictions", "gaps", "sources"],
        "stats": {
            "findings": len(session["findings"]),
            "sources": len(session["sources"]),
            "contradictions": len(session["contradictions"]),
            "gaps": len(session["gaps"]),
        },
    }, indent=2))
    return 0


def cmd_activity(args):
    """Show research activity log."""
    session_path = get_session_path(args.session)
    if not session_path:
        print("No active session")
        return 1

    session = load_session(session_path)

    for entry in session["research_log"]:
        print(f"[Step {entry['step']}] {entry['phase'].upper()} ({entry['timestamp']})")
        print(f"  {entry['content']}")
        print()

    return 0


def cmd_sessions(args):
    """List all research sessions."""
    if not os.path.exists(SESSIONS_DIR):
        print("No sessions found")
        return 0

    sessions = []
    for name in sorted(os.listdir(SESSIONS_DIR), reverse=True):
        session_path = os.path.join(SESSIONS_DIR, name)
        data = load_session(session_path)
        if data:
            sessions.append({
                "name": name,
                "topic": data["topic"][:60],
                "status": data["status"],
                "findings": len(data["findings"]),
                "sources": len(data["sources"]),
                "created": data["created"],
            })

    print(json.dumps({"total": len(sessions), "sessions": sessions[:10]}, indent=2))
    return 0


def cmd_contradiction(args):
    """Log a contradiction."""
    session_path = get_session_path(args.session)
    if not session_path:
        print("No active session")
        return 1

    session = load_session(session_path)
    session["contradictions"].append(args.description)

    session["research_log"].append({
        "step": len(session["research_log"]) + 1,
        "phase": "evaluation",
        "timestamp": timestamp(),
        "content": f"Contradiction found: {args.description}",
    })

    save_session(session_path, session)
    print(json.dumps({"logged": True, "total_contradictions": len(session["contradictions"])}, indent=2))
    return 0


def cmd_gap(args):
    """Log a knowledge gap."""
    session_path = get_session_path(args.session)
    if not session_path:
        print("No active session")
        return 1

    session = load_session(session_path)
    session["gaps"].append(args.description)

    session["research_log"].append({
        "step": len(session["research_log"]) + 1,
        "phase": "reflection",
        "timestamp": timestamp(),
        "content": f"Gap identified: {args.description}",
    })

    save_session(session_path, session)
    print(json.dumps({"logged": True, "total_gaps": len(session["gaps"])}, indent=2))
    return 0


def cmd_confidence(args):
    """Set confidence level."""
    session_path = get_session_path(args.session)
    if not session_path:
        print("No active session")
        return 1

    session = load_session(session_path)
    session["confidence"] = args.level

    session["research_log"].append({
        "step": len(session["research_log"]) + 1,
        "phase": "reflection",
        "timestamp": timestamp(),
        "content": f"Confidence set: {args.level}",
    })

    save_session(session_path, session)
    print(json.dumps({"confidence": args.level}, indent=2))
    return 0


def cmd_summary(args):
    """Set executive summary."""
    session_path = get_session_path(args.session)
    if not session_path:
        print("No active session")
        return 1

    session = load_session(session_path)
    session["summary"] = args.text
    save_session(session_path, session)
    print(json.dumps({"summary_set": True}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description="Deep Research — Research session manager")
    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Start new research session")
    p_init.add_argument("topic", help="Research topic or question")
    p_init.add_argument("--questions", help="Sub-questions separated by |")

    # plan
    p_plan = sub.add_parser("plan", help="Show/update research plan")
    p_plan.add_argument("--add-question", help="Add a sub-question")
    p_plan.add_argument("--session", help="Session name")

    # log
    p_log = sub.add_parser("log", help="Log a finding")
    p_log.add_argument("finding", help="Finding description")
    p_log.add_argument("--source", help="Source URL")
    p_log.add_argument("--credibility", type=int, help="Credibility score 1-5")
    p_log.add_argument("--type", help="Finding type: data, quote, analysis, observation")
    p_log.add_argument("--session", help="Session name")

    # source
    p_source = sub.add_parser("source", help="Add a source")
    p_source.add_argument("url", help="Source URL")
    p_source.add_argument("title", help="Source title")
    p_source.add_argument("source_type", help="Type: news, academic, blog, social, primary")
    p_source.add_argument("--credibility", type=int, help="Credibility 1-5")
    p_source.add_argument("--session", help="Session name")

    # reflect
    p_reflect = sub.add_parser("reflect", help="Run reflection phase")
    p_reflect.add_argument("--session", help="Session name")

    # report
    p_report = sub.add_parser("report", help="Generate research report")
    p_report.add_argument("--session", help="Session name")

    # activity
    p_activity = sub.add_parser("activity", help="Show research activity log")
    p_activity.add_argument("--session", help="Session name")

    # sessions
    sub.add_parser("sessions", help="List all sessions")

    # contradiction
    p_contra = sub.add_parser("contradiction", help="Log a contradiction")
    p_contra.add_argument("description", help="Contradiction description")
    p_contra.add_argument("--session", help="Session name")

    # gap
    p_gap = sub.add_parser("gap", help="Log a knowledge gap")
    p_gap.add_argument("description", help="Gap description")
    p_gap.add_argument("--session", help="Session name")

    # confidence
    p_conf = sub.add_parser("confidence", help="Set confidence level")
    p_conf.add_argument("level", choices=["high", "medium", "low"], help="Confidence level")
    p_conf.add_argument("--session", help="Session name")

    # summary
    p_summary = sub.add_parser("summary", help="Set executive summary")
    p_summary.add_argument("text", help="Summary text")
    p_summary.add_argument("--session", help="Session name")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "init": cmd_init,
        "plan": cmd_plan,
        "log": cmd_log,
        "source": cmd_source,
        "reflect": cmd_reflect,
        "report": cmd_report,
        "activity": cmd_activity,
        "sessions": cmd_sessions,
        "contradiction": cmd_contradiction,
        "gap": cmd_gap,
        "confidence": cmd_confidence,
        "summary": cmd_summary,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
