#!/usr/bin/env python3
"""
Forecast History — Self-Learning Calibration Tracker

Tracks forecast history, measures accuracy, and learns from mistakes.
Over time, this builds a calibration profile that improves future forecasts.

Usage:
  python3 forecast_history.py init                          # Initialize tracker
  python3 forecast_history.py log --question "X?" --prob 65 --method "polymarket"
  python3 forecast_history.py resolve --id "f-xxx" --outcome true
  python3 forecast_history.py calibration                   # Show calibration stats
  python3 forecast_history.py lessons                       # Extract lessons from mistakes
  python3 forecast_history.py report                        # Full accuracy report
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)


def get_state_dir():
    """Return a writable state directory."""
    override = os.environ.get("HERMES_DEEP_RESEARCH_STATE_DIR")
    if override:
        return override
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        return os.path.join(hermes_home, "state", "deep-research")
    return str(Path.home() / ".hermes" / "state" / "deep-research")


STATE_DIR = get_state_dir()
HISTORY_FILE = os.path.join(STATE_DIR, "forecast_history.json")


def timestamp():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")


def date_slug():
    return datetime.now().astimezone().strftime("%Y-%m-%d")


def load_history() -> dict:
    """Load forecast history."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "forecasts": [],
        "calibration": {
            "total": 0,
            "resolved": 0,
            "correct": 0,
            "brier_score": None,
            "calibration_curve": {},
        },
        "lessons": [],
        "topic_biases": {},
        "method_performance": {},
        "created": timestamp(),
    }


def save_history(data: dict):
    """Save forecast history."""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def calculate_brier_score(forecasts: List[dict]) -> float:
    """
    Calculate Brier score for resolved forecasts.
    Brier = (1/N) * Σ(forecast - outcome)²
    Lower is better (0 = perfect, 1 = worst).
    """
    resolved = [f for f in forecasts if f.get("resolved") and f.get("outcome") is not None]
    if not resolved:
        return None
    
    scores = []
    for f in resolved:
        prob = f["probability"] / 100.0  # Convert to 0-1 scale
        outcome = 1.0 if f["outcome"] else 0.0
        scores.append((prob - outcome) ** 2)
    
    return sum(scores) / len(scores)


def calculate_calibration_curve(forecasts: List[dict]) -> dict:
    """
    Calculate calibration curve.
    Groups forecasts by probability buckets and compares to actual outcomes.
    """
    buckets = {
        "0-20": {"forecasts": [], "outcomes": []},
        "20-40": {"forecasts": [], "outcomes": []},
        "40-60": {"forecasts": [], "outcomes": []},
        "60-80": {"forecasts": [], "outcomes": []},
        "80-100": {"forecasts": [], "outcomes": []},
    }
    
    for f in forecasts:
        if not f.get("resolved") or f.get("outcome") is None:
            continue
        
        prob = f["probability"]
        outcome = 1 if f["outcome"] else 0
        
        if prob < 20:
            buckets["0-20"]["forecasts"].append(prob)
            buckets["0-20"]["outcomes"].append(outcome)
        elif prob < 40:
            buckets["20-40"]["forecasts"].append(prob)
            buckets["20-40"]["outcomes"].append(outcome)
        elif prob < 60:
            buckets["40-60"]["forecasts"].append(prob)
            buckets["40-60"]["outcomes"].append(outcome)
        elif prob < 80:
            buckets["60-80"]["forecasts"].append(prob)
            buckets["60-80"]["outcomes"].append(outcome)
        else:
            buckets["80-100"]["forecasts"].append(prob)
            buckets["80-100"]["outcomes"].append(outcome)
    
    curve = {}
    for bucket, data in buckets.items():
        if data["forecasts"]:
            avg_forecast = sum(data["forecasts"]) / len(data["forecasts"])
            actual_rate = sum(data["outcomes"]) / len(data["outcomes"]) * 100
            curve[bucket] = {
                "avg_forecast": round(avg_forecast, 1),
                "actual_rate": round(actual_rate, 1),
                "count": len(data["forecasts"]),
                "calibration_error": round(abs(avg_forecast - actual_rate), 1),
            }
    
    return curve


def detect_topic_bias(forecasts: List[dict]) -> dict:
    """Detect bias patterns by topic category."""
    categories = {}
    
    for f in forecasts:
        if not f.get("resolved"):
            continue
        
        topic = f.get("topic_category", "general")
        if topic not in categories:
            categories[topic] = {"count": 0, "correct": 0, "overconfident": 0, "underconfident": 0}
        
        categories[topic]["count"] += 1
        if f.get("correct"):
            categories[topic]["correct"] += 1
        
        # Check over/underconfidence
        prob = f["probability"]
        outcome = f["outcome"]
        if outcome and prob < 50:
            categories[topic]["underconfident"] += 1
        elif not outcome and prob > 50:
            categories[topic]["overconfident"] += 1
    
    # Calculate accuracy by category
    for cat, stats in categories.items():
        if stats["count"] > 0:
            stats["accuracy"] = round(stats["correct"] / stats["count"] * 100, 1)
            stats["bias_direction"] = (
                "overconfident" if stats["overconfident"] > stats["underconfident"] else
                "underconfident" if stats["underconfident"] > stats["overconfident"] else
                "calibrated"
            )
    
    return categories


def cmd_init(args):
    """Initialize forecast tracker."""
    data = {
        "forecasts": [],
        "calibration": {
            "total": 0,
            "resolved": 0,
            "correct": 0,
            "brier_score": None,
            "calibration_curve": {},
        },
        "lessons": [],
        "topic_biases": {},
        "method_performance": {},
        "created": timestamp(),
    }
    save_history(data)
    print(json.dumps({"initialized": True, "file": HISTORY_FILE}, indent=2))
    return 0


def cmd_log(args):
    """Log a new forecast."""
    data = load_history()
    
    forecast_id = f"f-{uuid.uuid4().hex[:8]}"
    
    forecast = {
        "id": forecast_id,
        "question": args.question,
        "probability": args.prob,
        "probability_range": args.range,
        "method": args.method,
        "methodology": args.methodology,
        "confidence": args.confidence,
        "topic_category": args.category,
        "created": timestamp(),
        "resolve_by": args.resolve_by,
        "sources": args.sources.split("|") if args.sources else [],
        "notes": args.notes,
        "resolved": False,
        "outcome": None,
        "correct": None,
        "resolved_at": None,
    }
    
    data["forecasts"].append(forecast)
    data["calibration"]["total"] = len(data["forecasts"])
    
    # Track method usage
    method = args.method
    if method not in data["method_performance"]:
        data["method_performance"][method] = {"count": 0, "correct": 0}
    data["method_performance"][method]["count"] += 1
    
    save_history(data)
    
    print(json.dumps({
        "logged": True,
        "forecast_id": forecast_id,
        "question": args.question,
        "probability": f"{args.prob}%",
        "method": args.method,
    }, indent=2))
    return 0


def cmd_resolve(args):
    """Resolve a forecast with actual outcome."""
    data = load_history()
    
    # Find the forecast
    forecast = None
    for f in data["forecasts"]:
        if f["id"] == args.id:
            forecast = f
            break
    
    if not forecast:
        print(f"Forecast not found: {args.id}")
        return 1
    
    # Update forecast
    forecast["resolved"] = True
    forecast["outcome"] = args.outcome
    forecast["resolved_at"] = timestamp()
    
    # Calculate if forecast was "correct"
    # A forecast is correct if:
    # - Predicted >50% and outcome was True, OR
    # - Predicted <50% and outcome was False
    prob = forecast["probability"]
    outcome = args.outcome
    
    if (prob > 50 and outcome) or (prob < 50 and not outcome):
        forecast["correct"] = True
    elif prob == 50:
        forecast["correct"] = None  # Can't be correct/incorrect at exactly 50%
    else:
        forecast["correct"] = False
    
    # Update calibration
    resolved = [f for f in data["forecasts"] if f.get("resolved")]
    data["calibration"]["resolved"] = len(resolved)
    data["calibration"]["correct"] = sum(1 for f in resolved if f.get("correct"))
    data["calibration"]["brier_score"] = calculate_brier_score(data["forecasts"])
    data["calibration"]["calibration_curve"] = calculate_calibration_curve(data["forecasts"])
    
    # Update method performance
    method = forecast.get("method")
    if method and method in data["method_performance"]:
        if forecast["correct"]:
            data["method_performance"][method]["correct"] += 1
    
    # Update topic biases
    data["topic_biases"] = detect_topic_bias(data["forecasts"])
    
    # Generate lesson if forecast was wrong
    if forecast["correct"] == False and args.lesson:
        lesson = {
            "forecast_id": args.id,
            "question": forecast["question"],
            "predicted": f"{prob}%",
            "outcome": "Yes" if outcome else "No",
            "lesson": args.lesson,
            "category": forecast.get("topic_category", "general"),
            "timestamp": timestamp(),
            "time_decay": 1.0,  # Decreases over time
        }
        data["lessons"].append(lesson)
    
    save_history(data)
    
    print(json.dumps({
        "resolved": True,
        "forecast_id": args.id,
        "outcome": "Yes" if args.outcome else "No",
        "forecast_correct": forecast["correct"],
        "brier_score": data["calibration"]["brier_score"],
    }, indent=2))
    return 0


def cmd_calibration(args):
    """Show calibration statistics."""
    data = load_history()
    cal = data["calibration"]
    
    result = {
        "total_forecasts": cal["total"],
        "resolved": cal["resolved"],
        "accuracy": round(cal["correct"] / cal["resolved"] * 100, 1) if cal["resolved"] > 0 else None,
        "brier_score": cal["brier_score"],
        "interpretation": interpret_brier(cal["brier_score"]),
        "calibration_curve": cal["calibration_curve"],
        "topic_biases": data["topic_biases"],
        "method_performance": data["method_performance"],
    }
    
    print(json.dumps(result, indent=2))
    
    # Print human-readable summary
    print()
    print("=" * 50)
    print("  CALIBRATION SUMMARY")
    print("=" * 50)
    
    if cal["resolved"] > 0:
        print(f"  Resolved forecasts: {cal['resolved']}")
        print(f"  Accuracy (binary):  {result['accuracy']}%")
        if cal["brier_score"]:
            print(f"  Brier score:        {cal['brier_score']:.3f} ({interpret_brier(cal['brier_score'])})")
    
    if cal["calibration_curve"]:
        print()
        print("  CALIBRATION CURVE:")
        for bucket, stats in cal["calibration_curve"].items():
            print(f"    {bucket}: predicted {stats['avg_forecast']:.0f}% → actual {stats['actual_rate']:.0f}% (n={stats['count']})")
    
    return 0


def interpret_brier(score: Optional[float]) -> str:
    """Interpret Brier score."""
    if score is None:
        return "No resolved forecasts"
    if score <= 0.1:
        return "Excellent (very well calibrated)"
    if score <= 0.2:
        return "Good"
    if score <= 0.3:
        return "Fair"
    if score <= 0.5:
        return "Poor (needs improvement)"
    return "Very poor (major calibration issues)"


def cmd_lessons(args):
    """Show lessons learned from incorrect forecasts."""
    data = load_history()
    
    if not data["lessons"]:
        print("No lessons recorded yet. Lessons are added when resolving incorrect forecasts with --lesson.")
        return 0
    
    # Apply time decay (30-day half-life)
    now = datetime.now().astimezone()
    for lesson in data["lessons"]:
        if lesson.get("timestamp"):
            lesson_time = datetime.fromisoformat(lesson["timestamp"])
            days_old = (now - lesson_time).days
            lesson["time_decay"] = max(0.1, 0.5 ** (days_old / 30))
    
    # Sort by recency and time-decay
    lessons = sorted(data["lessons"], key=lambda x: x.get("timestamp", ""), reverse=True)
    
    if args.category:
        lessons = [l for l in lessons if l.get("category") == args.category]
    
    print(f"LESSONS LEARNED ({len(lessons)} total)")
    print("=" * 60)
    
    for i, lesson in enumerate(lessons[:20], 1):
        print(f"\n{i}. [{lesson.get('category', 'general')}] {lesson['question'][:60]}...")
        print(f"   Predicted: {lesson['predicted']} | Outcome: {lesson['outcome']}")
        print(f"   Lesson: {lesson['lesson']}")
        print(f"   Decay: {lesson['time_decay']:.2f}")
    
    return 0


def cmd_report(args):
    """Generate full accuracy report."""
    data = load_history()
    
    # Calculate various stats
    resolved = [f for f in data["forecasts"] if f.get("resolved")]
    unresolved = [f for f in data["forecasts"] if not f.get("resolved")]
    
    # By method
    method_stats = {}
    for f in resolved:
        method = f.get("method", "unknown")
        if method not in method_stats:
            method_stats[method] = {"total": 0, "correct": 0, "brier": []}
        method_stats[method]["total"] += 1
        if f.get("correct"):
            method_stats[method]["correct"] += 1
        # Brier component
        prob = f["probability"] / 100
        outcome = 1 if f["outcome"] else 0
        method_stats[method]["brier"].append((prob - outcome) ** 2)
    
    for method, stats in method_stats.items():
        stats["accuracy"] = round(stats["correct"] / stats["total"] * 100, 1) if stats["total"] > 0 else None
        stats["brier"] = round(sum(stats["brier"]) / len(stats["brier"]), 3) if stats["brier"] else None
    
    # By topic
    topic_stats = {}
    for f in resolved:
        topic = f.get("topic_category", "general")
        if topic not in topic_stats:
            topic_stats[topic] = {"total": 0, "correct": 0}
        topic_stats[topic]["total"] += 1
        if f.get("correct"):
            topic_stats[topic]["correct"] += 1
    
    for topic, stats in topic_stats.items():
        stats["accuracy"] = round(stats["correct"] / stats["total"] * 100, 1) if stats["total"] > 0 else None
    
    report = {
        "summary": {
            "total_forecasts": len(data["forecasts"]),
            "resolved": len(resolved),
            "unresolved": len(unresolved),
            "overall_accuracy": round(sum(1 for f in resolved if f.get("correct")) / len(resolved) * 100, 1) if resolved else None,
            "brier_score": data["calibration"]["brier_score"],
        },
        "by_method": method_stats,
        "by_topic": topic_stats,
        "calibration_curve": data["calibration"]["calibration_curve"],
        "lessons_count": len(data["lessons"]),
        "recommendations": generate_recommendations(data),
    }
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {args.output}")
    else:
        print(json.dumps(report, indent=2))
    
    return 0


def generate_recommendations(data: dict) -> List[str]:
    """Generate recommendations based on calibration data."""
    recommendations = []
    
    # Check calibration curve
    curve = data["calibration"].get("calibration_curve", {})
    for bucket, stats in curve.items():
        if stats.get("calibration_error", 0) > 20:
            if stats["avg_forecast"] > stats["actual_rate"]:
                recommendations.append(f"You're overconfident in {bucket}% range. Predicted {stats['avg_forecast']:.0f}% but only {stats['actual_rate']:.0f}% came true. Dial back predictions.")
            else:
                recommendations.append(f"You're underconfident in {bucket}% range. Predicted {stats['avg_forecast']:.0f}% but {stats['actual_rate']:.0f}% came true. Be bolder.")
    
    # Check topic biases
    for topic, stats in data.get("topic_biases", {}).items():
        if stats.get("count", 0) >= 3:
            if stats.get("bias_direction") == "overconfident":
                recommendations.append(f"You tend to be overconfident on '{topic}' topics. Consider wider probability ranges or more skepticism.")
            elif stats.get("bias_direction") == "underconfident":
                recommendations.append(f"You tend to be underconfident on '{topic}' topics. Your predictions are better than you think.")
    
    # Check Brier score
    brier = data["calibration"].get("brier_score")
    if brier and brier > 0.3:
        recommendations.append("Your Brier score is high (>0.3). Focus on: (1) using prediction markets more, (2) researching base rates, (3) being more humble with extreme predictions.")
    
    return recommendations


def cmd_list(args):
    """List forecasts (optionally unresolved only)."""
    data = load_history()
    
    forecasts = data["forecasts"]
    if args.unresolved:
        forecasts = [f for f in forecasts if not f.get("resolved")]
    
    if args.category:
        forecasts = [f for f in forecasts if f.get("topic_category") == args.category]
    
    forecasts = sorted(forecasts, key=lambda x: x.get("created", ""), reverse=True)
    
    print(f"FORECASTS ({len(forecasts)} shown)")
    print("=" * 60)
    
    for f in forecasts[:20]:
        status = "RESOLVED" if f.get("resolved") else "PENDING"
        outcome = f"→ {'Yes' if f['outcome'] else 'No'}" if f.get("resolved") else ""
        print(f"  [{f['id']}] {f['probability']}% {status} {outcome}")
        print(f"      {f['question'][:70]}...")
        print()
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="Forecast History — Self-Learning Calibration Tracker")
    sub = parser.add_subparsers(dest="command")
    
    # init
    sub.add_parser("init", help="Initialize forecast tracker")
    
    # log
    p_log = sub.add_parser("log", help="Log a new forecast")
    p_log.add_argument("--question", required=True, help="The forecasting question")
    p_log.add_argument("--prob", type=float, required=True, help="Probability estimate (0-100)")
    p_log.add_argument("--range", help="Probability range (e.g., '60-70')")
    p_log.add_argument("--method", required=True, help="Method used (polymarket, analyst, base_rate, intuition)")
    p_log.add_argument("--methodology", help="Detailed methodology")
    p_log.add_argument("--confidence", choices=["high", "medium", "low"], default="medium")
    p_log.add_argument("--category", help="Topic category (geopolitical, market, technology)")
    p_log.add_argument("--resolve-by", help="Expected resolution date")
    p_log.add_argument("--sources", help="Sources used, pipe-separated")
    p_log.add_argument("--notes", help="Additional notes")
    
    # resolve
    p_resolve = sub.add_parser("resolve", help="Resolve a forecast")
    p_resolve.add_argument("--id", required=True, help="Forecast ID")
    p_resolve.add_argument("--outcome", type=lambda x: x.lower() == "true", required=True, help="Actual outcome (true/false)")
    p_resolve.add_argument("--lesson", help="Lesson learned (for incorrect forecasts)")
    
    # calibration
    sub.add_parser("calibration", help="Show calibration statistics")
    
    # lessons
    p_lessons = sub.add_parser("lessons", help="Show lessons learned")
    p_lessons.add_argument("--category", help="Filter by category")
    
    # report
    p_report = sub.add_parser("report", help="Generate full accuracy report")
    p_report.add_argument("--output", help="Output file path")
    
    # list
    p_list = sub.add_parser("list", help="List forecasts")
    p_list.add_argument("--unresolved", action="store_true", help="Only show unresolved")
    p_list.add_argument("--category", help="Filter by category")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    commands = {
        "init": cmd_init,
        "log": cmd_log,
        "resolve": cmd_resolve,
        "calibration": cmd_calibration,
        "lessons": cmd_lessons,
        "report": cmd_report,
        "list": cmd_list,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())