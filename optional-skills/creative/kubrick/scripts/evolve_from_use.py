#!/usr/bin/env python3
"""
Kubrick Self-Evolution Engine

Evolves the symbolic corpus according to actual use.

Usage:
  python scripts/evolve_from_use.py --receipts-dir references/usage/receipts --outcomes-dir references/usage/outcomes

It:
- Aggregates retrieval receipts
- Incorporates project outcomes (success/failure signals from Forge/revision/ledger)
- Adjusts confidence, adds usage_history to sidecars
- Updates corpus-index with observed performance
- Emits an evolution_receipt
- Never mutates without producing a receipt and provenance note

Run periodically or after significant Forge project activity.
"""

import argparse
import json
import os
import glob
from datetime import datetime
from collections import defaultdict
import hashlib

SKILL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATTERNS_DIR = os.path.join(SKILL_ROOT, "references", "patterns")
USAGE_RECEIPTS = os.path.join(SKILL_ROOT, "references", "usage", "receipts")
USAGE_OUTCOMES = os.path.join(SKILL_ROOT, "references", "usage", "outcomes")
EVOLUTION_DIR = os.path.join(SKILL_ROOT, "references", "evolution")

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def find_sidecar(pattern_id):
    for root, _, files in os.walk(PATTERNS_DIR):
        for f in files:
            if f == f"{pattern_id}.json":
                return os.path.join(root, f)
    return None

def load_all_sidecars():
    sidecars = {}
    for root, _, files in os.walk(PATTERNS_DIR):
        for fname in files:
            if fname.endswith(".json"):
                path = os.path.join(root, fname)
                data = load_json(path)
                sidecars[data.get("pattern_id", fname)] = {"path": path, "data": data}
    return sidecars

def aggregate_usage(receipts_dir, outcomes_dir):
    usage = defaultdict(lambda: {"uses": 0, "total_score": 0.0, "success_signals": 0, "failure_signals": 0, "projects": []})

    # From receipts
    for path in glob.glob(os.path.join(receipts_dir, "*.json")) + glob.glob(os.path.join(receipts_dir, "*.yaml")):
        try:
            if path.endswith(".json"):
                receipt = load_json(path)
            else:
                import yaml
                receipt = yaml.safe_load(open(path))
            rec = receipt.get("retrieval_receipt", receipt)
            ranked = rec.get("ranked_patterns", [])
            for item in ranked[:3]:  # top patterns
                pid = item.get("pattern_id")
                if pid:
                    usage[pid]["uses"] += 1
                    usage[pid]["total_score"] += item.get("total_score", 0.5)
                    usage[pid]["projects"].append(rec.get("request_hash", "unknown"))
        except Exception as e:
            print(f"Warning: could not parse {path}: {e}")

    # From outcomes (simple signals)
    for path in glob.glob(os.path.join(outcomes_dir, "*.json")):
        try:
            outcome = load_json(path)
            pid = outcome.get("pattern_id")
            if pid:
                signal = outcome.get("outcome", "neutral")
                if signal == "success":
                    usage[pid]["success_signals"] += 1
                elif signal in ("failure", "debt", "collision", "revision_broken"):
                    usage[pid]["failure_signals"] += 1
                usage[pid]["projects"].append(outcome.get("project", "unknown"))
        except Exception as e:
            print(f"Warning: could not parse outcome {path}: {e}")

    return usage

def evolve_sidecar(sidecar_info, usage_stats):
    data = sidecar_info["data"]
    pid = data["pattern_id"]
    stats = usage_stats.get(pid, {})
    uses = stats.get("uses", 0)
    if uses == 0:
        return None

    avg_score = stats["total_score"] / uses if uses > 0 else 0.5
    success = stats.get("success_signals", 0)
    failure = stats.get("failure_signals", 0)
    total_signals = success + failure or 1
    success_rate = success / total_signals

    # Compute delta
    current_conf = data.get("confidence", 0.7)
    delta = 0.0

    if uses >= 3:
        delta += (avg_score - 0.6) * 0.15
        delta += (success_rate - 0.5) * 0.25

    # Penalty for repeated failures
    if failure > success and uses >= 2:
        delta -= 0.1

    new_conf = max(0.3, min(0.98, round(current_conf + delta, 4)))

    # Record history
    if "usage_history" not in data:
        data["usage_history"] = []
    data["usage_history"].append({
        "date": datetime.utcnow().isoformat() + "Z",
        "uses_in_window": uses,
        "avg_retrieval_score": round(avg_score, 4),
        "success_rate": round(success_rate, 4),
        "confidence_before": current_conf,
        "confidence_after": new_conf,
        "delta": round(delta, 4),
        "source_projects": list(set(stats.get("projects", [])))[:5]
    })

    data["confidence"] = new_conf
    data["version"] = data.get("version", "0.6.0")
    # bump patch for evolution
    parts = data["version"].split(".")
    if len(parts) >= 3:
        parts[2] = str(int(parts[2]) + 1)
        data["version"] = ".".join(parts)

    data["last_evolved"] = datetime.utcnow().isoformat() + "Z"

    return data

def update_index_from_usage(index_path, usage_stats, sidecars):
    # Light update: promote high performers in by_dramatic_problem hints
    try:
        import yaml
        with open(index_path, "r") as f:
            index = yaml.safe_load(f)
    except:
        return None

    changed = False
    for problem, data in index.get("by_dramatic_problem", {}).items():
        current_patterns = data.get("patterns", [])
        scored = []
        for pid in current_patterns:
            if pid in usage_stats and pid in sidecars:
                score = usage_stats[pid].get("success_signals", 0) + (sidecars[pid]["data"].get("confidence", 0.7) * 2)
                scored.append((pid, score))
        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            new_order = [p[0] for p in scored]
            if new_order != current_patterns:
                data["patterns"] = new_order
                changed = True

    if changed:
        with open(index_path, "w") as f:
            yaml.safe_dump(index, f, sort_keys=False)
        return "corpus-index.yaml updated with performance-based ordering"

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts-dir", default=USAGE_RECEIPTS)
    parser.add_argument("--outcomes-dir", default=USAGE_OUTCOMES)
    args = parser.parse_args()

    sidecars = load_all_sidecars()
    usage = aggregate_usage(args.receipts_dir, args.outcomes_dir)

    evolved = []
    for pid, info in sidecars.items():
        updated = evolve_sidecar(info, usage)
        if updated:
            save_json(info["path"], updated)
            evolved.append(pid)

    index_msg = update_index_from_usage(
        os.path.join(SKILL_ROOT, "references", "corpus-index.yaml"),
        usage,
        sidecars
    )

    evolution_receipt = {
        "evolution_receipt": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "patterns_evolved": evolved,
            "index_update": index_msg,
            "usage_window": {
                "receipts_scanned": len(glob.glob(os.path.join(args.receipts_dir, "*"))),
                "outcomes_scanned": len(glob.glob(os.path.join(args.outcomes_dir, "*")))
            }
        }
    }

    os.makedirs(EVOLUTION_DIR, exist_ok=True)
    receipt_path = os.path.join(EVOLUTION_DIR, f"evolution-{datetime.utcnow().strftime('%Y%m%d-%H%M')}.json")
    save_json(receipt_path, evolution_receipt)

    print(json.dumps(evolution_receipt, indent=2))
    if evolved:
        print(f"\nEvolved {len(evolved)} patterns. Sidecars updated in place with usage_history.")
    else:
        print("\nNo patterns met evolution thresholds in this window.")

if __name__ == "__main__":
    main()
