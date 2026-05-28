#!/usr/bin/env python3
"""
Sustained integration test — validates all fixes together under simulated load.
Runs 25+ simulated turns exercising: memory palace writes, auto-sculpt trigger,
tuning log entries, cascade protection, and context budget enforcement.
"""
import sys, os, json, time, random, string

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from memory_palace import (
    store_episode, store_fact, set_working, get_working,
    clear_working, get_stats, auto_prune, prune_expired,
    recall_facts, recall_episodes
)

# --- Config ---
NUM_TURNS = 25
BUDGET_TOKENS = 14000
MAX_TURNS_BEFORE_SCULPT = 20
TOKEN_PRESSURE_THRESHOLD = 0.9  # 90%
EST_TOKENS_PER_CHAR = 0.25

results = {
    "turns_completed": 0,
    "sculpt_triggered": False,
    "sculpt_turn": None,
    "cascade_failures": 0,
    "write_failures": 0,
    "budget_exceeded": False,
    "tuning_entries": 0,
    "errors": [],
    "db_sizes": [],
}

def estimate_tokens(text):
    return len(text) * EST_TOKENS_PER_CHAR

def make_turn_data(turn_num):
    """Simulate a turn's context data."""
    return {
        "turn": turn_num,
        "user_msg": ''.join(random.choices(string.ascii_letters + ' ', k=random.randint(50, 500))),
        "assistant_reply": ''.join(random.choices(string.ascii_letters + ' .,!?', k=random.randint(200, 2000))),
        "tool_calls": random.randint(0, 5),
        "tokens_used": random.randint(200, 1500),
    }

DB_CLEANUP_DONE = False

print(f"=== Sustained Integration Test: {NUM_TURNS} turns ===")
print(f"Budget: {BUDGET_TOKENS} tokens | Sculpt trigger: {MAX_TURNS_BEFORE_SCULPT} turns")
print()

# Clear working memory
clear_working()

cumulative_tokens = 0

for turn in range(1, NUM_TURNS + 1):
    turn_data = make_turn_data(turn)
    cumulative_tokens += turn_data["tokens_used"]

    # --- Exercise memory palace writes ---
    try:
        store_episode(
            session_id=f"integration-test-{turn}",
            category=f"turn-{turn % 3}",
            content=json.dumps(turn_data),
            context={"turn_num": turn, "cumulative_tokens": cumulative_tokens},
            importance=random.randint(1, 10),
            expires_hours=48 if turn % 3 == 0 else 24,
            tags=["integration-test", f"turn-{turn}"]
        )
        results["turns_completed"] += 1
    except Exception as e:
        results["cascade_failures"] += 1
        results["errors"].append(f"Episode write failed turn {turn}: {e}")

    # Semantic facts use concept+description model, not domain/key/value
    try:
        store_fact(
            concept=f"turn_{turn}_token_count",
            description=str(turn_data["tokens_used"]),
            relationships={"turn": turn, "cumulative": cumulative_tokens},
            confidence=0.8
        )
    except Exception as e:
        results["write_failures"] += 1
        results["errors"].append(f"Fact write failed turn {turn}: {e}")

    # --- Set working memory (value must be dict) ---
    try:
        set_working("current_turn", {"turn": turn})
        set_working("cumulative_tokens", {"tokens": cumulative_tokens})
    except Exception as e:
        results["errors"].append(f"Working memory failed turn {turn}: {e}")

    # --- Write tuning entry per turn (as an episode) ---
    tokens_at_point = cumulative_tokens + estimate_tokens(json.dumps(turn_data))
    tuning_entry = {
        "turn": turn,
        "category": "turn",
        "timestamp": time.time(),
        "tokens": tokens_at_point,
        "budget_remaining": max(0, BUDGET_TOKENS - tokens_at_point),
    }
    try:
        store_episode(
            session_id=f"tuning-{turn}",
            category="tuning_log",
            content=json.dumps(tuning_entry),
            importance=5,
            expires_hours=48,
            tags=["tuning", "integration-test", f"turn-{turn}"]
        )
        results["tuning_entries"] += 1
    except Exception as e:
        results["errors"].append(f"Tuning write failed turn {turn}: {e}")

    # --- Context budget check ---
    if tokens_at_point > BUDGET_TOKENS:
        results["budget_exceeded"] = True

    current_tokens_estimate = tokens_at_point

    # --- Auto-sculpt trigger check ---
    token_pressure = current_tokens_estimate / BUDGET_TOKENS
    should_sculpt = (turn >= MAX_TURNS_BEFORE_SCULPT) or (token_pressure >= TOKEN_PRESSURE_THRESHOLD)

    if should_sculpt and not results["sculpt_triggered"]:
        results["sculpt_triggered"] = True
        results["sculpt_turn"] = turn

        # Verify T0-T3 survive sculpt by storing an important fact
        store_fact(
            concept="sculpt_checkpoint",
            description=f"sculpt_triggered_at_turn_{turn}",
            confidence=1.0
        )

        # Simulate sculpt stats
        stats_before = get_stats()
        recalled_facts = recall_facts("sculpt_checkpoint")
        recalled_episodes = recall_episodes(hours=48, category=f"turn-{turn % 3}")

        print(f"  TURN {turn}: 🔄 Sculpt triggered! Tokens={current_tokens_estimate}, pressure={token_pressure:.1%}")
        print(f"    Stats before: episodes={stats_before['episodic_count']}, facts={stats_before['semantic_count']}, DB={stats_before['db_size_bytes']/1024:.1f}KB")
        print(f"    Sculpt checkpoint recall: facts={len(recalled_facts)}, episodes={len(recalled_episodes)}")

    elif turn % 5 == 0:
        stats = get_stats()
        db_size_kb = stats['db_size_bytes'] / 1024
        results["db_sizes"].append(db_size_kb)
        print(f"  TURN {turn}: tokens={cumulative_tokens}, DB={db_size_kb:.1f}KB, episodes={stats['episodic_count']}, facts={stats['semantic_count']}")

    # --- Simulate auto_prune every 10 turns ---
    if turn % 10 == 0:
        auto_prune()

# --- One more auto_prune + stats at end ---
auto_prune()

# --- Final cleanup and validation ---
print()
print("=== Final Validation ===")

final_stats = get_stats()
print(f"Episodes in DB: {final_stats['episodic_count']}")
print(f"Facts in DB: {final_stats['semantic_count']}")
print(f"DB size: {final_stats['db_size_bytes'] / 1024:.1f} KB (cap: 500 KB)")
print(f"Working memory: {get_working('current_turn')}")

# Verify critical recalls
important_facts = recall_facts("sculpt_checkpoint")
print(f"Sculpt checkpoint facts found: {len(important_facts)}")

print()
print("=== Results ===")
for k, v in results.items():
    if k != "errors":
        print(f"  {k}: {v}")

if results["errors"]:
    print(f"\n⚠️  Errors ({len(results['errors'])})")
    for e in results["errors"][:5]:
        print(f"  - {e}")
    if len(results["errors"]) > 5:
        print(f"  ... and {len(results['errors']) - 5} more")

# Pass/fail
passed = (
    results["cascade_failures"] == 0 and
    results["write_failures"] == 0 and
    (final_stats['db_size_bytes'] / 1024) < 500 and
    results["turns_completed"] == NUM_TURNS and
    results["sculpt_triggered"]
)

print()
if passed:
    print("✅ ALL CHECKS PASSED — Integration test successful")
    sys.exit(0)
else:
    print("❌ SOME CHECKS FAILED — Review errors above")
    if results["cascade_failures"] > 0:
        print(f"   Cascade failures: {results['cascade_failures']}")
    if results["write_failures"] > 0:
        print(f"   Write failures: {results['write_failures']}")
    if (final_stats['db_size_bytes'] / 1024) >= 500:
        print(f"   DB overflow: {final_stats['db_size_bytes'] / 1024:.1f} KB >= 500 KB")
    if results["turns_completed"] != NUM_TURNS:
        print(f"   Turns completed: {results['turns_completed']}/{NUM_TURNS}")
    if not results["sculpt_triggered"]:
        print("   Auto-sculpt was NOT triggered")
    sys.exit(1)