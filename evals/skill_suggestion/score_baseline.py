"""Score the baseline JSONL: exact-match vs category-path-resolvable loads."""
import json
from pathlib import Path

REC = Path(__file__).with_name("skill_baseline_results.jsonl")
RESOLVABLE = {"email/himalaya": "himalaya", "software-development:test-driven-development": "test-driven-development"}

recs = [json.loads(l) for l in REC.read_text().splitlines()]
pos = [r for r in recs if r["gold"]]
neg = [r for r in recs if not r["gold"]]

wrong_exact = [r for r in pos if r["loaded"] != r["gold"]]
wrong_strict = [r for r in pos if RESOLVABLE.get(r["loaded"], r["loaded"]) != r["gold"]]
needless = [r for r in neg if r["loaded"] is not None]

print(f"covered: {len(pos)}  uncovered: {len(neg)}")
print(f"wrong_load (exact name):  {len(wrong_exact)}/{len(pos)} = {len(wrong_exact)/len(pos):.1%}")
print(f"wrong_load (path-resolvable): {len(wrong_strict)}/{len(pos)} = {len(wrong_strict)/len(pos):.1%}")
print(f"needless_load:            {len(needless)}/{len(neg)} = {len(needless)/len(neg):.1%}")
print(f"\nwrong loads (strict):")
for r in wrong_strict:
    print(f"  gold={r['gold']!r:26} got={r['loaded']!r}  {r['request'][:60]}")
print(f"\nneedless loads:")
for r in needless:
    print(f"  loaded={r['loaded']!r:20} {r['request'][:60]}")