# swarm/merge — Semantic Conflict Resolution

Resolves conflicting file edits produced by multiple agents working in parallel.

## Core types

- **`Conflict`** — a region where two or more agents edited the same lines.
- **`Resolution`** — the chosen outcome for a conflict (which agent wins, or a merged result).
- **`ConflictResolver`** — stateless resolver with pluggable strategy.

## Strategies

| Strategy | Behaviour |
|----------|-----------|
| `auto`   | Resolve trivial (all-agree) conflicts; leave true conflicts with markers. |
| `ours`   | Always pick the first agent's version. |
| `theirs` | Always pick the last agent's version. |
| `manual` | Never auto-resolve; all conflicts get markers. |

## Usage

```python
from swarm.merge import ConflictResolver

resolver = ConflictResolver(strategy="auto")

base = open("file.py").read()
changes = [
    ("agent-1", agent1_version),
    ("agent-2", agent2_version),
]

conflicts = resolver.detect_conflicts(base, changes, file_path="file.py")
resolutions = resolver.resolve_trivial(conflicts)
merged = resolver.merge_changes(base, changes, conflicts, resolutions)

# Or use three_way_merge for the common two-agent case:
merged, unresolved = resolver.three_way_merge(base, ours, theirs)

# Human-readable summary:
print(resolver.format_conflict_report(unresolved))
```

## How it works

1. `detect_conflicts` diffs each agent's output against the base using `difflib.SequenceMatcher`, collects changed regions, and clusters overlapping regions touched by different agents.
2. `resolve_trivial` auto-resolves conflicts where all agents agree or where the strategy dictates a winner.
3. `merge_changes` applies non-conflicting edits plus resolutions; unresolved conflicts emit git-style conflict markers.
4. `three_way_merge` is a convenience wrapper for the two-agent case.

No external dependencies — uses only `difflib` from the standard library.
