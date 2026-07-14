# Roundtable Discussion

Multi-agent roundtable discussion system for Hermes Agent. Enable multiple AI agents to participate in structured, multi-round discussions with convergence detection and conclusion generation.

## Quick Start

```python
# 1. Create a discussion
roundtable_init(
    topic="Database selection: PostgreSQL vs MySQL",
    participants=[
        {"profile": "bingge", "role": "Product Director", "perspective": "Focus on UX"},
        {"profile": "mafei", "role": "Tech Lead", "perspective": "Focus on feasibility"},
    ],
    max_rounds=3
)

# 2. Coordinator speaks first (Round 0)
roundtable_speak(discussion_id="rt_xxxxxxxx", participant="coordinator", content="...")

# 3. Participants take turns
roundtable_speak(discussion_id="rt_xxxxxxxx", participant="bingge", content="...")

# 4. Check convergence
roundtable_status(discussion_id="rt_xxxxxxxx")

# 5. Generate conclusion data
roundtable_summarize(discussion_id="rt_xxxxxxxx")

# 6. End discussion
roundtable_end(discussion_id="rt_xxxxxxxx")
```

## Architecture

```
src/
├── hermes_cli/
│   └── roundtable_db.py      # SQLite data layer (5 tables)
├── tools/
│   └── roundtable_tools.py   # Tool handlers (7 tools)
├── skills/
│   └── SKILL.md              # Coordinator flow + participant template
└── toolsets.py               # Toolset registration

tests/
├── hermes_cli/
│   └── test_roundtable_db.py  # DB layer tests (28 tests)
└── tools/
    └── test_roundtable_tools.py  # Tool layer tests (16 tests)
```

## Tools

| Tool | Purpose |
|------|---------|
| `roundtable_init` | Create discussion with topic + participants |
| `roundtable_speak` | Record participant speech (auto-advances rounds) |
| `roundtable_read` | Read discussion history (full or since round N) |
| `roundtable_status` | Check status + convergence metrics |
| `roundtable_summarize` | Get structured data for conclusion document |
| `roundtable_end` | Conclude or force-cancel a discussion |
| `roundtable_list` | List all discussions with optional status filter |

## Data Model

- **discussions** — Topic, status, round tracking, speech order
- **participants** — Registered profiles with roles and perspectives
- **speeches** — Multi-round speeches with reply-to support
- **findings** — Consensus/disagreement/new_point categorization
- **convergence_history** — Per-round convergence scoring

## Configuration

- **Database**: `~/.hermes/roundtable.db` (override with `HERMES_ROUNDTABLE_DB`)
- **Toolset**: Enable `roundtable` in profile config or pass `enabled_toolsets: ["roundtable"]`

## Testing

```bash
cd roundtable
python -m pytest tests/ -v
```

## License

Part of [Hermes Agent](https://github.com/ParsifalC/hermes-agent).
