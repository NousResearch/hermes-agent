# Operator Packet Format

Load on demand from Gate 7. Produced after implementation and before commit.

## Questions to answer

1. **What's ready?** — modules implemented, tests green, CLI available
2. **What's blocked?** — what requires operator approval
3. **What needs validation?** — remaining work (research, legal review, etc.)

## Structure

```markdown
# Operator Packet: <MISSION_NAME>

## Scope
- Mission ID: <id>
- Repo: <path>
- Branch: <name>
- SHA: <commit>

## Ready
- Module: <path> — status, test count, coverage
- CLI commands: <list>
- Tests: <pass/fail count>

## Blocked
- Item: <description> — requires: operator approval / legal / security

## Remaining
- Item: <description> — planned for next mission

## Safety
- FastSafe: PASS / FAIL (list failed checks if any)
- publish_allowed: false
- No live mutations confirmed

## Continuity
- Continuity JSON: <path>
- Memory lock: <done/pending>
```

## CLI Usage

For lah-stack-tools missions, output to `runs/<mission>/operator-packet.json`.
