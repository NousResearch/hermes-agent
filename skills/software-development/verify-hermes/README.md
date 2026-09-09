# Verify Hermes Skill

Comprehensive verification framework for Hermes Agent features.

## What This Is

A systematic 5-phase verification workflow that proves features work correctly across all Hermes surfaces (CLI, TUI, gateway). No dedicated bot - the skill contains everything needed to launch, check, drive, prove, and clean up verifications.

## Quick Start

```bash
# 1. Launch verification for a feature
python .cursor/skills/verify-hermes/scripts/verify.py launch \
  --feature "memory-search" \
  --surfaces cli,tui,gateway

# 2. Check environment health
python .cursor/skills/verify-hermes/scripts/verify.py doctor

# 3. Run tests for each surface
python .cursor/skills/verify-hermes/scripts/verify.py drive --phase cli
python .cursor/skills/verify-hermes/scripts/verify.py drive --phase tui
python .cursor/skills/verify-hermes/scripts/verify.py drive --phase gateway

# 4. Collect evidence and generate report
python .cursor/skills/verify-hermes/scripts/verify.py evidence

# 5. Archive results and cleanup
python .cursor/skills/verify-hermes/scripts/verify.py cleanup
```

## Directory Structure

```
.cursor/skills/verify-hermes/
├── SKILL.md                    # Main skill documentation
├── scripts/
│   └── verify.py              # Control CLI (Launch/Doctor/Drive/Evidence/Cleanup)
├── references/
│   └── feature-map.md         # Map features to CLI/TUI/gateway paths
└── templates/
    ├── test-cli-template.py   # CLI test patterns
    ├── test-tui-template.py   # TUI test patterns
    └── test-gateway-template.py  # Gateway test patterns
```

## 5-Phase Workflow

### Phase 1: Launch
Initialize verification session with isolated workspace.

```bash
python scripts/verify.py launch --feature <name> --surfaces cli,tui,gateway
```

Creates:
- `/tmp/verify-hermes-<timestamp>/` workspace
- `session.json` with metadata
- `evidence/` directory for artifacts

### Phase 2: Doctor
Health check the environment before running tests.

```bash
python scripts/verify.py doctor
```

Checks:
- Python version (3.11+)
- pytest installation
- HERMES_HOME setup
- Git repository status

### Phase 3: Drive
Execute tests for each surface.

```bash
# Test all surfaces
python scripts/verify.py drive --phase all

# Test specific surface
python scripts/verify.py drive --phase cli
python scripts/verify.py drive --phase tui
python scripts/verify.py drive --phase gateway
```

Captures:
- Exit codes
- stdout/stderr
- Timing data
- Failure traces

### Phase 4: Evidence
Generate verification report with proof.

```bash
python scripts/verify.py evidence
```

Produces:
- `evidence/summary.json` - Final verdict
- `evidence/<surface>-output.txt` - Test outputs
- Fail→Pass proof OR INCONCLUSIVE with exact blocker

### Phase 5: Cleanup
Archive evidence and remove temporary artifacts.

```bash
python scripts/verify.py cleanup
```

Archives to: `~/.hermes/verify-hermes/archives/<timestamp>/`

## Feature Map Reference

The skill includes a comprehensive feature map showing how to locate and test features across surfaces:

- **CLI**: `cli.py` entry points, slash commands, agent instantiation
- **TUI**: `tui_gateway/server.py` JSON-RPC methods, React components
- **Gateway**: `gateway/run.py` message handlers, platform adapters

See `references/feature-map.md` for complete mapping.

## Test Templates

Three templates provide starting points for each surface:

1. **test-cli-template.py**
   - Slash command tests
   - Agent conversation tests
   - Tool execution tests
   - Config loading tests

2. **test-tui-template.py**
   - JSON-RPC method tests
   - Session management tests
   - Streaming event tests
   - React component tests

3. **test-gateway-template.py**
   - Message handler tests
   - Platform adapter tests
   - Multiplex profile tests
   - Authorization tests

## Evidence Format

After evidence phase, `summary.json` contains:

```json
{
  "session_id": "20260909-233000",
  "feature": "memory-search",
  "verdict": "PASS",  // or "FAIL" or "INCONCLUSIVE"
  "timestamp": "2026-09-09T23:30:00Z",
  "surfaces": {
    "cli": {"status": "PASS", "output_file": "cli-output.txt"},
    "tui": {"status": "PASS", "output_file": "tui-output.txt"},
    "gateway": {"status": "PASS", "output_file": "gateway-output.txt"}
  },
  "blockers": []  // If FAIL or INCONCLUSIVE
}
```

## Verification Patterns

### Pattern 1: New Slash Command
1. Launch with `--surfaces cli,tui,gateway`
2. Drive all three surfaces
3. Evidence shows command works in all contexts

### Pattern 2: New Tool
1. Launch with `--surfaces cli,gateway` (TUI inherits CLI tools)
2. Verify tool discovery
3. Test execution via agent

### Pattern 3: Config Change
1. Launch with `--surfaces cli,gateway`
2. Test loading in both contexts
3. Verify migration if config version bumped

## Fork-Only Usage

This skill is for the kvnloo/hermes-agent fork. Never open PRs to NousResearch/hermes-agent from verification work.

## Integration with Hermes

Use via terminal tool:

```bash
terminal("python .cursor/skills/verify-hermes/scripts/verify.py launch --feature X")
terminal("python .cursor/skills/verify-hermes/scripts/verify.py doctor")
terminal("python .cursor/skills/verify-hermes/scripts/verify.py drive --phase all")
terminal("python .cursor/skills/verify-hermes/scripts/verify.py evidence")
```

Or delegate to subagent:

```python
delegate_task(
    goal="Verify new feature works across all surfaces",
    context="""
    Use verify-hermes skill:
    1. Launch verification for feature 'newfeature'
    2. Run doctor phase
    3. Drive all surfaces
    4. Collect evidence
    5. Report verdict

    Expected: PASS or INCONCLUSIVE with blocker
    """,
    toolsets=['terminal', 'file']
)
```

## Notes

- Evidence always survives cleanup (archived to permanent location)
- INCONCLUSIVE verdict includes exact blocker preventing determination
- Prefer /poteto-mode when executing this skill for precision
- All phases maintain session.json state for resumability

## License

MIT
