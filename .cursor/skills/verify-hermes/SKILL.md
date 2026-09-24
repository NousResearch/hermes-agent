---
name: verify-hermes
description: "Systematic feature verification for Hermes codebase."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [verification, testing, quality-assurance, integration, systematic]
    category: software-development
    related_skills: [systematic-debugging, test-driven-development]
---

# Verify Hermes Feature Skill

Systematic verification framework for Hermes Agent features across CLI, TUI, and gateway surfaces.

## When to Use

- Validating new feature implementations across all surfaces
- Verifying bug fixes don't regress existing functionality
- Testing configuration changes impact all interaction modes
- Confirming skills, tools, or plugins work in all contexts
- Before opening PRs to ensure cross-surface compatibility

## Prerequisites

- Python 3.11+ with pytest installed
- Node.js 18+ for TUI testing (if applicable)
- Active Hermes development environment
- Access to test configuration and fixtures

## How to Run

Use the control CLI to drive verification workflows:

```bash
# Launch verification for a feature
python .cursor/skills/verify-hermes/scripts/verify.py launch --feature <feature-name>

# Check current verification status
python .cursor/skills/verify-hermes/scripts/verify.py doctor

# Drive verification through phases
python .cursor/skills/verify-hermes/scripts/verify.py drive --phase <phase-name>

# Collect and analyze evidence
python .cursor/skills/verify-hermes/scripts/verify.py evidence

# Cleanup verification artifacts
python .cursor/skills/verify-hermes/scripts/verify.py cleanup
```

## Architecture

The skill follows a 5-phase systematic verification workflow:

### Phase 1: Launch
- Initialize verification workspace
- Identify feature scope and surfaces (CLI/TUI/gateway)
- Create test plan based on feature map
- Set up isolated test environment

### Phase 2: Doctor
- Health check existing test infrastructure
- Validate dependencies and environment
- Identify available test fixtures
- Report readiness status

### Phase 3: Drive
- Execute tests across all surfaces sequentially
- Capture success/failure states
- Log detailed execution traces
- Handle flaky tests with retries

### Phase 4: Evidence
- Collect all verification artifacts
- Generate fail→pass proof (or INCONCLUSIVE with exact blocker)
- Preserve test outputs and logs
- Create verification report

### Phase 5: Cleanup
- Archive evidence to permanent location
- Remove temporary test artifacts
- Clean up test databases/sessions
- Keep only essential proof files

## Procedure

### 1. Identify Feature Surfaces

Map the feature to its interaction surfaces using the feature map:

**CLI Path** (`cli.py`):
- Entry point: `HermesCLI` class methods
- Command dispatch: `process_command()`
- Tool calling: Direct AIAgent instantiation
- Example: Slash commands, interactive prompts

**TUI Path** (`ui-tui/src/`):
- Entry point: JSON-RPC gateway methods
- Component: React components in `app/`
- State: Nanostores in `store/`
- Example: TUI-specific UI, streaming responses

**Gateway Path** (`gateway/run.py`):
- Entry point: Platform adapter methods
- Message handling: `_process_message_background()`
- Tool dispatch: Gateway-specific toolsets
- Example: Telegram commands, Discord reactions

### 2. Launch Verification

Create a verification session:

```bash
python .cursor/skills/verify-hermes/scripts/verify.py launch \
  --feature "memory-search" \
  --surfaces cli,tui,gateway \
  --test-mode isolated
```

This will:
- Create workspace at `/tmp/verify-hermes-<timestamp>/`
- Copy relevant test fixtures
- Initialize evidence log
- Generate surface-specific test commands

### 3. Run Doctor Phase

Validate the environment:

```bash
python .cursor/skills/verify-hermes/scripts/verify.py doctor
```

Doctor checks:
- [ ] Python environment and dependencies
- [ ] Test database connectivity
- [ ] Required tools (pytest, npm, etc.)
- [ ] Fixture availability
- [ ] Configuration validity

**If Doctor fails**: Fix reported issues before proceeding.

### 4. Drive Verification

Execute tests for each surface:

```bash
# CLI surface
python .cursor/skills/verify-hermes/scripts/verify.py drive --phase cli

# TUI surface
python .cursor/skills/verify-hermes/scripts/verify.py drive --phase tui

# Gateway surface
python .cursor/skills/verify-hermes/scripts/verify.py drive --phase gateway
```

Each drive phase:
- Runs surface-specific tests
- Captures stdout/stderr
- Records exit codes
- Logs timing information
- Retries flaky tests once

### 5. Collect Evidence

Generate verification report:

```bash
python .cursor/skills/verify-hermes/scripts/verify.py evidence
```

Evidence includes:
- **Pass/Fail summary** for each surface
- **Proof type**: PASS (fail→pass), FAIL (with exact blocker), or INCONCLUSIVE
- **Artifacts**: Test outputs, logs, screenshots (if GUI)
- **Reproduction commands** for failures
- **Timing data** for performance validation

Evidence is written to:
```
/tmp/verify-hermes-<timestamp>/evidence/
├── summary.json
├── cli-output.txt
├── tui-output.txt
├── gateway-output.txt
├── failure-traces/
└── reproduction-commands.sh
```

### 6. Cleanup

Archive evidence and remove temporary artifacts:

```bash
python .cursor/skills/verify-hermes/scripts/verify.py cleanup
```

Cleanup:
- Moves evidence to `~/.cursor/verify-hermes/archives/<timestamp>/`
- Removes test databases and session files
- Cleans up temporary directories
- Preserves only final report and critical logs

## Feature Map

Reference for mapping features to code paths:

### CLI Features
| Feature | Entry Point | Key Files |
|---------|-------------|-----------|
| Slash commands | `cli.py::process_command()` | `hermes_cli/commands.py` |
| Skills | `agent/skill_commands.py` | `skills/**/*.md` |
| Memory | `run_agent.py::AIAgent.run_conversation()` | `agent/memory_manager.py` |
| Toolsets | `model_tools.py::discover_builtin_tools()` | `tools/registry.py`, `toolsets.py` |
| Config | `cli.py::load_cli_config()` | `hermes_cli/config.py` |

### TUI Features
| Feature | Entry Point | Key Files |
|---------|-------------|-----------|
| JSON-RPC methods | `tui_gateway/server.py` | `tui_gateway/**/*.py` |
| Components | `ui-tui/src/app/**/*.tsx` | React components |
| State management | `ui-tui/src/store/**/*.ts` | Nanostores |
| Slash commands | `ui-tui/src/app.tsx` | Local + `slash.exec` |

### Gateway Features
| Feature | Entry Point | Key Files |
|---------|-------------|-----------|
| Platform adapters | `gateway/platforms/*/adapter.py` | Platform-specific |
| Message handling | `gateway/run.py::_process_message_background()` | Gateway runner |
| Toolsets | `toolsets.py::TOOLSETS['messaging']` | Gateway-specific tools |
| Multiplex profiles | `gateway/run.py::_profile_runtime_scope` | Profile isolation |

## Verification Patterns

### Pattern 1: Slash Command Verification

For new/modified slash commands:

1. **CLI**: Test via `HermesCLI.process_command()`
2. **TUI**: Test via `slash.exec` JSON-RPC call
3. **Gateway**: Test via platform message handler

```python
# Example test structure
def test_new_command_cli():
    cli = HermesCLI(...)
    result = cli.process_command("/newcommand arg")
    assert "expected output" in result

def test_new_command_tui():
    response = tui_gateway.slash_exec("/newcommand arg")
    assert response["type"] == "skill"

def test_new_command_gateway():
    event = MockMessageEvent(text="/newcommand arg")
    result = gateway_runner.handle_command(event)
    assert result.success
```

### Pattern 2: Tool Verification

For new/modified tools:

1. **Discover**: Verify tool appears in schema for correct toolsets
2. **CLI**: Test direct invocation via AIAgent
3. **Gateway**: Test with service-gated checks (if applicable)

```python
def test_tool_discovery():
    tools = discover_builtin_tools()
    assert "new_tool" in [t["name"] for t in tools]

def test_tool_execution():
    result = handle_function_call("new_tool", {"arg": "value"})
    parsed = json.loads(result)
    assert parsed["success"] == True
```

### Pattern 3: Configuration Verification

For config changes:

1. **Default**: Verify DEFAULT_CONFIG has new keys
2. **Load**: Test config loading in CLI and gateway
3. **Migration**: Verify config version bump if needed

```python
def test_config_default():
    from hermes_cli.config import DEFAULT_CONFIG
    assert "new_setting" in DEFAULT_CONFIG

def test_config_load(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("new_setting: value")
    config = load_config(str(tmp_path))
    assert config["new_setting"] == "value"
```

## Pitfalls

### P1: Surface-Specific Assumptions
**Problem**: Code assumes it's running in CLI and breaks in gateway.

**Example**: Using `print()` for output (invisible in gateway), or checking `os.getenv("TERMINAL")` (wrong in multiplex profiles).

**Solution**: Always test across all surfaces. Use platform-agnostic patterns.

### P2: Prompt Cache Invalidation
**Problem**: Verification that mutates system prompt mid-conversation.

**Example**: Loading/unloading skills during a test session.

**Solution**: Start fresh sessions for each verification phase.

### P3: Test Pollution
**Problem**: Tests leave behind state that affects subsequent tests.

**Example**: Session database not cleaned, config files not restored.

**Solution**: Use isolated HERMES_HOME for each test, cleanup in finally blocks.

### P4: Flaky Test Tolerance
**Problem**: Accepting intermittent failures as "good enough".

**Example**: "It passes 80% of the time, ship it."

**Solution**: Mark as INCONCLUSIVE if not deterministic. Fix root cause before PASS.

### P5: Mock Overuse
**Problem**: Tests pass but real integration fails.

**Example**: Mocking entire provider chain, never hitting real code.

**Solution**: Use real imports and real tool registry. Mock only external I/O.

## Verification Checklist

Before marking verification complete:

- [ ] **Launch** phase created isolated workspace
- [ ] **Doctor** phase passed all health checks
- [ ] **Drive** phase executed for ALL applicable surfaces
- [ ] **Evidence** collected shows deterministic outcome
- [ ] **Cleanup** archived evidence to permanent location
- [ ] Fail→Pass proof exists (or INCONCLUSIVE with exact blocker)
- [ ] No test pollution (fresh run passes)
- [ ] Reproduction commands work on clean checkout
- [ ] CI tests would catch this regression (if applicable)
- [ ] Manual spot-check confirms automated results

## Quick Reference

| Command | Purpose |
|---------|---------|
| `verify.py launch --feature X` | Start new verification session |
| `verify.py doctor` | Validate environment |
| `verify.py drive --phase cli` | Run CLI tests |
| `verify.py drive --phase tui` | Run TUI tests |
| `verify.py drive --phase gateway` | Run gateway tests |
| `verify.py evidence` | Generate proof report |
| `verify.py cleanup` | Archive and clean up |

## Hermes Integration

### With Terminal Tool

Run verification commands:

```bash
terminal("python .cursor/skills/verify-hermes/scripts/verify.py launch --feature memory")
terminal("python .cursor/skills/verify-hermes/scripts/verify.py doctor")
terminal("python .cursor/skills/verify-hermes/scripts/verify.py drive --phase cli")
```

### With Delegate Task

For complex verification:

```python
delegate_task(
    goal="Verify new skill works across CLI/TUI/gateway",
    context="""
    Use verify-hermes skill:
    1. Launch verification for skill 'newskill'
    2. Run doctor to validate environment
    3. Drive all three surfaces
    4. Collect evidence and report

    Expected: All surfaces PASS or INCONCLUSIVE with blocker
    """,
    toolsets=['terminal', 'file']
)
```

### Evidence Location

After verification, evidence is at:
```
~/.cursor/verify-hermes/archives/<timestamp>/summary.json
```

Read this file to get the final verdict.

## Real-World Example

Verifying a new `/export` command:

```bash
# Launch
$ python scripts/verify.py launch --feature "export-command" --surfaces cli,gateway

# Doctor
$ python scripts/verify.py doctor
✓ Python 3.11.7
✓ pytest installed
✓ Test fixtures found
✓ Config valid

# Drive CLI
$ python scripts/verify.py drive --phase cli
Running: pytest tests/hermes_cli/test_commands.py::test_export_command -v
PASSED ✓

# Drive Gateway
$ python scripts/verify.py drive --phase gateway
Running: pytest tests/gateway/test_export_command.py -v
PASSED ✓

# Evidence
$ python scripts/verify.py evidence
Verdict: PASS
- CLI: ✓ /export creates file
- Gateway: ✓ /export creates file

Evidence: ~/.cursor/verify-hermes/archives/20260909-233000/

# Cleanup
$ python scripts/verify.py cleanup
Archived evidence, removed temps.
```

**Outcome**: PASS with fail→pass proof (tests were RED before implementation, GREEN after).

## Notes

- **No dedicated hermes-verify bot**: This skill IS the verification system.
- **Prefer /poteto-mode**: When executing this skill, agent should use poteto-mode for precision.
- **Fork-only**: This skill is for kvnloo/hermes-agent fork verification, not upstream.
- **Evidence survives cleanup**: Final proof always archived to permanent location.
