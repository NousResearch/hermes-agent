# Software Development Skills

Professional software development workflows for Hermes Agent, covering planning, debugging, code review, and test-driven development.

## Overview

This category contains skills that help you build software more effectively — from planning and architecture to debugging and code quality. Whether you're starting a new feature, hunting down a bug, or preparing code for review, these skills provide structured workflows and best practices.

## Available Skills

### Planning & Architecture

#### **plan**
Write markdown implementation plans instead of executing code immediately.

**Use when:** You want to think through an approach before implementing, document a strategy for review, or break down a complex task into steps.

**Key features:**
- Creates timestamped plans in `.hermes/plans/`
- Includes goals, approach, file paths, and validation steps
- Read-only inspection allowed; no code execution

---

#### **writing-plans**
Create detailed, actionable implementation plans with bite-sized tasks.

**Use when:** Planning complex features that need structured breakdown with concrete file paths and verification steps.

**Key features:**
- Bite-sized, concrete task lists
- Explicit file paths and code locations
- Built-in verification and testing strategy

---

### Debugging

#### **systematic-debugging**
Four-phase root cause analysis: understand the bug before attempting fixes.

**Use when:** Facing complex bugs that need methodical investigation rather than trial-and-error.

**Key features:**
- Structured debugging workflow (understand → hypothesize → test → fix)
- Prevents premature fixes
- Documents reasoning for future reference

---

#### **python-debugpy**
Debug Python code using pdb REPL and debugpy remote debugging (DAP protocol).

**Use when:** You need to step through Python code, inspect variables, or debug remotely.

**Key features:**
- Interactive pdb REPL for quick debugging
- Remote debugging via Debug Adapter Protocol
- Breakpoint and variable inspection support

---

#### **node-inspect-debugger**
Debug Node.js applications via --inspect and Chrome DevTools Protocol CLI.

**Use when:** Debugging Node.js/JavaScript code with breakpoints and inspection.

**Key features:**
- Chrome DevTools Protocol integration
- Command-line debugging interface
- Works with Node.js --inspect flag

---

#### **debugging-hermes-tui-commands**
Debug Hermes TUI slash commands, Python runtime, gateway, and Ink UI components.

**Use when:** Contributing to Hermes Agent itself or debugging skill execution issues.

**Key features:**
- Hermes-specific debugging workflows
- TUI command inspection
- Gateway and runtime diagnostics

---

### Code Quality & Review

#### **requesting-code-review**
Pre-commit code review automation: security scanning, quality gates, and auto-fixes.

**Use when:** Preparing code for pull request or ensuring quality before committing.

**Key features:**
- Automated security and quality scanning
- Suggests improvements and auto-fixes
- Enforces quality gates

---

#### **test-driven-development**
Enforce RED-GREEN-REFACTOR workflow: write tests before implementation.

**Use when:** Building new features with TDD methodology to ensure testability and correctness.

**Key features:**
- Enforces test-first approach
- RED (failing test) → GREEN (minimal pass) → REFACTOR cycle
- Prevents implementing before testing

---

### Development Workflows

#### **subagent-driven-development**
Execute implementation plans via delegate_task subagents with two-stage review.

**Use when:** Breaking complex work into parallel, isolated tasks executed by subagents.

**Key features:**
- Spawns isolated subagent workspaces
- Two-stage review (plan review + implementation review)
- Parallel task execution

---

#### **hermes-agent-skill-authoring**
Author skills in-repo with proper SKILL.md frontmatter, validation, and structure.

**Use when:** Creating new skills for Hermes Agent or documenting workflows.

**Key features:**
- SKILL.md template and structure guide
- Frontmatter validator
- Best practices for skill authoring

---

## Quick Start

### Example: Plan-First Development

```bash
# 1. Plan the feature
/plan "Add user authentication to the API"

# 2. Review the plan in .hermes/plans/

# 3. Implement using TDD
/tdd "Implement user authentication based on plan"
```

### Example: Systematic Bug Investigation

```bash
# 1. Start systematic debugging workflow
/systematic-debugging "API returns 500 on user creation"

# 2. Follow the four phases:
#    - Understand: Reproduce and gather context
#    - Hypothesize: Form theories about root cause
#    - Test: Verify hypotheses
#    - Fix: Implement the solution
```

### Example: Code Review Preparation

```bash
# Run pre-commit quality checks
/requesting-code-review "Review auth feature changes"

# Fix any issues, then commit
```

## Skill Combinations

**Planning → TDD → Review:**
1. Use plan or writing-plans to design the feature
2. Implement with test-driven-development
3. Run requesting-code-review before committing

**Complex Debugging:**
1. Start with systematic-debugging for methodical investigation
2. Use python-debugpy or node-inspect-debugger for deep inspection
3. Document findings in a plan for future reference

**Large Feature Development:**
1. Create high-level plan with writing-plans
2. Break into tasks with subagent-driven-development
3. Each subagent follows TDD workflow
4. Final review before merging

## Contributing

Found a bug in a skill? Have an idea for improvement?

1. Open an issue describing the enhancement
2. Fork the repository
3. Make your changes to the relevant SKILL.md
4. Submit a pull request

## Related Categories

- **github/** - GitHub integration and workflows
- **productivity/** - General productivity tools
- **research/** - Research and analysis tools
- **mlops/** - Machine learning operations

---

**Questions?** Check the [Hermes Agent documentation](https://hermes-agent.nousresearch.com/docs/) or ask in the [Discord community](https://discord.gg/nousresearch).
