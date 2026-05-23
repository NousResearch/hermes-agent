# CLAUDE.md — Agent & Developer Workspaces Playbook

Workspace directives, conventions, and operational procedures for `hermes-agent`.

---

## ─── DEVELOPMENT QUICKSTART ───

### Build & Virtual Environment
The repository utilizes a standard Python virtual environment inside the `venv` directory.
```bash
# Activation
source venv/bin/activate

# Installation & Dependencies
pip install -e .
```

### Automation & Shortcuts
We enforce workflow operations via standard `Makefile` recipes:
```bash
make lint         # Run fast static code analysis (ruff check)
make format       # Automatically clean up formatting (ruff format)
make review       # Execute fast local CodeRabbit review on change deltas
make test         # Run complete test suite (via scripts/run_tests.sh)
make test-smoke   # Run fast verification suite (specific test file)
make debug-smoke  # Run fast smoke test with immediate assertions + trace compression
make debug-test   # Run full test suite with immediate assertions + trace compression
make rgt-log      # View Regent VCS agent transaction and step logs
make rgt-status   # Display Regent VCS workspace audit states
```

---

## ─── RUNNING TESTS ───

Standard development REQUIRES running tests before certifying any code changes.

### Canonical Test Runner
Do NOT execute bare `pytest` on the root. Always use our specialized isolated test runner script:
```bash
# Run the entire pytest suite in parallel with file-isolation
scripts/run_tests.sh

# Target specific directories or files
scripts/run_tests.sh tests/agent/
scripts/run_tests.sh tests/test_smoke.py

# Run with custom pytest flags (passed after '--')
scripts/run_tests.sh tests/test_smoke.py -- -vv --tb=short
```

---

## ─── CODE STYLE & QUALITY GATES ───

To keep coding agents and humans synchronized without code churn or regression:

### Code Formatting & Quality
- **Code Style:** Black conformant style enforced strictly via **Ruff**.
- **Linter Rule Set:** `ruff` configuration is defined under the `[tool.ruff]` section of `pyproject.toml`.
- **Target Formatting:** Run `make format` to run Ruff's automatic fixer.

### Typing & Type Checks
- Strong typing preference for all newly introduced core logic in `run_agent.py`, `model_tools.py`, or gateway platforms.
- Avoid introducing blind `Any` where actual types (e.g., `str`, `list`, dataclasses, pydantic models) are known.

---

## ─── WORKFLOW & PR POLICIES ───

### 1. Plan-Before-Code Gating
- For any change larger than 5 lines, write or output an inline Implementation Plan before editing files.
- Walk Felipe (the human) through the plan first, confirming scope and potential trade-offs.

### 2. Guardrails (Approval Boundaries)
- **Production Quarantine:** Do not make, copy, or link binaries under `/Applications/` or target system folders unless Felipe explicitly inputs `Deploy to production`. Keep all activities within `/Users/felipelamartine/Documents/Oryn.ai/` or active sandboxes.
- **Gateway Control:** Never attempt to restart, stop, or kill the gateway process (running on port 8642) using programmatic execution or programs. Let Felipe manage gateway lifecycles.

### 3. Repository Git Disciplines
- **Branches:** Work inside clean developmental namespaces:
  - `dev/felipe/feature-<name>` for humans
  - `agent/dev/feature-<name>` for coding agents
- **Commits:** Conventional commits only (`feat(scope): ...`, `fix(scope): ...`). When writing as a worker agent, prefix/appended metadata identifying your agent (e.g., `feat(agent): ...`).
- **Isolation:** No direct commits to the upstream `main` branch. Build a Pull Request branch, push to staging, and request a code review.

### 4. CodeRabbit AI Review Gates
- **Mandatory Self-Review:** Before requesting a human review or filing a final PR, execute `make review` locally.
- **Handling Findings:** Coding agents MUST parse the review findings. Any high or medium severity warning (security flaws, performance regressions, logic errors) must be fixed immediately.
- **Verification Gating:** Repeat `make review` until the code delta is clean or any remaining findings are manually approved by Dev (as false positives).

### 5. High-Resolution Auditing (Regent VCS)
- **Trace Line Provenances:** During advanced systematic debugging (or when tracebacks point to a recently edited line inside first-party files), query Regent VCS to see line provenances:
  ```bash
  make rgt-blame FILE=path/to/problematic_file.py
  ```
- **Contextualize Decisions:** Use the outputted session and conversational hashes to understand the underlying context and decisions that led to the buggy implementation, avoiding breaking implicit assumptions.
