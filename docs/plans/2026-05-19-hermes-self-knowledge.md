# Hermes Self-Knowledge Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a living, repo-grounded Hermes self-knowledge document that refreshes from code, can be drift-checked, and can feed a slim summary into the system prompt without manual upkeep.

**Architecture:** Add a small `hermes_cli.self_knowledge` package that owns markdown AUTO-block parsing, generators, rendering, drift checks, and CLI handlers. The repo self-knowledge doc lives at `context/self/hermes-agent.md`; generated sections read from Hermes’ existing runtime sources of truth: `tools/registry.py`, `toolsets.py`, `hermes_cli/commands.py`, `gateway/platforms/`, skills directories, plugins, and voice/STT/TTS config surfaces. Prompt injection uses a slim cached summary from `agent/prompt_builder.py` so prompt cache churn stays controlled.

**Tech Stack:** Python, argparse, pytest, markdown files, GitHub Actions. No network calls in rendering, hooks, or checks.

---

## Ground Truth Discovered

- Tool registry: `tools/registry.py`, especially `discover_builtin_tools()`, `registry`, `ToolEntry`, and snapshot helpers.
- Toolsets: `toolsets.py`, especially `_HERMES_CORE_TOOLS` and `TOOLSETS`.
- Slash commands: `hermes_cli/commands.py`, especially `COMMAND_REGISTRY`.
- CLI entry: `hermes_cli/main.py`; add a top-level `self-knowledge` subcommand alongside existing subcommands.
- Prompt assembly: `agent/prompt_builder.py`; `AIAgent._build_system_prompt()` calls this module.
- Gateway/voice surface: `gateway/platforms/`, `gateway/run.py`, `tools/transcription_tools.py`, `tools/tts_tool.py`, `tools/voice_mode.py`.
- CI exists under `.github/workflows/`; use a soft check at first.
- Current working tree already has unrelated local modifications in `gateway/platforms/discord.py`, `tests/tools/test_discord_tool.py`, and `tools/discord_tool.py`. Do not overwrite them.

---

## Proposed Self-Knowledge Document Layout

Create `context/self/hermes-agent.md`:

1. Identity — hand-written
2. Operating principles — hand-written
3. Capabilities at a glance — AUTO from `tools/registry.py`
4. Toolsets — AUTO from `toolsets.py`
5. Slash commands — AUTO from `hermes_cli/commands.py`
6. Gateway platforms — AUTO from `gateway/platforms/`
7. Voice / STT / TTS loop — AUTO from voice modules and config schema, not secrets
8. Skills and agent profiles — AUTO from repo skills plus profile skill metadata
9. Plugins and integrations — AUTO from `plugins/`, config schema, and env var names only, never values
10. Recent activity — AUTO from local `git log --since=14.days`
11. Open questions / unknowns — hand-written
12. Pointers — hand-written links to `AGENTS.md`, website developer docs, and relevant runbooks

All AUTO blocks use:

```markdown
<!-- AUTO-START: capabilities -->
_this section is generated; do not edit by hand_
<!-- AUTO-END: capabilities -->
```

---

## Phase 1 — Doc Scaffold + AUTO Block Parser

### Task 1: Add the initial self-knowledge markdown scaffold

**Objective:** Create the doc with stable hand-written sections and placeholder AUTO blocks.

**Files:**
- Create: `context/self/hermes-agent.md`

**Step 1: Write the scaffold**

Include the layout above. Keep hand-written sections short. Each generated section must have a unique AUTO block name.

**Step 2: Verify markers are paired**

Run:

```bash
python - <<'PY'
from pathlib import Path
p = Path('context/self/hermes-agent.md')
text = p.read_text()
starts = text.count('<!-- AUTO-START:')
ends = text.count('<!-- AUTO-END:')
print(starts, ends)
assert starts == ends and starts >= 8
PY
```

Expected: prints matching counts and exits 0.

**Step 3: Commit**

```bash
git add context/self/hermes-agent.md
git commit -m "docs: add Hermes self-knowledge scaffold"
```

### Task 2: Add parser module and fixtures

**Objective:** Parse and replace AUTO blocks without touching hand-written content.

**Files:**
- Create: `hermes_cli/self_knowledge/__init__.py`
- Create: `hermes_cli/self_knowledge/parser.py`
- Create: `tests/self_knowledge/test_parser.py`

**Step 1: Write failing parser tests**

Test cases:
- Parse block names and bodies.
- Round-trip serialization is exact for LF files.
- Round-trip serialization is exact for CRLF files.
- Replacing one block leaves surrounding handwritten text byte-for-byte unchanged.
- Duplicate block names raise `ValueError`.
- Missing end marker raises `ValueError`.

**Step 2: Implement parser**

Recommended API:

```python
@dataclass(frozen=True)
class AutoBlock:
    name: str
    start: int
    end: int
    body_start: int
    body_end: int
    body: str

AUTO_START_RE = re.compile(r"<!--\s*AUTO-START:\s*([a-z0-9_-]+)\s*-->")
AUTO_END_RE = re.compile(r"<!--\s*AUTO-END:\s*([a-z0-9_-]+)\s*-->")

def parse_auto_blocks(text: str) -> dict[str, AutoBlock]: ...
def replace_auto_blocks(text: str, replacements: Mapping[str, str]) -> str: ...
```

Implementation detail: preserve the file's existing newline convention by replacing only the body span between markers.

**Step 3: Run tests**

```bash
pytest tests/self_knowledge/test_parser.py -q
```

Expected: all parser tests pass.

**Step 4: Commit**

```bash
git add hermes_cli/self_knowledge tests/self_knowledge/test_parser.py
git commit -m "feat: add self-knowledge auto-block parser"
```

---

## Phase 2 — Introspecting Generators + Renderer

### Task 3: Add renderer orchestration

**Objective:** Wire block names to generator functions and support dry-run vs write.

**Files:**
- Create: `hermes_cli/self_knowledge/renderer.py`
- Create: `tests/self_knowledge/test_renderer.py`

**Step 1: Write tests**

Test that:
- Unknown AUTO block names render a clear unavailable placeholder, not a crash.
- Known block replacements preserve hand-written sections.
- Renderer returns the rendered markdown string and can optionally write to disk.

**Step 2: Implement renderer**

Recommended API:

```python
DOC_PATH = Path('context/self/hermes-agent.md')

Generator = Callable[[], str]
GENERATORS: dict[str, Generator] = {}

def render_self_knowledge(doc_path: Path = DOC_PATH) -> str: ...
def refresh_self_knowledge(doc_path: Path = DOC_PATH) -> bool: ...  # True if changed
```

Use a placeholder like:

```markdown
_unavailable: no generator registered for this AUTO block_
```

**Step 3: Run tests**

```bash
pytest tests/self_knowledge/test_renderer.py -q
```

Expected: pass.

**Step 4: Commit**

```bash
git add hermes_cli/self_knowledge/renderer.py tests/self_knowledge/test_renderer.py
git commit -m "feat: add self-knowledge renderer"
```

### Task 4: Add capabilities generator from live registry

**Objective:** Generate tool capability rows from the same registry runtime uses.

**Files:**
- Create: `hermes_cli/self_knowledge/generators.py`
- Modify: `hermes_cli/self_knowledge/renderer.py`
- Create/Modify: `tests/self_knowledge/test_generators.py`

**Step 1: Write fixture test**

Use lightweight fake `ToolEntry`-like objects or monkeypatch `tools.registry.registry._snapshot_entries()`.

Expected markdown columns:

```markdown
| Tool | Toolset | Description |
|---|---|---|
| web_search | web | Search the web... |
```

**Step 2: Implement generator**

Import discovery safely:

```python
def generate_capabilities() -> str:
    try:
        from tools.registry import discover_builtin_tools, registry
        discover_builtin_tools()
        entries = sorted(registry._snapshot_entries(), key=lambda e: (e.toolset, e.name))
    except Exception as exc:
        return f"_unavailable: could not load tool registry ({type(exc).__name__})_"
    ...
```

Do not call each tool's external check function here; this is inventory, not availability probing.

**Step 3: Register generator for `capabilities`**

Update `GENERATORS` in `renderer.py`.

**Step 4: Verify**

```bash
pytest tests/self_knowledge/test_generators.py::test_generate_capabilities -q
python -m hermes_cli.self_knowledge.dev_render_preview
```

If no dev preview module exists yet, use a small inline Python import for now.

**Step 5: Commit**

```bash
git add hermes_cli/self_knowledge tests/self_knowledge/test_generators.py
git commit -m "feat: generate self-knowledge capabilities"
```

### Task 5: Add toolsets and slash-command generators

**Objective:** Summarize named toolsets and commands from existing single sources of truth.

**Files:**
- Modify: `hermes_cli/self_knowledge/generators.py`
- Modify: `hermes_cli/self_knowledge/renderer.py`
- Modify: `tests/self_knowledge/test_generators.py`

**Step 1: Add tests**

- `generate_toolsets()` imports `TOOLSETS` and `_HERMES_CORE_TOOLS` from `toolsets.py`.
- `generate_slash_commands()` imports `COMMAND_REGISTRY` from `hermes_cli.commands`.
- Assert markdown includes name, description/category, and count/availability scope.

**Step 2: Implement generators**

Keep output compact:

```markdown
| Toolset | Description | Tools | Includes |
|---|---|---:|---|
```

```markdown
| Command | Category | Scope | Description |
|---|---|---|---|
```

Scope logic: `cli_only`, `gateway_only`, otherwise `cli+gateway`.

**Step 3: Register blocks**

Block names: `toolsets`, `slash_commands`.

**Step 4: Run tests and commit**

```bash
pytest tests/self_knowledge/test_generators.py -q
git add hermes_cli/self_knowledge tests/self_knowledge/test_generators.py
git commit -m "feat: generate toolset and command self-knowledge"
```

### Task 6: Add gateway, voice, skills, plugins, and recent activity generators

**Objective:** Cover the Hermes-specific surfaces not in the generic prompt.

**Files:**
- Modify: `hermes_cli/self_knowledge/generators.py`
- Modify: `hermes_cli/self_knowledge/renderer.py`
- Modify: `tests/self_knowledge/test_generators.py`

**Step 1: Gateway platforms**

Read `gateway/platforms/*.py`; list platform module names. Prefer static file scan over importing adapters to avoid optional dependency side effects.

**Step 2: Voice loop**

Static summary from known files:
- `tools/transcription_tools.py`
- `tools/tts_tool.py`
- `tools/voice_mode.py`
- `gateway/platforms/discord.py`

Do not read or print secret values. If reading config for provider names, use keys only and redact values by design.

**Step 3: Skills and profiles**

Use `agent.skill_utils` helpers where safe; otherwise scan `skills/**/SKILL.md` and `~/.hermes/skills/**/SKILL.md` frontmatter using existing parsing helpers. Summarize counts by category and list profile skills separately.

**Step 4: Plugins and integrations**

Scan `plugins/*` directory names and config/env key names only. Never include API key values.

**Step 5: Recent activity**

Use local git only:

```python
subprocess.run(['git', 'log', '--since=14 days ago', '--pretty=format:%h %ad %s', '--date=short'], ...)
```

If git is unavailable, return `_unavailable: git log failed_`.

**Step 6: Run and commit**

```bash
pytest tests/self_knowledge/test_generators.py -q
git add hermes_cli/self_knowledge tests/self_knowledge/test_generators.py
git commit -m "feat: generate Hermes platform self-knowledge"
```

### Task 7: Refresh the doc end-to-end

**Objective:** Render all AUTO blocks into `context/self/hermes-agent.md`.

**Files:**
- Modify: `context/self/hermes-agent.md`

**Step 1: Run renderer**

Use the renderer import until CLI lands:

```bash
python - <<'PY'
from hermes_cli.self_knowledge.renderer import refresh_self_knowledge
changed = refresh_self_knowledge()
print('changed=', changed)
PY
```

**Step 2: Verify no stale diff after second render**

```bash
python - <<'PY'
from hermes_cli.self_knowledge.renderer import refresh_self_knowledge
assert refresh_self_knowledge() is False
PY
```

**Step 3: Commit**

```bash
git add context/self/hermes-agent.md
git commit -m "docs: refresh Hermes self-knowledge inventory"
```

---

## Phase 3 — Drift Checker + CLI

### Task 8: Add drift checker core

**Objective:** Check hand-written sections for stale file/path/name references.

**Files:**
- Create: `hermes_cli/self_knowledge/drift.py`
- Create: `context/self/.hermes-agent-allowlist.txt`
- Create: `tests/self_knowledge/test_drift.py`

**Step 1: Write tests**

Test `DriftFinding` output for:
- Missing file path reference in hand-written text.
- Existing file path passes.
- References inside AUTO blocks are ignored.
- Allowlisted reference is ignored.

**Step 2: Implement dataclass**

```python
@dataclass(frozen=True)
class DriftFinding:
    kind: str
    reference: str
    location_in_doc: str
    reason: str
```

**Step 3: Implement extraction**

Start conservative:
- File paths matching `(?:[\w.-]+/)+[\w.-]+`.
- Backticked qualified symbols like `ToolRegistry.get_entry`.
- Backticked tool/subagent/integration names only if present in generated inventory.

Avoid `grep`; use Python `Path.exists()` and AST for symbols.

**Step 4: Run tests and commit**

```bash
pytest tests/self_knowledge/test_drift.py -q
git add hermes_cli/self_knowledge/drift.py context/self/.hermes-agent-allowlist.txt tests/self_knowledge/test_drift.py
git commit -m "feat: add self-knowledge drift checker"
```

### Task 9: Add `hermes self-knowledge` CLI

**Objective:** Expose render, refresh, and drift checks as official CLI commands.

**Files:**
- Create: `hermes_cli/self_knowledge/cli.py`
- Modify: `hermes_cli/main.py`
- Create: `tests/self_knowledge/test_cli.py`

**Step 1: Add CLI module**

Support:

```bash
hermes self-knowledge --render
hermes self-knowledge --refresh
hermes self-knowledge --check
hermes self-knowledge --check --strict
```

Return codes:
- render: `0`
- refresh: `0`
- check soft: `0`, prints warnings
- check strict: `1` if any finding

**Step 2: Wire into `hermes_cli/main.py`**

Add a top-level parser near other maintenance subcommands. Keep import lazy:

```python
self_knowledge_parser = subparsers.add_parser('self-knowledge', help='Render/check Hermes self-knowledge')
from hermes_cli.self_knowledge.cli import configure_parser as configure_self_knowledge_parser
configure_self_knowledge_parser(self_knowledge_parser)
```

**Step 3: Run tests and smoke commands**

```bash
pytest tests/self_knowledge/test_cli.py -q
hermes self-knowledge --render >/tmp/hermes-self-knowledge.md
hermes self-knowledge --check
```

Expected: no crash; soft warnings acceptable if narrative is not final.

**Step 4: Commit**

```bash
git add hermes_cli/main.py hermes_cli/self_knowledge/cli.py tests/self_knowledge/test_cli.py
git commit -m "feat: add self-knowledge CLI"
```

---

## Phase 4 — Pre-Commit Hook + CI Soft Check

### Task 10: Add idempotent hook installer

**Objective:** Install a local hook that refreshes and stages the self-knowledge doc without clobbering existing hooks.

**Files:**
- Create: `scripts/install-self-knowledge-hook.sh`
- Create: `tests/self_knowledge/test_hook_installer.py` or shell test if existing convention prefers it

**Step 1: Implement installer behavior**

Requirements:
- Idempotent; running twice does not duplicate the block.
- Refuses to overwrite a foreign hook unless `--force` is passed.
- Hook body runs:

```bash
hermes self-knowledge --refresh
if ! git diff --quiet -- context/self/hermes-agent.md; then
  git add context/self/hermes-agent.md
fi
```

**Step 2: Verify manually in a temp repo**

Use a temp directory in tests, not the real `.git/hooks`.

**Step 3: Commit**

```bash
git add scripts/install-self-knowledge-hook.sh tests/self_knowledge/test_hook_installer.py
git commit -m "feat: add self-knowledge pre-commit hook installer"
```

### Task 11: Add GitHub Actions soft check

**Objective:** Warn on drift in PRs without blocking initially.

**Files:**
- Modify: `.github/workflows/tests.yml` or create `.github/workflows/self-knowledge.yml`

**Step 1: Prefer a separate workflow**

Create `.github/workflows/self-knowledge.yml`:

```yaml
name: Self Knowledge
on:
  pull_request:
  push:
    branches: [main]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --all-extras --dev
      - run: uv run hermes self-knowledge --check || true
```

Soft by design. Once stable, remove `|| true` and use `--strict`.

**Step 2: Commit**

```bash
git add .github/workflows/self-knowledge.yml
git commit -m "ci: add soft self-knowledge drift check"
```

---

## Phase 5 — Prompt Injection

### Task 12: Add slim-summary builder

**Objective:** Build a small prompt-safe summary from the rendered doc.

**Files:**
- Create: `hermes_cli/self_knowledge/summary.py`
- Create: `tests/self_knowledge/test_summary.py`

**Step 1: Write tests**

Assert summary includes:
- Identity heading/body excerpt
- Operating principles excerpt
- Capability names only, not full descriptions
- Maximum approximate character budget, e.g. `<= 2500` chars

**Step 2: Implement summary**

Recommended API:

```python
def build_slim_summary(doc_text: str, max_chars: int = 2500) -> str: ...
def load_slim_summary(path: Path = DOC_PATH) -> str: ...
```

If doc is missing, return empty string. No crashes in prompt assembly.

**Step 3: Commit**

```bash
pytest tests/self_knowledge/test_summary.py -q
git add hermes_cli/self_knowledge/summary.py tests/self_knowledge/test_summary.py
git commit -m "feat: add self-knowledge slim summary"
```

### Task 13: Inject summary into prompt behind config flag

**Objective:** Make Hermes aware of its current capabilities without blowing up prompt size or cache stability.

**Files:**
- Modify: `agent/prompt_builder.py`
- Modify: `tests/run_agent/test_run_agent.py` or add `tests/self_knowledge/test_prompt_injection.py`
- Optionally document config in website docs later

**Step 1: Add config gate**

Use config shape:

```yaml
self_knowledge:
  enabled: true
  prompt_mode: slim   # off|slim|full later; default slim
```

Do not require users to configure this manually. Default should be safe and small.

**Step 2: Add prompt block function**

In `agent/prompt_builder.py`, add a small helper near other prompt section renderers:

```python
def render_self_knowledge_prompt_block(config: dict | None = None) -> str:
    mode = ((config or {}).get('self_knowledge') or {}).get('prompt_mode', 'slim')
    enabled = ((config or {}).get('self_knowledge') or {}).get('enabled', True)
    if not enabled or mode == 'off':
        return ''
    try:
        from hermes_cli.self_knowledge.summary import load_slim_summary
        summary = load_slim_summary()
    except Exception:
        return ''
    if not summary.strip():
        return ''
    return '## Hermes self-knowledge\n' + summary.strip()
```

Then integrate in the existing prompt assembly path with minimal movement. Preserve existing ordering and prompt caching as much as possible.

**Step 3: Add tests**

Tests should verify:
- Disabled config omits block.
- Missing doc omits block cleanly.
- Enabled config includes a known capability name from fixture doc.

**Step 4: Commit**

```bash
pytest tests/self_knowledge/test_prompt_injection.py tests/run_agent/test_run_agent.py -q
git add agent/prompt_builder.py tests/self_knowledge/test_prompt_injection.py
git commit -m "feat: inject slim self-knowledge into system prompt"
```

---

## Final Verification

Run:

```bash
pytest tests/self_knowledge -q
hermes self-knowledge --refresh
hermes self-knowledge --check
python - <<'PY'
from pathlib import Path
text = Path('context/self/hermes-agent.md').read_text()
assert '<!-- AUTO-START: capabilities -->' in text
assert 'web_search' in text or 'terminal' in text
print('self-knowledge doc looks populated')
PY
git diff --check
```

Optional broader test if time allows:

```bash
scripts/run_tests.sh tests/self_knowledge tests/run_agent/test_run_agent.py
```

---

## Risks / Guardrails

- **Prompt cache churn:** inject only slim summary by default. Full doc should remain optional and off.
- **Import side effects:** use static scans for plugins/platforms where imports may require optional dependencies.
- **Secrets:** never print env/config values. Print provider names and key names only.
- **Hook annoyance:** hook refreshes and stages; CI soft-checks first. Do not strict-gate until the doc has survived real commits.
- **Current dirty tree:** do not touch unrelated modified files from voice work unless this plan is being implemented in a fresh branch/worktree.

---

## Follow-Ups After Initial Ship

1. Promote CI from soft check to `--check --strict` after a week of clean runs.
2. Add profile/persona-specific summaries for Hermes agents like `profile-main`, `profile-heidi`, and `profile-chad`.
3. Add a voice-mode self-test command that confirms Discord voice, STT, TTS, and latency surfaces are configured.
