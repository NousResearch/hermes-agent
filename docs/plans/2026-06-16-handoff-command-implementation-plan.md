# /handoff Command Implementation Plan

> For Hermes: use subagent-driven-development skill to implement this plan task-by-task.

Goal: add a first-class `/handoff` workflow to Hermes that packages a narrow slice of context for a fresh session and can optionally consume a saved handoff, while keeping the implementation on the "CLI command + shared helper" rung instead of adding a new core model tool.

Architecture: implement `/handoff` as a slash command plus a shared helper module under `hermes_cli/`, following the existing `blueprint_cmd.py` pattern. Start with a CLI-first MVP that supports generation, save/inline modes, and file-based consume. Keep gateway parity as a follow-up unless the code path is already cheap to share safely.

Tech Stack: Python, existing Hermes CLI slash-command registry, `SessionDB`/conversation history already present in `cli.py`, filesystem helpers via `get_hermes_home()`, pytest.

---

## Ground truth gathered before implementation

Relevant existing files and behavior verified in this checkout:
- Repo root: `/usr/local/lib/hermes-agent`
- Slash command registry lives in: `/usr/local/lib/hermes-agent/hermes_cli/commands.py`
- CLI slash dispatch lives in: `/usr/local/lib/hermes-agent/cli.py`
  - `process_command()` starts at around line 7224
  - `/save` dispatches to `save_conversation()` around lines 7468-7469
  - `/compress` dispatches to `_manual_compress()` around lines 7503-7504
- Existing one-shot seed pattern exists in:
  - `/usr/local/lib/hermes-agent/hermes_cli/cli_commands_mixin.py`
  - `_handle_blueprint_command()` sets `self._pending_agent_seed`
- Existing save-location precedent exists in:
  - `/usr/local/lib/hermes-agent/cli.py` `save_conversation()`
  - `/usr/local/lib/hermes-agent/tests/cli/test_save_conversation_location.py`
- Existing registry tests live in:
  - `/usr/local/lib/hermes-agent/tests/hermes_cli/test_commands.py`
- Existing CLI command tests and gateway command tests exist in:
  - `/usr/local/lib/hermes-agent/tests/cli/`
  - `/usr/local/lib/hermes-agent/tests/gateway/`

Design constraint from `AGENTS.md`: prefer a CLI command + skill over adding a new core tool.

---

## Scope for MVP

Implement now:
- `/handoff`
- `/handoff <mission>`
- `/handoff inline [mission]`
- `/handoff save [path] [mission]`
- `/handoff consume <path>`

Defer for follow-up unless implementation ends up trivial:
- gateway-native `/handoff`
- `/handoff --from-session <id>`
- automatic cross-profile launch
- direct `/handoff run`
- non-file `consume` targets like `@session:<profile>/<id>`

Reason: keep the first shipped version small, testable, and aligned with the existing skill-first prototype.

---

## Output contract for MVP

The generator must produce:
- a concise markdown handoff
- a suggested filename
- a recommended next-step mode

Required handoff sections:
- Title
- Purpose of next session
- Current status
- Relevant artifacts
- Constraints and non-goals
- Recommended skills
- Recommended toolsets
- Exact first prompt
- Success criteria

Saved handoffs should default under:
- `get_hermes_home() / "sessions" / "handoffs"`

This mirrors `/save` using Hermes-owned storage instead of the current working directory.

---

## Task 1: Add registry coverage for `/handoff`

Objective: register the new slash command in the central command registry without yet implementing behavior.

Files:
- Modify: `/usr/local/lib/hermes-agent/hermes_cli/commands.py`
- Modify: `/usr/local/lib/hermes-agent/tests/hermes_cli/test_commands.py`

Step 1: Add the command definition.

Add a `CommandDef` near other Session commands:

```python
CommandDef(
    "handoff",
    "Create or consume a scoped handoff document",
    "Session",
    cli_only=True,
    args_hint="[inline|save|consume] [args...]",
    subcommands=("inline", "save", "consume"),
)
```

Step 2: Extend registry tests.

Add assertions covering:
- `/handoff` exists in `COMMANDS`
- it resolves correctly through `resolve_command("handoff")`
- `cli_only=True` keeps it out of gateway-known commands unless later widened intentionally

Step 3: Run the focused tests.

Run:
`python -m pytest tests/hermes_cli/test_commands.py -q -o 'addopts='`

Expected: PASS

Step 4: Commit.

```bash
git add hermes_cli/commands.py tests/hermes_cli/test_commands.py
git commit -m "feat: register handoff slash command"
```

---

## Task 2: Create a shared `handoff_cmd` helper module

Objective: put parsing, generation, serialization, and consume parsing in one shared place instead of burying it in `cli.py`.

Files:
- Create: `/usr/local/lib/hermes-agent/hermes_cli/handoff_cmd.py`
- Test: `/usr/local/lib/hermes-agent/tests/hermes_cli/test_handoff_cmd.py`

Step 1: Define small result dataclasses.

Create dataclasses similar in spirit to `BlueprintCommandResult`:

```python
@dataclass
class HandoffDocument:
    markdown: str
    suggested_filename: str
    next_mode: str

@dataclass
class HandoffCommandResult:
    text: str
    saved_path: str | None = None
    agent_seed: str | None = None
```
```

Step 2: Implement pure helpers first.

Add pure functions for:
- command arg parsing
- slug generation
- default handoff path generation
- markdown assembly
- lightweight section validation for consume

Suggested function names:
- `parse_handoff_args(raw_args: str) -> tuple[str, str | None, str | None]`
- `build_handoff_document(...) -> HandoffDocument`
- `default_handoff_path(hermes_home, slug, now=None) -> Path`
- `parse_handoff_markdown(text: str) -> dict`
- `build_handoff_consume_seed(parsed: dict, source_path: str) -> str`

Step 3: Write pure unit tests before wiring CLI.

Test cases:
- `save` path parsing
- `inline` parsing
- default filename formatting
- missing required sections rejected on consume
- consume seed contains mission + constraints + success criteria

Step 4: Run tests.

Run:
`python -m pytest tests/hermes_cli/test_handoff_cmd.py -q -o 'addopts='`

Expected: PASS

Step 5: Commit.

```bash
git add hermes_cli/handoff_cmd.py tests/hermes_cli/test_handoff_cmd.py
git commit -m "feat: add shared handoff command helpers"
```

---

## Task 3: Build the markdown generator around real CLI context

Objective: convert current-session context into a narrow handoff artifact without adding a new model tool.

Files:
- Modify: `/usr/local/lib/hermes-agent/hermes_cli/handoff_cmd.py`
- Test: `/usr/local/lib/hermes-agent/tests/hermes_cli/test_handoff_cmd.py`

Step 1: Define the minimal context inputs.

The generator should accept concrete inputs from the CLI layer, not reach into global state itself:

```python
def build_handoff_document(
    *,
    mission: str | None,
    conversation_history: list[dict],
    session_id: str | None,
    workdir: str | None,
    suggested_skills: list[str] | None = None,
    suggested_toolsets: list[str] | None = None,
) -> HandoffDocument:
    ...
```
```

Step 2: Keep the generator deterministic.

For MVP, do not call the model.

Use deterministic heuristics:
- title from explicit mission or last user message
- current status from the last 1-3 user/assistant exchanges
- relevant artifacts from obvious absolute paths / URLs found in recent messages if present
- default skills/toolsets conservative and short
- next mode defaults to `fresh session`

Step 3: Make explicit mission win.

If the user runs `/handoff fix auth drift`, that mission should drive:
- title
- purpose
- exact first prompt
- suggested filename slug

Step 4: Add tests proving the generator stays narrow.

Assertions:
- markdown contains all required sections
- markdown stays shorter than an intentionally bloated transcript fixture
- explicit mission overrides transcript-derived title

Step 5: Run tests.

Run:
`python -m pytest tests/hermes_cli/test_handoff_cmd.py -q -o 'addopts='`

Expected: PASS

Step 6: Commit.

```bash
git add hermes_cli/handoff_cmd.py tests/hermes_cli/test_handoff_cmd.py
git commit -m "feat: generate deterministic handoff markdown"
```

---

## Task 4: Wire `/handoff` into CLI dispatch

Objective: expose the new command through the existing slash-command flow.

Files:
- Modify: `/usr/local/lib/hermes-agent/cli.py`
- Modify: `/usr/local/lib/hermes-agent/hermes_cli/cli_commands_mixin.py`
- Create or modify tests: `/usr/local/lib/hermes-agent/tests/cli/test_handoff_command.py`

Step 1: Add a handler method in the mixin.

Follow the `blueprint` pattern and add:

```python
def _handle_handoff_command(self, cmd: str):
    from hermes_cli.handoff_cmd import handle_handoff_command
    result = handle_handoff_command(
        cmd=cmd,
        conversation_history=self.conversation_history,
        session_id=self.session_id,
        workdir=os.getenv("TERMINAL_CWD", os.getcwd()),
        hermes_home=get_hermes_home(),
    )
    self._console_print(result.text)
    if result.agent_seed:
        self._pending_agent_seed = result.agent_seed
```
```

Step 2: Add dispatch in `process_command()`.

Near other slash-command branches:

```python
elif canonical == "handoff":
    self._handle_handoff_command(cmd_original)
```
```

Step 3: Write CLI tests first.

Test cases:
- `/handoff` prints markdown or summary output
- `/handoff inline foo` does not write a file
- `/handoff save` writes under Hermes home
- `/handoff save /tmp/x.md foo` writes the explicit path

Step 4: Run tests.

Run:
`python -m pytest tests/cli/test_handoff_command.py -q -o 'addopts='`

Expected: PASS

Step 5: Commit.

```bash
git add cli.py hermes_cli/cli_commands_mixin.py tests/cli/test_handoff_command.py
git commit -m "feat: wire handoff slash command into cli"
```

---

## Task 5: Implement default save location and path safety

Objective: make saved handoffs discoverable and safe by default.

Files:
- Modify: `/usr/local/lib/hermes-agent/hermes_cli/handoff_cmd.py`
- Modify: `/usr/local/lib/hermes-agent/tests/cli/test_handoff_command.py`

Step 1: Use Hermes-owned storage by default.

Default directory:

```python
get_hermes_home() / "sessions" / "handoffs"
```

Suggested filename format:

```text
handoff-YYYYMMDD-HHMM-<slug>.md
```

Step 2: Prevent silent relative-path surprises.

Rules:
- explicit absolute path: honor it
- explicit relative path: resolve relative to current working directory and print the absolute resolved path back
- no path given: use Hermes home handoff dir

Step 3: Write tests mirroring `/save` location discipline.

Specifically prove:
- default output does not land in random CWD
- explicit path works
- returned user-facing text includes absolute saved path

Step 4: Run tests.

Run:
`python -m pytest tests/cli/test_handoff_command.py -q -o 'addopts='`

Expected: PASS

Step 5: Commit.

```bash
git add hermes_cli/handoff_cmd.py tests/cli/test_handoff_command.py
git commit -m "feat: save handoffs under hermes home by default"
```

---

## Task 6: Implement `/handoff consume <path>` using the existing agent-seed pattern

Objective: consuming a handoff should validate the file, restate the mission, and kick off the next turn without bespoke runtime plumbing.

Files:
- Modify: `/usr/local/lib/hermes-agent/hermes_cli/handoff_cmd.py`
- Modify: `/usr/local/lib/hermes-agent/tests/cli/test_handoff_command.py`
- Optional test: `/usr/local/lib/hermes-agent/tests/cli/test_handoff_consume_seed.py`

Step 1: Parse and validate the saved markdown.

Required sections to validate:
- Purpose of next session
- Current status
- Relevant artifacts
- Constraints and non-goals
- Exact first prompt
- Success criteria

If missing, return a clear error text and do not set an agent seed.

Step 2: Build the seed text.

The consume seed should tell the agent to:
- restate the mission in one sentence
- validate referenced artifacts first
- honor constraints/non-goals
- begin with the handoff's exact first prompt or an operational equivalent

Step 3: Reuse `_pending_agent_seed`.

This avoids inventing a new command-execution mechanism and follows the already-proven `/blueprint` pattern.

Step 4: Add tests.

Test cases:
- valid handoff file -> `agent_seed` is set
- invalid handoff file -> clear error, no seed
- seed text includes success criteria and source path

Step 5: Run tests.

Run:
`python -m pytest tests/cli/test_handoff_command.py -q -o 'addopts='`

Expected: PASS

Step 6: Commit.

```bash
git add hermes_cli/handoff_cmd.py tests/cli/test_handoff_command.py
git commit -m "feat: add handoff consume flow"
```

---

## Task 7: Add documentation and help-text coverage

Objective: make the command discoverable and explain that it complements `/compress` rather than replacing it.

Files:
- Modify: `/usr/local/lib/hermes-agent/hermes_cli/commands.py`
- Modify: `/usr/local/lib/hermes-agent/AGENTS.md` only if contributor guidance needs a tiny note
- Modify or create: `/usr/local/lib/hermes-agent/website/docs/reference/slash-commands.md` if the docs source exists in this tree
- Modify tests if needed: `/usr/local/lib/hermes-agent/tests/hermes_cli/test_commands.py`

Step 1: Tighten command description/help text.

The user-facing phrasing should reinforce:
- `/compress` = compress same thread
- `/handoff` = package context for another/fresh thread

Step 2: Update docs source if present.

If slash-command docs are generated from registry, only update the registry description.
If there is a curated docs page in `website/docs/`, add a short `/handoff` section.

Step 3: Run the affected tests.

Run:
`python -m pytest tests/hermes_cli/test_commands.py -q -o 'addopts='`

Expected: PASS

Step 4: Commit.

```bash
git add hermes_cli/commands.py website/docs/reference/slash-commands.md tests/hermes_cli/test_commands.py
git commit -m "docs: document handoff slash command"
```

Note: if the docs file path differs, commit the real touched path instead.

---

## Task 8: End-to-end verification

Objective: prove the feature works through the real CLI path, not only through helper-unit tests.

Files:
- Reuse existing tests plus any new CLI integration test if needed
- Optional create: `/usr/local/lib/hermes-agent/tests/cli/test_handoff_e2e.py`

Step 1: Add one integration-style CLI test.

Scenario:
- seed a fake conversation history
- run `/handoff save auth-review`
- assert file written under Hermes home
- run `/handoff consume <that-file>`
- assert `_pending_agent_seed` becomes non-empty

Step 2: Run the full focused suite.

Run:
```bash
python -m pytest \
  tests/hermes_cli/test_commands.py \
  tests/hermes_cli/test_handoff_cmd.py \
  tests/cli/test_handoff_command.py \
  -q -o 'addopts='
```

Expected: PASS

Step 3: Optional wider confidence run.

Run:
```bash
python -m pytest \
  tests/cli/test_save_conversation_location.py \
  tests/cli/test_branch_command.py \
  tests/gateway/test_compress_command.py \
  tests/hermes_cli/test_commands.py \
  tests/hermes_cli/test_handoff_cmd.py \
  tests/cli/test_handoff_command.py \
  -q -o 'addopts='
```

Expected: PASS

Step 4: Commit.

```bash
git add tests/hermes_cli/test_handoff_cmd.py tests/cli/test_handoff_command.py
if [ -f tests/cli/test_handoff_e2e.py ]; then git add tests/cli/test_handoff_e2e.py; fi
git commit -m "test: cover handoff command end to end"
```

---

## Follow-up work after MVP lands

Do not include in the first PR unless the MVP is already clean and green.

1. Gateway parity
   - decide whether `/handoff` should be gateway-visible or remain CLI-first
   - if added to gateway, mirror the safe subset and add dedicated gateway tests

2. `--from-session <id>` support
   - likely shared with `session_search`
   - keep it deterministic; do not auto-load huge transcripts

3. richer artifact extraction
   - detect file paths / URLs from recent turns more intelligently
   - still keep output narrow and bounded

4. direct run mode
   - `/handoff run` could generate and immediately branch/spawn a fresh session
   - only after the basic save/consume lifecycle proves itself

---

## Common pitfalls to avoid

1. Do not implement `/handoff` as a new model tool.
   This belongs on the CLI-command rung, not the core tool-schema rung.

2. Do not let the generator become a transcript dump.
   It must stay narrow and artifact-first.

3. Do not ship gateway exposure accidentally.
   Keep `cli_only=True` for MVP unless gateway handling is intentionally implemented.

4. Do not invent a second "seed the next turn" mechanism.
   Reuse `_pending_agent_seed`.

5. Do not save handoffs into arbitrary CWD by default.
   Use Hermes-owned storage, mirroring `/save` discipline.

---

## Verification checklist

- [ ] `/handoff` is registered in `hermes_cli/commands.py`
- [ ] command dispatch in `cli.py` reaches a dedicated handler
- [ ] shared handoff logic lives in `hermes_cli/handoff_cmd.py`
- [ ] default saved handoffs land under `~/.hermes/sessions/handoffs/`
- [ ] `/handoff consume <path>` uses `_pending_agent_seed`
- [ ] focused tests pass with `-o 'addopts='`
- [ ] MVP remains CLI-first and does not add a new core model tool
