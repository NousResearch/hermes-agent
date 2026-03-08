# Skill Secure Setup On Load Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** When a skill is loaded through `skill_view()` or a slash command, Hermes should securely request any missing required environment variables outside the LLM chat, store them directly in `~/.hermes/.env`, allow the user to skip setup and continue anyway, and keep platform incompatibility as the only hard skill gating rule.

**Architecture:** Skill frontmatter gains a setup-oriented metadata field for required environment variables. Skill load paths detect missing required env vars before the model sees the skill content, then invoke a CLI-only secure prompt flow that never exposes secret values to the LLM, logs, tool output, or trajectories. If the user skips setup, the skill still loads and the model can proceed with reduced functionality. Gateway and other messaging surfaces do not support secure in-band secret capture and must return a clear local-setup instruction instead.

**Tech Stack:** Python, pytest, Hermes CLI callbacks, `~/.hermes/.env` config helpers, skill loading in `tools/skills_tool.py`, slash command loading in `agent/skill_commands.py`, agent loop hooks in `run_agent.py`, redaction in `agent/redact.py`.

---

### Task 1: Replace hard prerequisite hiding with setup-on-load metadata

**Files:**
- Modify: `tools/skills_tool.py`
- Modify: `agent/prompt_builder.py`
- Test: `tests/tools/test_skills_tool.py`
- Test: `tests/agent/test_prompt_builder.py`

**Step 1: Write the failing tests**

Add tests that assert:
- skills with missing env vars are still listed in `skills_list()`
- skills with missing env vars still appear in `build_skills_system_prompt()`
- platform-incompatible skills remain hidden
- loading a skill exposes missing required env vars as setup metadata instead of deactivating the skill

Example tests:

```python
def test_missing_env_var_skill_still_appears_in_skills_list(tmp_path, monkeypatch):
    monkeypatch.delenv("TENOR_API_KEY", raising=False)
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(
            tmp_path,
            "gif-search",
            frontmatter_extra=(
                "required_environment_variables:\n"
                "  - name: TENOR_API_KEY\n"
                "    prompt: Tenor API key\n"
            ),
        )
        result = json.loads(skills_list())
    assert any(skill["name"] == "gif-search" for skill in result["skills"])
```

```python
def test_platform_incompatible_skill_is_still_hidden(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skill_dir = tmp_path / "skills" / "apple" / "imessage"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: imessage\ndescription: mac only\nplatforms: [macos]\n---\n"
    )
    with patch("tools.skills_tool.sys") as mock_sys:
        mock_sys.platform = "linux"
        result = build_skills_system_prompt()
    assert "imessage" not in result
```

**Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tests/tools/test_skills_tool.py tests/agent/test_prompt_builder.py -q
```

Expected: failure because missing env vars are still treated as readiness failures in parts of the current code.

**Step 3: Write minimal implementation**

In `tools/skills_tool.py`:
- add support for a new top-level frontmatter field:

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get an API key from https://developers.google.com/tenor
    required_for: full functionality
```

- normalize old `prerequisites.env_vars` into the new internal representation for backward compatibility
- stop using missing env vars to hide or suppress skills from discovery
- keep `platforms` as the only hard compatibility gate
- expose setup metadata in `skill_view()` payloads:
  - `missing_required_environment_variables`
  - `setup_needed: true|false`
  - `setup_help`

In `agent/prompt_builder.py`:
- do not exclude skills for missing env vars or commands
- continue excluding only platform-incompatible skills

**Step 4: Run tests to verify they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tests/tools/test_skills_tool.py tests/agent/test_prompt_builder.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tools/skills_tool.py agent/prompt_builder.py tests/tools/test_skills_tool.py tests/agent/test_prompt_builder.py
git commit -m "feat: keep setup-needed skills discoverable"
```

### Task 2: Add secure required-env collection during skill load in CLI

**Files:**
- Modify: `tools/skills_tool.py`
- Modify: `agent/skill_commands.py`
- Modify: `run_agent.py`
- Modify: `cli.py`
- Modify: `hermes_cli/callbacks.py`
- Modify: `hermes_cli/config.py`
- Test: `tests/tools/test_skills_tool.py`
- Test: `tests/agent/test_skill_commands.py`
- Test: `tests/test_run_agent.py`

**Step 1: Write the failing tests**

Add tests that assert:
- `skill_view()` or slash-command skill load triggers a secure setup callback before returning full content when required env vars are missing
- the callback stores the secret directly in `.env`
- the secret value is never returned to the model
- skip is allowed and still returns the skill content

Example tests:

```python
def test_skill_view_requests_missing_env_securely_and_continues(monkeypatch, tmp_path):
    events = []

    def fake_secret_callback(request):
        events.append(request)
        return {"success": True, "stored_as": "TENOR_API_KEY", "skipped": False}

    monkeypatch.setattr("tools.skills_tool._secret_capture_callback", fake_secret_callback)
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(
            tmp_path,
            "gif-search",
            frontmatter_extra=(
                "required_environment_variables:\n"
                "  - name: TENOR_API_KEY\n"
                "    prompt: Tenor API key\n"
            ),
        )
        result = json.loads(skill_view("gif-search"))
    assert result["success"] is True
    assert events[0]["env_var"] == "TENOR_API_KEY"
```

```python
def test_skill_load_allows_skip(monkeypatch, tmp_path):
    def fake_secret_callback(request):
        return {"success": True, "stored_as": request["env_var"], "skipped": True}
    monkeypatch.setattr("tools.skills_tool._secret_capture_callback", fake_secret_callback)
    ...
    assert result["success"] is True
    assert result["setup_skipped"] is True
```

**Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tests/tools/test_skills_tool.py tests/agent/test_skill_commands.py tests/test_run_agent.py -q
```

Expected: failure because there is no setup-on-load callback path yet.

**Step 3: Write minimal implementation**

In `tools/skills_tool.py`:
- add a callback registration helper, e.g.:

```python
_secret_capture_callback = None

def set_secret_capture_callback(cb):
    global _secret_capture_callback
    _secret_capture_callback = cb
```

- during `skill_view()` and any slash-command skill load path, detect missing required env vars
- if running in CLI and callback exists, request secrets one-by-one before returning the final skill payload
- if the user skips, continue loading the skill and include `setup_skipped: true`
- if the user provides the value, write it directly to `.env` via `save_env_value()` without returning the raw value

In `hermes_cli/callbacks.py`:
- add a hidden-input prompt for env-var setup, using the same UI discipline as sudo password handling
- return metadata only:

```python
{
    "success": True,
    "stored_as": "TENOR_API_KEY",
    "skipped": False,
    "validated": False,
}
```

In `cli.py` / `run_agent.py`:
- register the callback with the skills layer before conversation handling starts

In `agent/skill_commands.py`:
- ensure slash-command invocation uses the same skill-load path so secure setup happens there too

**Step 4: Run tests to verify they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tests/tools/test_skills_tool.py tests/agent/test_skill_commands.py tests/test_run_agent.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tools/skills_tool.py agent/skill_commands.py run_agent.py cli.py hermes_cli/callbacks.py hermes_cli/config.py tests/tools/test_skills_tool.py tests/agent/test_skill_commands.py tests/test_run_agent.py
git commit -m "feat: securely collect required env vars on skill load"
```

### Task 3: Make gateway and messaging surfaces refuse in-band secret entry

**Files:**
- Modify: `tools/skills_tool.py`
- Modify: `run_agent.py`
- Modify: `gateway/platforms/base.py` if needed for a capability flag
- Test: `tests/test_run_agent.py`
- Test: `tests/gateway/test_platform_base.py`

**Step 1: Write the failing tests**

Add tests that assert:
- if a skill load on a gateway surface needs a required env var, Hermes does not ask for the secret in chat
- Hermes returns setup guidance telling the user to use local CLI or `hermes setup`
- the skill can still load if the design says “continue anyway”, but the setup path is clearly marked unavailable on gateway

Example test:

```python
def test_gateway_skill_load_refuses_secret_capture_and_returns_local_setup_guidance():
    result = _handle_skill_setup_request(..., surface="gateway")
    assert result["success"] is False
    assert result["reason"] == "unsupported_surface"
    assert "hermes setup" in result["message"]
```

**Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tests/test_run_agent.py tests/gateway/test_platform_base.py -q
```

Expected: failure because surface-specific refusal is not implemented.

**Step 3: Write minimal implementation**

In the setup callback dispatch path:
- detect gateway / messaging context from existing session metadata or env
- refuse secure secret entry in-band and return a structured message:

```python
{
    "success": False,
    "reason": "unsupported_surface",
    "message": "Secure secret entry is not supported over messaging. Run `hermes setup` or update ~/.hermes/.env locally.",
}
```

In `tools/skills_tool.py`:
- if this happens during skill load, continue returning the skill payload but include setup guidance and `secure_setup_available: false`

**Step 4: Run tests to verify they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tests/test_run_agent.py tests/gateway/test_platform_base.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tools/skills_tool.py run_agent.py gateway/platforms/base.py tests/test_run_agent.py tests/gateway/test_platform_base.py
git commit -m "feat: refuse in-band skill secret entry on gateway"
```

### Task 4: Ensure secrets never reach logs, tool output, or trajectories

**Files:**
- Modify: `agent/redact.py`
- Modify: `run_agent.py`
- Modify: `agent/trajectory.py`
- Modify: `agent/display.py`
- Test: `tests/agent/test_redact.py`
- Test: `tests/test_run_agent.py`

**Step 1: Write the failing tests**

Add tests that assert:
- submitted env-var secrets are never present in trajectory export
- CLI activity feed does not include secret values
- log redaction covers any structured callback result related to secure setup
- only variable names and statuses are visible

Example test:

```python
def test_secret_capture_value_not_present_in_display_or_trajectory():
    secret = "sk-test-secret-123"
    rendered = render_event({"stored_as": "TENOR_API_KEY", "raw": secret})
    assert secret not in rendered
```

**Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tests/agent/test_redact.py tests/test_run_agent.py -q
```

Expected: failure because new secret-capture plumbing is not yet fully redacted everywhere.

**Step 3: Write minimal implementation**

In `run_agent.py` and `agent/trajectory.py`:
- store only metadata from secret capture events
- never serialize user-entered secret values

In `agent/display.py`:
- render generic messages like:

```text
🔐 stored secret TENOR_API_KEY
```

In `agent/redact.py`:
- extend redaction to cover any env-write / setup-event patterns introduced by this feature

**Step 4: Run tests to verify they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tests/agent/test_redact.py tests/test_run_agent.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add agent/redact.py run_agent.py agent/trajectory.py agent/display.py tests/agent/test_redact.py tests/test_run_agent.py
git commit -m "fix: redact secure skill setup data"
```

### Task 5: Add backward-compatible parsing and namespacing guidance for required env vars

**Files:**
- Modify: `tools/skills_tool.py`
- Modify: `AGENTS.md`
- Test: `tests/tools/test_skills_tool.py`

**Step 1: Write the failing tests**

Add tests that assert:
- legacy `prerequisites.env_vars` still works as setup metadata
- new `required_environment_variables` format parses correctly
- malformed entries degrade safely
- duplicate env-var names are normalized cleanly

Example test:

```python
def test_legacy_prerequisites_env_vars_are_normalized_to_required_env_vars():
    fm, _ = _parse_frontmatter(
        "---\nname: x\ndescription: y\nprerequisites:\n  env_vars: [TENOR_API_KEY]\n---\n"
    )
    normalized = _get_required_environment_variables(fm)
    assert normalized[0]["name"] == "TENOR_API_KEY"
```

**Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tests/tools/test_skills_tool.py -q
```

Expected: failure because normalization and new parsing are not fully implemented.

**Step 3: Write minimal implementation**

In `tools/skills_tool.py`:
- add a normalization helper:

```python
def _get_required_environment_variables(frontmatter: dict) -> list[dict]:
    ...
```

- support both:
  - legacy `prerequisites.env_vars`
  - new `required_environment_variables`

In `AGENTS.md`:
- document preferred env-var naming conventions
- recommend provider-scoped names
- document that missing env vars trigger secure setup-on-load in CLI and are skippable
- document that gateway must use local setup instead

**Step 4: Run tests to verify they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tests/tools/test_skills_tool.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tools/skills_tool.py AGENTS.md tests/tools/test_skills_tool.py
git commit -m "docs: add required env var setup metadata"
```

### Task 6: Full verification pass

**Files:**
- Test only

**Step 1: Run focused suites**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tests/tools/test_skills_tool.py \
  tests/tools/test_skill_view_traversal.py \
  tests/tools/test_skill_view_path_check.py \
  tests/agent/test_prompt_builder.py \
  tests/agent/test_skill_commands.py \
  tests/test_run_agent.py \
  tests/hermes_cli/test_config.py \
  tests/agent/test_redact.py \
  tests/gateway/test_platform_base.py -q
```

Expected: PASS.

**Step 2: Run adjacent terminal/file suites**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tests/tools/test_file_tools.py \
  tests/tools/test_terminal_disk_usage.py \
  tests/tools/test_daytona_environment.py -q
```

Expected: PASS.

**Step 3: Manual CLI verification**

Run:

```bash
source .venv/bin/activate && hermes chat -q "Use the gif-search skill and tell me what setup is needed"
```

Expected:
- the skill remains discoverable
- if loaded and `TENOR_API_KEY` is missing, CLI asks for it securely outside chat
- if skipped, the skill still loads and the model proceeds
- the secret value is not shown in terminal output or logs

Then test slash command load manually:
- run a slash command for a skill with missing required env vars
- verify the same hidden prompt flow happens
- skip once and verify the skill still loads

**Step 4: Final integration commit**

```bash
git status
git log --oneline -n 10
```

If clean and verified, create a final integration commit:

```bash
git add -A
git commit -m "feat: add secure skill setup on load"
```
