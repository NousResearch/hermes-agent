# Reduce Tool Context Overhead Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make delegated MCP access explicit and suppress unchanged repeated `skill_view` bodies within one process-local session.

**Architecture:** Delegate MCP inheritance defaults to false in source and live config while explicit profile opt-in remains supported. `tools/skills_tool.py` owns a bounded, thread-safe, fail-open session map keyed by canonical skill identity and rendered-content SHA-256; the registry wrapper passes the existing `session_id` into this path.

**Tech Stack:** Python 3.11+, pytest, Hermes tool registry, YAML configuration, SHA-256, `OrderedDict`, `threading.RLock`.

## Global Constraints

- Parent-agent MCP access remains unchanged.
- Children without explicit toolsets retain current complete-parent-toolset behavior.
- Missing session IDs and dedupe-state failures return full skill content.
- Hash final rendered content after linked-file selection and preprocessing.
- Do not persist dedupe state across gateway restart, resume, or compression-created session changes.
- Counters contain metadata only and never skill content, rendered output, command output, or secrets.
- Preserve existing skill preprocessing and usage-counter behavior.
- Follow RED, GREEN, REFACTOR for every production behavior.
- Do not add a lifecycle abstraction or database state.

---

## File Structure

- Modify `hermes_cli/config.py`: change default delegate MCP inheritance policy.
- Modify `tools/delegate_tool.py`: make runtime fallback match config default.
- Modify `tests/tools/test_delegate.py`: lock root/profile opt-in and strict-intersection behavior.
- Modify `tools/skills_tool.py`: own bounded process-local dedupe state, receipt construction, counters, and registry session propagation.
- Modify `tests/tools/test_skills_tool.py`: cover local/main/reference/session/hash/eviction/fail-open behavior.
- Modify `tests/test_plugin_skills.py`: cover plugin-qualified identity isolation through existing plugin fixture.
- Modify `/Users/atorres/.hermes/config.yaml`: disable live root inheritance and remove blanket profile overrides.

### Task 1: Make Delegate MCP Inheritance Explicit

**Files:**
- Modify: `hermes_cli/config.py:2139-2145`
- Modify: `tools/delegate_tool.py:660-664`
- Modify: `tests/tools/test_delegate.py:2601-2652`

**Interfaces:**
- Consumes: merged delegation config dictionaries.
- Produces: `_get_inherit_mcp_toolsets(cfg: dict | None = None) -> bool`, defaulting to `False`; profile-level `true` still opts in.

- [ ] **Step 1: Replace the old-default test with failing opt-in-policy tests**

Add or update tests under `TestBuildChildAgent` using `_make_mock_parent()` and existing child-agent patches:

```python
def test_build_child_agent_root_false_strict_intersection(self):
    parent = _make_mock_parent()
    parent.enabled_toolsets = ["file", "mcp-activix_lsp"]
    cfg = {
        "inherit_mcp_toolsets": False,
        "toolsets": ["file"],
    }

    child = _build_child_agent(parent, cfg, "task")

    assert child.enabled_toolsets == ["file"]


def test_build_child_agent_profile_true_opts_back_into_mcp_inheritance(self):
    parent = _make_mock_parent()
    parent.enabled_toolsets = ["file", "mcp-activix_lsp"]
    cfg = {
        "inherit_mcp_toolsets": True,
        "toolsets": ["file"],
    }

    child = _build_child_agent(parent, cfg, "task")

    assert child.enabled_toolsets == ["file", "mcp-activix_lsp"]


def test_build_child_agent_explicit_mcp_toolset_survives_strict_intersection(self):
    parent = _make_mock_parent()
    parent.enabled_toolsets = ["file", "mcp-activix_lsp"]
    cfg = {
        "inherit_mcp_toolsets": False,
        "toolsets": ["file", "activix_lsp"],
    }

    child = _build_child_agent(parent, cfg, "task")

    assert child.enabled_toolsets == ["file", "mcp-activix_lsp"]
```

Retain focused coverage proving profile `false` overrides root `true` after `_merge_delegation_profile()`, and proving no explicit `toolsets` keeps the parent toolset list.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python -m pytest \
  tests/tools/test_delegate.py::TestBuildChildAgent::test_build_child_agent_root_false_strict_intersection \
  tests/tools/test_delegate.py::TestBuildChildAgent::test_build_child_agent_profile_true_opts_back_into_mcp_inheritance \
  tests/tools/test_delegate.py::TestBuildChildAgent::test_build_child_agent_explicit_mcp_toolset_survives_strict_intersection \
  -q
```

Expected: at least the default-policy assertion fails because source still defaults to inheritance.

- [ ] **Step 3: Change source defaults without changing no-explicit-toolsets behavior**

In `hermes_cli/config.py`, set:

```python
"inherit_mcp_toolsets": False,
```

In `tools/delegate_tool.py`, use:

```python
def _get_inherit_mcp_toolsets(cfg=None) -> bool:
    source = cfg if cfg is not None else _load_config()
    return _coerce_bool(source.get("inherit_mcp_toolsets"), default=False)
```

Adapt names to the current helper already used by this function. Do not alter `_build_child_agent()` branches except where tests reveal a production-shaped alias needs existing registry setup.

- [ ] **Step 4: Run focused and full delegate tests and verify GREEN**

Run:

```bash
python -m pytest tests/tools/test_delegate.py -q
```

Expected: PASS with no warnings introduced by this change.

- [ ] **Step 5: Commit Task 1**

```bash
git add hermes_cli/config.py tools/delegate_tool.py tests/tools/test_delegate.py
git commit -m "perf: make delegated MCP access explicit"
```

### Task 2: Deduplicate Unchanged Skill Bodies Per Session

**Files:**
- Modify: `tools/skills_tool.py:69-112,760-869,1194-1304,1451-1534,1630-1652`
- Modify: `tests/tools/test_skills_tool.py:3-21,366-478`
- Modify: `tests/test_plugin_skills.py:160-220`

**Interfaces:**
- Consumes: canonical skill name, optional linked file, final rendered content, optional registry `session_id`.
- Produces: `clear_skill_view_dedupe_session(session_id: str | None) -> None`; extended `skill_view(..., session_id: str | None = None) -> str`; compact successful receipts for unchanged content.

- [ ] **Step 1: Write failing local/session/hash tests**

Create `TestSkillViewDedupe` beside `TestSkillView`. Use unique session IDs and clear state during teardown. Tests must parse returned JSON and assert body presence or absence:

```python
class TestSkillViewDedupe:
    def teardown_method(self):
        clear_skill_view_dedupe_session("session-a")
        clear_skill_view_dedupe_session("session-b")

    def test_same_session_same_main_skill_returns_compact_receipt(self, tmp_path):
        _make_skill(tmp_path, "sample", "# Sample\n\nInstructions")
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            first = json.loads(skill_view("sample", session_id="session-a"))
            second = json.loads(skill_view("sample", session_id="session-a"))

        assert first["content"].endswith("Instructions")
        assert "content" not in second
        assert second["success"] is True
        assert second["unchanged"] is True
        assert second["content_hash"] == first["content_hash"]

    def test_same_session_changed_rendered_content_returns_full_payload(self, tmp_path):
        skill = _make_skill(tmp_path, "sample", "# Sample\n\nFirst")
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            first = json.loads(skill_view("sample", session_id="session-a"))
            skill.joinpath("SKILL.md").write_text("# Sample\n\nSecond")
            second = json.loads(skill_view("sample", session_id="session-a"))

        assert first["content_hash"] != second["content_hash"]
        assert second["content"].endswith("Second")

    def test_different_sessions_each_receive_full_payload(self, tmp_path):
        _make_skill(tmp_path, "sample", "# Sample\n\nInstructions")
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            first = json.loads(skill_view("sample", session_id="session-a"))
            second = json.loads(skill_view("sample", session_id="session-b"))

        assert "content" in first
        assert "content" in second

    def test_missing_session_id_fails_open_with_full_payload(self, tmp_path):
        _make_skill(tmp_path, "sample", "# Sample\n\nInstructions")
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            first = json.loads(skill_view("sample"))
            second = json.loads(skill_view("sample"))

        assert "content" in first
        assert "content" in second
```

Also add focused tests for linked-file identity, rendered preprocessing changes, capacity eviction, state failure, metadata-only counters, and wrapper usage-counter bumps. Patch capacity constants to small values for eviction tests rather than creating hundreds of skills.

- [ ] **Step 2: Write failing registry propagation test**

Exercise `_skill_view_with_bump()` or `registry.dispatch()` with `session_id="session-a"` twice and verify the second response is compact. This proves `session_id`, not `task_id`, owns dedupe identity.

- [ ] **Step 3: Write failing plugin-qualified identity test**

Extend the existing plugin fixture in `tests/test_plugin_skills.py` so local `sample` and plugin-qualified `plugin:sample` with identical bytes each return full content once in the same session.

- [ ] **Step 4: Run focused skill tests and verify RED**

Run:

```bash
python -m pytest \
  tests/tools/test_skills_tool.py::TestSkillViewDedupe \
  tests/test_plugin_skills.py::TestSkillViewQualifiedName \
  -q
```

Expected: collection or assertion failures because `session_id`, content hashes, compact receipts, and clear helper do not exist.

- [ ] **Step 5: Add bounded state and helper functions**

Add imports:

```python
import hashlib
import threading
from collections import OrderedDict
```

Add module state near `logger`:

```python
_SKILL_VIEW_MAIN_MARKER = "<main>"
_SKILL_VIEW_DEDUPE_MAX_SESSIONS = 128
_SKILL_VIEW_DEDUPE_MAX_IDENTITIES_PER_SESSION = 256

_skill_view_dedupe_lock = threading.RLock()
_skill_view_dedupe: OrderedDict[
    str, OrderedDict[tuple[str, str], str]
] = OrderedDict()
_skill_view_dedupe_counters = {
    "skill_view_dedupe_hits": 0,
    "skill_view_chars_avoided": 0,
    "skill_view_approx_tokens_avoided": 0,
}
```

Add helpers:

```python
def _skill_view_identity(
    canonical_name: str,
    linked_file: str | None,
) -> tuple[str, str]:
    return canonical_name, linked_file or _SKILL_VIEW_MAIN_MARKER


def _skill_view_dedupe_before_payload(
    *,
    session_id: str | None,
    canonical_name: str,
    linked_file: str | None,
    rendered_content: str,
) -> tuple[bool, str]:
    content_hash = hashlib.sha256(rendered_content.encode("utf-8")).hexdigest()
    if not session_id or not session_id.strip():
        return False, content_hash

    identity = _skill_view_identity(canonical_name, linked_file)
    try:
        with _skill_view_dedupe_lock:
            session = _skill_view_dedupe.setdefault(session_id, OrderedDict())
            _skill_view_dedupe.move_to_end(session_id)
            unchanged = session.get(identity) == content_hash
            session[identity] = content_hash
            session.move_to_end(identity)

            while len(session) > _SKILL_VIEW_DEDUPE_MAX_IDENTITIES_PER_SESSION:
                session.popitem(last=False)
            while len(_skill_view_dedupe) > _SKILL_VIEW_DEDUPE_MAX_SESSIONS:
                _skill_view_dedupe.popitem(last=False)

            if unchanged:
                chars = len(rendered_content)
                _skill_view_dedupe_counters["skill_view_dedupe_hits"] += 1
                _skill_view_dedupe_counters["skill_view_chars_avoided"] += chars
                _skill_view_dedupe_counters[
                    "skill_view_approx_tokens_avoided"
                ] += (chars + 3) // 4
            return unchanged, content_hash
    except Exception as exc:
        logger.warning(
            "Skill-view dedupe failed open for session=%s skill=%s error=%s",
            session_id,
            canonical_name,
            type(exc).__name__,
        )
        return False, content_hash


def clear_skill_view_dedupe_session(session_id: str | None) -> None:
    if not session_id:
        return
    with _skill_view_dedupe_lock:
        _skill_view_dedupe.pop(session_id, None)
```

Do not log linked-file content, rendered content, exception messages, or hashes.

- [ ] **Step 6: Add receipt construction and integrate all rendering paths**

Add one compact receipt helper:

```python
def _skill_view_unchanged_receipt(
    *,
    canonical_name: str,
    linked_file: str | None,
    content_hash: str,
) -> str:
    payload = {
        "success": True,
        "name": canonical_name,
        "content_hash": content_hash,
        "unchanged": True,
        "message": "Unchanged skill content was already returned in this session.",
    }
    if linked_file:
        payload["file"] = linked_file
    return json.dumps(payload)
```

Extend the public function compatibly:

```python
def skill_view(
    name,
    file_path=None,
    task_id=None,
    preprocess=True,
    session_id=None,
) -> str:
```

For plugin, linked-file, and main-skill paths:

1. Finish current preprocessing/rendering.
2. Call `_skill_view_dedupe_before_payload(...)`.
3. Return `_skill_view_unchanged_receipt(...)` when unchanged.
4. Add `content_hash` to normal successful payloads.
5. Preserve existing fields and errors.

Use plugin canonical identity `f"{namespace}:{bare_name}"`. Use canonical resolved local skill name for local paths. Use the requested normalized linked path for linked files.

- [ ] **Step 7: Pass the actual registry session ID**

Change `_skill_view_with_bump()` to:

```python
result = skill_view(
    name,
    file_path=args.get("file_path"),
    task_id=kw.get("task_id"),
    session_id=kw.get("session_id"),
)
```

Keep successful compact receipts eligible for existing `bump_view()` and `bump_use()` behavior.

- [ ] **Step 8: Run focused tests and verify GREEN**

Run:

```bash
python -m pytest tests/tools/test_skills_tool.py -q
python -m pytest tests/test_plugin_skills.py -q
```

Expected: PASS with no content present in compact receipts and no cross-session collision.

- [ ] **Step 9: Run combined tool regressions**

Run:

```bash
python -m pytest \
  tests/tools/test_skills_tool.py \
  tests/test_plugin_skills.py \
  tests/tools/test_skill_view_path_check.py \
  tests/tools/test_skill_view_traversal.py \
  tests/tools/test_skill_size_limits.py \
  tests/tools/test_delegate.py \
  -q
```

Expected: PASS.

- [ ] **Step 10: Commit Task 2**

```bash
git add tools/skills_tool.py tests/tools/test_skills_tool.py tests/test_plugin_skills.py
git commit -m "perf: deduplicate unchanged skill content"
```

### Task 3: Apply Live Delegation Policy and Verify End to End

**Files:**
- Modify: `/Users/atorres/.hermes/config.yaml`
- Mirror through: `~/hermes-autotrader/bin/hermes-backup.sh`

**Interfaces:**
- Consumes: approved source behavior from Tasks 1 and 2.
- Produces: saved live config with root MCP inheritance disabled and no blanket profile opt-ins.

- [ ] **Step 1: Record immutable pre-edit evidence**

Run a Python YAML probe that asserts:

```python
assert config["delegation"]["inherit_mcp_toolsets"] is True
assert true_profiles == {
    "file-explorer",
    "jira-auditor",
    "coder",
    "reviewer-codex",
    "reviewer-opus",
    "oracle",
    "git-surgeon",
    "git-surgeon-medium",
    "git-surgeon-high",
    "debugger",
    "researcher",
    "verify-php-standards",
    "verify-frontend-standards",
    "planner-codex",
    "planner-opus",
    "verify-behavioral",
    "verify-mobile-standards",
}
assert false_profiles == {"verify-pr-description", "worktree-preparer"}
```

Record SHA-256 of live config and backup-repo HEAD before editing.

- [ ] **Step 2: Apply anchored live config edits**

Use `hermes config edit` with a deterministic editor script. The script must:

1. Replace only the root `delegation.inherit_mcp_toolsets: true` with `false`.
2. Delete only the 17 named profile-level `inherit_mcp_toolsets: true` lines.
3. Preserve the two explicit `false` profile lines.
4. Preserve `portfolio-standup` without an override.
5. Refuse the edit unless all expected anchors occur exactly once.

Do not use `yaml.safe_dump()` because it would normalize unrelated formatting.

- [ ] **Step 3: Validate exact config semantics and diff**

Run:

```bash
hermes config check
```

Run a YAML probe asserting root false, zero profile true overrides, two unchanged profile false overrides, and `portfolio-standup` unset. Compare live config against its backup baseline and require only the intended 18-line policy diff.

- [ ] **Step 4: Run source verification**

Run:

```bash
python -m pytest \
  tests/tools/test_skills_tool.py \
  tests/test_plugin_skills.py \
  tests/tools/test_skill_view_path_check.py \
  tests/tools/test_skill_view_traversal.py \
  tests/tools/test_skill_size_limits.py \
  tests/tools/test_delegate.py \
  -q
python -m compileall -q tools hermes_cli
```

Expected: all tests PASS and compileall exits zero.

- [ ] **Step 5: Parent reviews final source and config diffs**

Inspect:

```bash
git diff origin/main...HEAD --stat
git diff origin/main...HEAD --check
git diff origin/main...HEAD -- tools/skills_tool.py tools/delegate_tool.py hermes_cli/config.py tests
```

Verify no unrelated source files, comments, names, or behavior changed. Inspect live config diff separately because it is not in this repository.

- [ ] **Step 6: Push maintained fork and report commit URLs**

Push normal `main` only after all checks pass:

```bash
git push origin main
```

Verify local `HEAD` equals `origin/main`. Report clickable GitHub commit URLs for every created commit.

- [ ] **Step 7: Mirror live Hermes config through the blessed backup path**

Run:

```bash
~/hermes-autotrader/bin/hermes-backup.sh -m "perf(config): make delegated MCP access explicit"
```

Verify backup repository `HEAD` equals `origin/main`, live config SHA-256 equals committed mirror SHA-256, and unrelated concurrent captured files are reported plainly.

- [ ] **Step 8: Restart and smoke live behavior**

Restart gateway because delegation config is process-start loaded. Then:

1. Start a constrained delegate with native file/search toolsets.
2. Verify child effective toolsets exclude `mcp-activix_lsp`, `mcp-exa`, and `mcp-figma`.
3. In one fresh session, call `skill_view` twice for the same unchanged skill.
4. Verify first result has full content and hash.
5. Verify second result has compact unchanged receipt without content.
6. Verify a second session receives full content.

If restart is deferred, mark status as restart-pending rather than complete.

- [ ] **Step 9: Establish seven-day measurement**

Use existing local telemetry and the new counters to compare:

- input tokens per delegated API call;
- `skill_view_dedupe_hits`;
- `skill_view_chars_avoided`;
- `skill_view_approx_tokens_avoided`;
- enabled delegated MCP schema tokens;
- API p50 and p90;
- compression frequency;
- provider-reported spend when available.

Do not claim dollar savings from local token counts alone.
