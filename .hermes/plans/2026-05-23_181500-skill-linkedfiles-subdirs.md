# Plan: Skill Linked Files Custom Subdirectory Discovery

## Goal

Enable skill authors to place files in **arbitrary subdirectories** (e.g. `steps/`, `phases/`, `checks/`) inside a skill folder and have them automatically discovered by `skill_view`'s `linked_files` response — removing the current whitelist restriction to only 4 hardcoded directories.

## Current Context

- **`skill_manager_tool.py:171`**: `ALLOWED_SUBDIRS = {"references", "templates", "scripts", "assets"}` — used for two purposes:
  1. `_validate_file_path()` (line 315): rejects any write/remove targeting a subdirectory outside this set
  2. `_remove_file()` error path (line 684): lists existing files by iterating these 4 subdirs only

- **`skills_tool.py:1226-1268`**: `skill_view()` builds `linked_files` by hardcoding lookups for exactly 4 subdirectory names (`references/`, `templates/`, `assets/`, `scripts/`). Any other subdir (e.g. `steps/`) is invisible.

- **Test patterns**:
  - `tests/tools/test_skill_manager_tool.py` — `tmp_path` + `patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path)` pattern. Tests for `_validate_file_path` check `ALLOWED_SUBDIRS` membership.
  - `tests/tools/test_skills_tool.py` — `test_view_shows_linked_files()` (line 467) creates a `references/` dir and asserts `linked_files` contains it.

## Proposed Approach

**Options analysis (from issue):**
1. *Auto-discover all subdirectories* — scan the skill directory, include any subdir containing `.md` or `.py` files. No frontmatter changes needed.
2. *Configurable via frontmatter* — skills declare `metadata.hermes.subdirs: [steps, checks]` in SKILL.md.

**Recommendation: Approach 1 (auto-discover).** Simpler, backward-compatible, no protocol change. The only concern — write validation — should still enforce a baseline allowlist for write safety. But for *reading*/discovery, scanning the filesystem is safe.

## Step-by-Step Plan

### Step 1: Add `_discover_skill_subdirs(skill_dir)` helper in `skill_manager_tool.py`

**File:** `tools/skill_manager_tool.py`
**Location:** After the `ALLOWED_SUBDIRS` constant (line 171)

```python
def _discover_skill_subdirs(skill_dir: Path) -> set[str]:
    """Scan a skill directory and return names of subdirectories that
    contain markdown, Python, or script files — for linked_files discovery.
    
    This is used for read-side discovery only. Write validation still
    uses ALLOWED_SUBDIRS for safety.
    """
    seen: set[str] = set()
    for entry in skill_dir.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        # Only include directories that have discoverable content
        for f in entry.rglob("*"):
            if f.is_file() and f.suffix in {".md", ".py", ".sh", ".bash", ".yaml", ".yml", ".json", ".js", ".ts", ".rb", ".tex", ".txt"}:
                seen.add(entry.name)
                break
    return seen
```

### Step 2: Update `_remove_file()` error path in `skill_manager_tool.py`

**File:** `tools/skill_manager_tool.py`
**Location:** Lines 682–689 — the "File not found" error branch that lists available files

Replace the loop iterating `ALLOWED_SUBDIRS` with a call to `_discover_skill_subdirs()`:

```python
if not target.exists():
    available = []
    found_subdirs = _discover_skill_subdirs(skill_dir)
    for subdir in found_subdirs | ALLOWED_SUBDIRS:  # union guarantees baseline
        d = skill_dir / subdir
        if d.exists():
            for f in d.rglob("*"):
                if f.is_file():
                    available.append(str(f.relative_to(skill_dir)))
```

### Step 3: Replace hardcoded subdir lookups in `skills_tool.py:skill_view()` with auto-discovery

**File:** `tools/skills_tool.py`
**Location:** Lines 1225–1268 and 1282–1291

Replace the 4 hardcoded `skill_dir / "references"` etc. blocks with a single auto-discovery loop:

```python
if skill_dir:
    for entry in sorted(skill_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        subdir_name = entry.name
        if subdir_name in ("references", "templates", "assets", "scripts"):
            # Keep existing behaviour for known dirs (globs by extension)
            ... existing code ...
        else:
            # Auto-discover: include all files in any other subdir
            files = []
            for f in entry.rglob("*"):
                if f.is_file():
                    files.append(str(f.relative_to(skill_dir)))
            if files:
                linked_files[subdir_name] = files
```

**Cleaner refactor:** Extract the subdir-to-file-list logic into a helper function `_get_files_in_subdir(skill_dir, subdir_name)` shared by both the known-4 and the auto-discover branches. This avoids duplicating the `rglob` pattern.

### Step 4: Update `_validate_file_path()` to accept any real subdirectory (not just `ALLOWED_SUBDIRS`)

**File:** `tools/skill_manager_tool.py`
**Location:** Lines 314–317

Currently rejects paths not in `ALLOWED_SUBDIRS`. Change to also accept any subdirectory that exists in the skill's target dir:

```python
# In _validate_file_path, we now need context about WHICH skill dir
# this path refers to. The function currently takes (file_path) only.
# Option A: Change signature to _validate_file_path(file_path, skill_dir=None)
# Option B: Keep ALLOWED_SUBDIRS as the write-time guard (simpler, safe)
```

**Recommendation: Option B** — keep `ALLOWED_SUBDIRS` as the exclusive set for `_validate_file_path()` and `write_file()` / `remove_file()`. The auto-discovery is read-only. This avoids changing the validation signature and keeps write operations sandboxed to the 4 known-safe dirs. User can still place files in custom subdirs manually if needed.

> **Risk tradeoff:** This means `skill_manage(action='write_file')` and `skill_manage(action='remove_file')` won't be able to target custom subdirectories like `steps/` even after discovery. If the user needs programmatic write access to custom subdirs, revisit with Option A in a follow-up issue.

### Step 5: Add tests

**File:** `tests/tools/test_skills_tool.py`

Add test cases to `test_view_shows_linked_files` or create a new test class:

```python
def test_view_discovers_custom_subdirectory(self, tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        skill_dir = _make_skill(tmp_path, "multi-phase")
        steps_dir = skill_dir / "steps"
        steps_dir.mkdir()
        (steps_dir / "step-1.md").write_text("# Phase 1")
        (steps_dir / "step-2.md").write_text("# Phase 2")
        raw = skill_view("multi-phase")
    result = json.loads(raw)
    assert result["linked_files"] is not None
    assert "steps" in result["linked_files"]
    assert "steps/step-1.md" in result["linked_files"]["steps"]
    assert "steps/step-2.md" in result["linked_files"]["steps"]
```

```python
def test_view_hides_empty_subdirectory(self, tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        skill_dir = _make_skill(tmp_path, "empty-steps")
        (skill_dir / "steps").mkdir()  # empty dir
        raw = skill_view("empty-steps")
    result = json.loads(raw)
    assert "steps" not in (result.get("linked_files") or {})
```

**File:** `tests/tools/test_skill_manager_tool.py`

```python
def test_discover_subdirs_finds_custom_dirs(self, tmp_path):
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "steps").mkdir()
    (skill_dir / "steps" / "phase1.md").write_text("...")
    (skill_dir / "checks").mkdir()
    (skill_dir / "checks" / "preflight.py").write_text("...")
    result = _discover_skill_subdirs(skill_dir)
    assert "steps" in result
    assert "checks" in result
    assert "references" not in result  # doesn't exist yet
```

### Step 6: Verify

1. Run existing tests: `python -m pytest tests/tools/test_skill_manager_tool.py tests/tools/test_skills_tool.py -q`
2. Run full suite: `python -m pytest tests/ -o 'addopts=' -q`
3. Manual smoke test: create a skill with `steps/`, call `skill_view("my-skill")` and confirm `linked_files` includes the custom subdir

## Files Changed

| File | Change |
|------|--------|
| `tools/skill_manager_tool.py` | Add `_discover_skill_subdirs()` helper; update `_remove_file()` error path to use it |
| `tools/skills_tool.py` | Replace 4 hardcoded subdir lookups with auto-discovery + fallback to known dirs |
| `tests/tools/test_skills_tool.py` | Add tests: custom subdir visible, empty subdir hidden |
| `tests/tools/test_skill_manager_tool.py` | Add tests: `_discover_skill_subdirs` finds custom dirs |

## Tests / Validation

- **Unit:** 3 new tests (above)
- **Existing guard:** 0 existing tests break — `ALLOWED_SUBDIRS` is unchanged for write validation
- **Integration:** Manual skill creation + `skill_view` check

## Risks, Tradeoffs & Open Questions

1. **Write vs read asymmetry:** `write_file`/`remove_file` still restricted to the 4 known dirs. Auto-discovery is read-only. If a user needs programmatic write to `steps/`, this needs a follow-up. Mitigation: works fine for skills created manually or git-cloned with custom subdirs.
2. **Hidden/dot directories:** Excluded by `entry.name.startswith(".")` — `.git/`, `.hermes/` etc. never appear in linked_files.
3. **Empty directories:** Excluded by the "found files" check — an empty `steps/` won't appear.
4. **Performance:** `rglob("*")` on every `skill_view()` call. For skills with large subdirs (hundreds of files), this adds latency. Consider caching if profiling shows it's a bottleneck. For the common case (10-50 files), it's fine.
5. **Edge case — symlink loops:** `entry.rglob("*")` follows symlinks. A symlink back to the skill root creates an infinite loop. Mitigation: skip symlink dirs with `entry.is_symlink()` check, or use `entry.glob("**/*")` + `f.is_file()` guards.