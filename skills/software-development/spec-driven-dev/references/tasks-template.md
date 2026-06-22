# Tasks Template (`tasks.md`)

`tasks.md` is the contract handed to `subagent-driven-development`. Each task
is a 2-5 minute slice with full TDD cycle, exact paths, exact commands,
exact expected output. If a slice is bigger than 5 minutes, split it.

## Sections

### 1. Header (required)

```markdown
# [Feature] Implementation Tasks

> Source spec: `.spec/<slug>/spec.md`
> Source plan: `.spec/<slug>/plan.md`

**Goal:** [one sentence from spec section 1]
**Estimated tasks:** N
**Execution skill:** `subagent-driven-development`
```

### 2. Task list

Each task uses this exact structure:

```markdown
### Task N: <verb-first descriptive name>

**Spec reference:** AC-1, AC-4
**Plan reference:** Plan section 3 "Module X"

**Files:**
- Create: `path/to/new_file.py`
- Modify: `path/to/existing.py:45-67`
- Test: `tests/path/test_file.py`

**Step 1 -- Write failing test**

[code block]

**Step 2 -- Run test, verify FAIL**

Run: `pytest tests/path/test_file.py::test_specific_behavior -v`
Expected: FAIL with "function not defined" or assertion mismatch.

**Step 3 -- Minimal implementation**

[code block]

**Step 4 -- Run test, verify PASS**

Run: `pytest tests/path/test_file.py::test_specific_behavior -v`
Expected: PASS.

**Step 5 -- Regression sweep**

Run: `pytest tests/ -q`
Expected: no new failures (baseline pass count + 1).

**Step 6 -- Commit**

```bash
git add <files>
git commit -m "<type>: <description>"
```
```

### 3. Task ordering principles

1. Tests first, code second. Every code-creating task starts with a failing test.
2. Independence where possible. Tasks should not require re-reading prior tasks.
3. Verify before commit. Never commit on red.
4. Reuse the `plan` skill bite-size rules. 2-5 minutes per task.

### 4. Definition of done

- [ ] All N tasks committed.
- [ ] `pytest tests/ -q` green.
- [ ] `git diff main --stat` shows only expected files.
- [ ] Spec acceptance criteria re-checked: all AC-N marked done in spec.md.

## Anti-patterns

- "Implement and test X" with no test code -- split into two tasks.
- Tasks that depend on a previous task uncommitted state -- merge them.
- "Verify it works" with no exact command -- replace with the actual line.
- Tasks over 5 minutes -- split.
