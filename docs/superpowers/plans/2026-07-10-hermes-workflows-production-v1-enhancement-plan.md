# Hermes Workflows Production v1 Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an AI-first, production-quality Hermes Workflows v1 that closes every 2026-07-10 audit finding and makes authoring, publishing, running, feed operations, and execution history trustworthy from the Dashboard.

**Architecture:** Keep the current typed workflow spec, SQLite store, dispatcher, CLI/model tools, Dashboard plugin API, React, and React Flow. Add only the missing lifecycle contracts and UI modes; move the hand-authored plugin bundle into source files built with the repository's existing Vite/Vitest toolchain. Preserve `tick()` and deployed-definition compatibility while adding bounded APIs for drafts, publish conflicts, feed transitions, filtered history, and operator diagnostics.

**Tech Stack:** Python 3.13 via `uv`, Pydantic, SQLite/WAL, FastAPI, pytest, JavaScript, React 19 from the Dashboard plugin SDK, React Flow from the Dashboard plugin SDK, existing Vite 8/Vitest 4 workspace tooling, existing `agent-browser` CLI.

## Global Constraints

- Work in a fresh git worktree created from the branch containing this plan; do not edit the dirty primary checkout.
- TDD is mandatory: every behavior change starts with the smallest failing test and a recorded RED result.
- No new Python or npm dependencies. Reuse Vite, Vitest, jsdom, React, React Flow, and `agent-browser` already present in the repository.
- Keep Workflows a Dashboard plugin; do not move it into the main Dashboard application.
- Natural-language generation/refinement is primary; structured editing and the graph are review/correction surfaces.
- Every v1 runtime-supported field must be editable through structured UI. YAML/JSON is import, export, and diagnostics only.
- Published versions are immutable. Draft changes never mutate a published version, and drafts cannot run.
- Closed feeds are terminal. Starting again creates a distinct feed pinned to a published version.
- Batch splitting, document attachment persistence/splitting, explicit graph loops, repeated node visits, and in-execution input nodes remain unsupported and must fail validation clearly.
- Preserve existing CLI, model-tool, schedule, manual-run, and deployed-definition behavior unless a task explicitly changes its contract.
- Use the existing `redact_sensitive()`, `validate_graph()`, `require_implemented_primitives()`, `start_manual_execution()`, `deploy_definition()`, and Dashboard SDK auth helpers; do not duplicate them.
- Keep generated `plugins/workflows/dashboard/dist/` assets tracked, but edit authored files under `src/` after Task 1.
- Supported viewport gates: `1440x900`, `1280x576`, `1024x768`, `768x1024`, and `390x844`.
- Short-height gate: at `height <= 600px`, Build retains at least 240px of usable editor area.
- Each task ends with its focused tests, relevant regression suite, `git diff --check`, and one conventional commit.

---

## File Structure

### Existing backend files retained

- `hermes_cli/workflows_spec.py` — typed spec and graph validation.
- `hermes_cli/workflows_capabilities.py` — implemented primitive/capability truth.
- `hermes_cli/workflows_assistant.py` — AI draft/refine generation and repair.
- `hermes_cli/workflows_db.py` — definitions, drafts, executions, feeds, items, events.
- `hermes_cli/workflows_dispatcher.py` — schedule/feed admission and execution advancement.
- `hermes_cli/workflows.py` — CLI workflow commands.
- `tools/workflow_tools.py` — model-facing workflow tools.
- `plugins/workflows/dashboard/plugin_api.py` — Dashboard HTTP boundary.

### Frontend source introduced incrementally

- `plugins/workflows/dashboard/src/index.js` — SDK bootstrap and plugin registration.
- `plugins/workflows/dashboard/src/app.js` — top-level state orchestration; initially the migrated legacy app, then reduced as modes are extracted.
- `plugins/workflows/dashboard/src/api.js` — SDK-backed API calls and error normalization.
- `plugins/workflows/dashboard/src/graph.js` — graph view models and status decoration.
- `plugins/workflows/dashboard/src/workspace.js` — Build/Run/History shell.
- `plugins/workflows/dashboard/src/build.js` — AI authoring, structured editor, and publish flow.
- `plugins/workflows/dashboard/src/run.js` — single-run and continuous-feed operator UI.
- `plugins/workflows/dashboard/src/history.js` — execution list/detail operations.
- `plugins/workflows/dashboard/src/style.css` — authored plugin styles.
- `plugins/workflows/dashboard/vite.config.js` — deterministic IIFE/CSS build.
- `plugins/workflows/dashboard/vitest.config.js` — focused source tests using existing Vitest/jsdom.

Do not create empty directories or one-component files up front. Create each source module only in the task that gives it real behavior and tests.

---

### Task 0: Restore a Green, Reproducible Baseline

**Files:**
- Modify: `tests/hermes_cli/test_workflows_e2e.py:110`
- Create: `tests/plugins/test_workflows_audit_regressions.py`

**Interfaces:**
- Consumes: current Kanban provenance format from `hermes_cli/workflows_dispatcher.py`.
- Produces: a green focused suite and executable regression inventory for the two High audit defects.

- [ ] **Step 1: Update only the stale provenance assertion**

Replace the old assertion with the current production contract:

```python
assert task.created_by == f"workflow:{exec_id}:version:1:node:implement"
```

Do not change dispatcher production code; `tests/hermes_cli/test_workflows_dispatcher.py` already confirms the new contract.

- [ ] **Step 2: Run the previously failing E2E test**

Run:

```bash
uv run --extra dev python -m pytest tests/hermes_cli/test_workflows_e2e.py -q
```

Expected: `2 passed`.

- [ ] **Step 3: Add audit-regression inventory tests**

Create tests that pin the defects until source-level tests replace them:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JS = ROOT / "plugins/workflows/dashboard/dist/index.js"
CSS = ROOT / "plugins/workflows/dashboard/dist/style.css"


def test_trigger_adapter_cannot_be_overwritten_by_persisted_subtype():
    text = JS.read_text(encoding="utf-8")
    assert "Object.assign({}, trigger || {}, {" in text
    assert 'type: "trigger"' in text
    assert "trigger_type:" in text


def test_feed_panel_cannot_share_the_fixed_editor_column():
    text = JS.read_text(encoding="utf-8")
    body = text[text.index('className: "hermes-workflows-body"'):]
    assert body.index("renderInputFeedPanel()") < body.index("renderBottomPanel()")
    css = CSS.read_text(encoding="utf-8")
    assert ".hermes-workflows-run-mode" in css
```

The tests must fail against the current bundle. They are temporary contract tests and are deleted in Tasks 2 and 4 after stronger source tests exist.

- [ ] **Step 4: Run RED and record both failures**

Run:

```bash
uv run --extra dev python -m pytest tests/plugins/test_workflows_audit_regressions.py -q
```

Expected: two failures, one for trigger adaptation and one for feed placement.

- [ ] **Step 5: Run the full focused baseline**

Run:

```bash
uv run --extra dev python -m pytest \
  tests/hermes_cli/test_workflows_spec.py \
  tests/hermes_cli/test_workflows_intake.py \
  tests/hermes_cli/test_workflows_db.py \
  tests/hermes_cli/test_workflows_dispatcher.py \
  tests/hermes_cli/test_workflows_e2e.py \
  tests/tools/test_workflow_tools.py \
  tests/plugins/test_workflows_dashboard_plugin.py \
  tests/plugins/test_workflows_dashboard_assets.py -q
```

Expected: all existing tests pass; only the two intentionally new audit-regression tests remain red when run separately.

- [ ] **Step 6: Commit**

```bash
git add tests/hermes_cli/test_workflows_e2e.py tests/plugins/test_workflows_audit_regressions.py
git commit -m "test: pin workflow audit regressions"
```

---

### Task 1: Make the Dashboard Plugin Source-First and Reproducible

**Files:**
- Modify: `web/package.json`
- Create: `plugins/workflows/dashboard/vite.config.js`
- Create: `plugins/workflows/dashboard/vitest.config.js`
- Create: `plugins/workflows/dashboard/eslint.config.js`
- Create: `plugins/workflows/dashboard/src/index.js`
- Rename: `plugins/workflows/dashboard/dist/index.js` -> `plugins/workflows/dashboard/src/app.js`
- Rename: `plugins/workflows/dashboard/dist/style.css` -> `plugins/workflows/dashboard/src/style.css`
- Regenerate: `plugins/workflows/dashboard/dist/index.js`
- Regenerate: `plugins/workflows/dashboard/dist/style.css`
- Create: `tests/plugins/test_workflows_dashboard_build.py`

**Interfaces:**
- Consumes: `window.__HERMES_PLUGIN_SDK__`, `window.__HERMES_PLUGINS__`, existing React/React Flow globals.
- Produces: `npm run build:workflows --workspace web`, `npm run test:workflows --workspace web`, `npm run lint:workflows --workspace web`, deterministic `dist/index.js` and `dist/style.css`.

- [ ] **Step 1: Add a failing deterministic-build test**

```python
from pathlib import Path
import os
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "plugins" / "workflows" / "dashboard"


def test_workflows_plugin_build_matches_tracked_assets(tmp_path):
    env = os.environ.copy()
    env["WORKFLOWS_PLUGIN_OUT_DIR"] = str(tmp_path)
    subprocess.run(
        ["npm", "run", "build:workflows", "--workspace", "web"],
        cwd=ROOT,
        env=env,
        check=True,
    )
    assert (tmp_path / "index.js").read_bytes() == (PLUGIN / "dist/index.js").read_bytes()
    assert (tmp_path / "style.css").read_bytes() == (PLUGIN / "dist/style.css").read_bytes()
```

- [ ] **Step 2: Run RED**

Run:

```bash
uv run --extra dev python -m pytest tests/plugins/test_workflows_dashboard_build.py -q
```

Expected: failure because `build:workflows` does not exist.

- [ ] **Step 3: Add existing-tooling scripts**

Add to `web/package.json` scripts:

```json
"build:workflows": "vite build --config ../plugins/workflows/dashboard/vite.config.js",
"test:workflows": "vitest run --config ../plugins/workflows/dashboard/vitest.config.js",
"lint:workflows": "eslint ../plugins/workflows/dashboard/src --config ../plugins/workflows/dashboard/eslint.config.js"
```

Do not add dependencies.

- [ ] **Step 4: Move authored assets and add the bootstrap**

Move the existing JS and CSS without behavioral edits. Add `src/index.js`:

```javascript
import "./style.css";
import "./app.js";
```

The existing `app.js` remains its current self-invoking SDK bootstrap for this task. Modular extraction starts in Task 2.

- [ ] **Step 5: Add deterministic Vite config**

```javascript
import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.dirname(fileURLToPath(import.meta.url));
const outDir = process.env.WORKFLOWS_PLUGIN_OUT_DIR || path.join(root, "dist");

export default defineConfig({
  root,
  build: {
    emptyOutDir: true,
    outDir,
    cssCodeSplit: false,
    minify: false,
    lib: {
      entry: path.join(root, "src/index.js"),
      formats: ["iife"],
      name: "HermesWorkflowsPlugin",
      fileName: () => "index.js",
    },
    rollupOptions: {
      output: { assetFileNames: "style.css" },
    },
  },
});
```

- [ ] **Step 6: Add focused Vitest config**

```javascript
import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root,
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.js"],
  },
});
```

Add `eslint.config.js` using the already-installed ESLint packages:

```javascript
import js from "@eslint/js";
import globals from "globals";

export default [
  {
    files: ["src/**/*.js"],
    ...js.configs.recommended,
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      globals: globals.browser,
    },
  },
];
```

- [ ] **Step 7: Build twice and verify deterministic bytes**

Run:

```bash
npm run build:workflows --workspace web
shasum -a 256 plugins/workflows/dashboard/dist/index.js plugins/workflows/dashboard/dist/style.css > /tmp/workflows-build-1.sha
npm run build:workflows --workspace web
shasum -a 256 plugins/workflows/dashboard/dist/index.js plugins/workflows/dashboard/dist/style.css > /tmp/workflows-build-2.sha
diff -u /tmp/workflows-build-1.sha /tmp/workflows-build-2.sha
npm run lint:workflows --workspace web
uv run --extra dev python -m pytest tests/plugins/test_workflows_dashboard_build.py -q
```

Expected: no hash diff; build test passes.

- [ ] **Step 8: Run plugin regressions and commit**

```bash
uv run --extra dev python -m pytest tests/plugins/test_workflows_dashboard_assets.py tests/plugins/test_workflows_dashboard_plugin.py -q
git diff --check
git add web/package.json plugins/workflows/dashboard tests/plugins/test_workflows_dashboard_build.py
git commit -m "build: add workflows plugin source pipeline"
```

---

### Task 2: Fix Graph Identity, Status Decoration, and API Error Copy

**Files:**
- Create: `plugins/workflows/dashboard/src/graph.js`
- Create: `plugins/workflows/dashboard/src/graph.test.js`
- Create: `plugins/workflows/dashboard/src/api.js`
- Create: `plugins/workflows/dashboard/src/api.test.js`
- Modify: `plugins/workflows/dashboard/src/app.js`
- Delete: `tests/plugins/test_workflows_audit_regressions.py::test_trigger_adapter_cannot_be_overwritten_by_persisted_subtype`
- Regenerate: `plugins/workflows/dashboard/dist/index.js`

**Interfaces:**
- Produces: `graphItems(spec) -> Array<GraphItem>`, `decorateGraphItems(items, statuses) -> Array<GraphItem>`, `formatApiError(error) -> string`, `createApi(fetchJSON, basePath) -> function`.
- `GraphItem` fields: `{ id, rendererType, specKind, triggerType, spec }`.

- [ ] **Step 1: Write graph RED tests**

```javascript
import { describe, expect, it } from "vitest";
import { decorateGraphItems, graphItems } from "./graph.js";

const spec = {
  triggers: [{ id: "manual", type: "manual" }],
  nodes: { pass: { type: "pass" } },
};

describe("graphItems", () => {
  it("keeps renderer identity separate from trigger subtype", () => {
    expect(graphItems(spec)[0]).toMatchObject({
      id: "manual",
      rendererType: "trigger",
      specKind: "trigger",
      triggerType: "manual",
    });
  });

  it("preserves membership when statuses change", () => {
    const before = graphItems(spec);
    const after = decorateGraphItems(before, { pass: "succeeded" });
    expect(after.map((item) => item.id)).toEqual(["manual", "pass"]);
    expect(after.find((item) => item.id === "pass").status).toBe("succeeded");
  });
});
```

- [ ] **Step 2: Write error-copy RED tests**

```javascript
import { expect, it } from "vitest";
import { formatApiError } from "./api.js";

it("unwraps FastAPI detail from Error.message", () => {
  expect(formatApiError(new Error('400: {"detail":"repo_path is required"}')))
    .toBe("repo_path is required");
});

it("prefers structured field error messages", () => {
  expect(formatApiError({
    code: "workflow_input_invalid",
    message: "Repository path is required.",
  })).toBe("Repository path is required.");
});
```

- [ ] **Step 3: Run RED**

```bash
npm run test:workflows --workspace web
```

Expected: module-not-found failures.

- [ ] **Step 4: Implement the pure graph adapter**

```javascript
export function graphItems(spec) {
  const triggers = (spec?.triggers || []).map((trigger, index) => ({
    id: trigger.id || trigger.name || `trigger_${index + 1}`,
    rendererType: "trigger",
    specKind: "trigger",
    triggerType: trigger.type,
    spec: trigger,
  }));
  const nodes = Object.entries(spec?.nodes || {}).map(([id, node]) => ({
    id,
    rendererType: node.type,
    specKind: "node",
    triggerType: null,
    spec: node,
  }));
  return [...triggers, ...nodes];
}

export function decorateGraphItems(items, statuses) {
  return items.map((item) => ({ ...item, status: statuses[item.id] || "idle" }));
}
```

Use these functions from `app.js`; remove the old `triggerList()`/node membership path instead of keeping two implementations.

- [ ] **Step 5: Implement recursive API error normalization**

`formatApiError()` must normalize `Error.message` before object handling, parse a JSON suffix when present, and return `message`, string `detail`, or a safe fallback. Keep API calls on `SDK.fetchJSON` through `createApi()`; do not use raw `fetch`.

- [ ] **Step 6: Preserve viewport on status-only updates**

In `app.js`, derive graph membership from `graphItems(spec)` and apply execution status with `decorateGraphItems()`. Call fit-view only when the ordered membership key changes:

```javascript
const membershipKey = items.map((item) => `${item.specKind}:${item.id}`).join("|");
```

Status polling must not call `fitView()` or reset `nodePositions` when `membershipKey` is unchanged.

- [ ] **Step 7: Run GREEN and rebuild**

```bash
npm run test:workflows --workspace web
npm run build:workflows --workspace web
uv run --extra dev python -m pytest \
  tests/plugins/test_workflows_dashboard_assets.py \
  tests/plugins/test_workflows_dashboard_build.py -q
```

Expected: all pass.

- [ ] **Step 8: Replace the temporary trigger regression and commit**

Delete only the superseded trigger test from `test_workflows_audit_regressions.py`; keep the layout RED test for Task 4.

```bash
git diff --check
git add plugins/workflows/dashboard tests/plugins
git commit -m "fix: preserve workflow graph identity"
```

---

### Task 3: Add Durable Drafts, Immutable Publish Conflicts, Archive, and Terminal Feed State

**Files:**
- Modify: `hermes_cli/workflows_db.py`
- Modify: `plugins/workflows/dashboard/plugin_api.py`
- Modify: `tests/hermes_cli/test_workflows_db.py`
- Modify: `tests/plugins/test_workflows_dashboard_plugin.py`
- Modify: `tests/hermes_cli/test_workflows_e2e.py`

**Interfaces:**
- Produces DB functions:
  - `save_draft(conn, spec, *, base_version, updated_at=None) -> WorkflowDraftRecord`
  - `get_draft(conn, workflow_id) -> WorkflowDraftRecord`
  - `delete_draft(conn, workflow_id) -> bool`
  - `list_workflow_summaries(conn, *, include_archived=False) -> list[dict[str, Any]]`
  - `set_definition_enabled(conn, workflow_id, enabled, *, version=None) -> WorkflowDefinitionRecord` (existing, reused)
  - `set_workflow_archived(conn, workflow_id, archived) -> None`
  - `delete_definition(conn, workflow_id, *, purge=False) -> bool`
  - `publish_draft(conn, workflow_id, *, expected_latest_version, created_by) -> WorkflowDefinitionRecord`
- Produces API routes:
  - `GET /workflows?include_archived=false`
  - `PUT /definitions/{workflow_id}/draft`
  - `GET /definitions/{workflow_id}/draft`
  - `DELETE /definitions/{workflow_id}/draft`
  - `POST /definitions/{workflow_id}/publish`
  - `POST /definitions/{workflow_id}/enabled`
  - `POST /definitions/{workflow_id}/archive`
  - `DELETE /definitions/{workflow_id}?purge=false`
- Feed transitions: open -> paused/closed; paused -> open/closed; closed -> none.

- [ ] **Step 1: Write DB RED tests for drafts and publish conflicts**

```python
def test_draft_publish_is_immutable_and_conflict_checked(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    spec = _demo_spec().model_copy(update={"version": 1})
    with wfdb.connect() as conn:
        wfdb.save_draft(conn, spec, base_version=None)
        published = wfdb.publish_draft(
            conn, spec.id, expected_latest_version=None, created_by="test"
        )
        assert published.version == 1
        changed = spec.model_copy(update={"name": "Changed"})
        wfdb.save_draft(conn, changed, base_version=1)
        with pytest.raises(wfdb.WorkflowVersionConflict):
            wfdb.publish_draft(
                conn, spec.id, expected_latest_version=0, created_by="test"
            )
        assert wfdb.get_definition(conn, spec.id, 1).name == spec.name
```

- [ ] **Step 2: Write DB RED tests for feed transitions and write gates**

```python
def test_closed_feed_is_terminal_and_rejects_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _continuous_demo_spec(), created_by="test")
        feed = wfdb.open_input_feed(conn, "demo", trigger_id="manual")
        wfdb.set_input_feed_status(conn, feed.feed_id, "closed")
        with pytest.raises(ValueError, match="closed feed cannot transition"):
            wfdb.set_input_feed_status(conn, feed.feed_id, "open")
        with pytest.raises(ValueError, match="feed is closed"):
            wfdb.enqueue_input_item(conn, feed.feed_id, {"repo_path": "/repo"})


def test_paused_feed_rejects_item_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _continuous_demo_spec(), created_by="test")
        feed = wfdb.open_input_feed(conn, "demo", trigger_id="manual")
        wfdb.set_input_feed_status(conn, feed.feed_id, "paused")
        with pytest.raises(ValueError, match="feed is paused"):
            wfdb.enqueue_input_item(conn, feed.feed_id, {"repo_path": "/repo"})
```

- [ ] **Step 3: Write API RED tests**

Test saving/reloading a draft, publish conflict returning HTTP 409 with code `workflow_version_conflict`, archive hiding from `GET /workflows` by default, `include_archived=true` restoring it, enabled/disabled state in the workflow summary, paused/closed writes returning HTTP 409, and closed -> open returning HTTP 409.

Add one destructive-delete test:

```python
def test_dashboard_delete_requires_explicit_purge_when_history_exists(client):
    definition = _deploy(client, PASS_SPEC)
    client.post(
        f"/api/plugins/workflows/definitions/{definition['workflow_id']}/run",
        json={"input": {}},
    )
    blocked = client.delete(
        f"/api/plugins/workflows/definitions/{definition['workflow_id']}"
    )
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "workflow_history_exists"
    purged = client.delete(
        f"/api/plugins/workflows/definitions/{definition['workflow_id']}?purge=true"
    )
    assert purged.status_code == 200
```

`GET /workflows` summary assertions must cover `has_draft`, `latest_version`, `enabled`, `archived`, `latest_execution_status`, and `open_feed_count`.

Use exact envelope assertions:

```python
assert response.json() == {
    "detail": {
        "code": "workflow_version_conflict",
        "message": "Workflow changed since this draft was created.",
        "field_errors": {},
        "hint": "Reload the latest version and review the draft again.",
    }
}
```

- [ ] **Step 4: Run RED**

```bash
uv run --extra dev python -m pytest \
  tests/hermes_cli/test_workflows_db.py \
  tests/plugins/test_workflows_dashboard_plugin.py -q
```

Expected: failures for missing draft table/functions/routes and permissive feed transitions.

- [ ] **Step 5: Add the minimal schema migration**

Add `workflow_drafts` and `archived` without a new migration framework:

```sql
CREATE TABLE IF NOT EXISTS workflow_drafts (
    workflow_id TEXT PRIMARY KEY,
    spec_json TEXT NOT NULL,
    base_version INTEGER,
    updated_at INTEGER NOT NULL
);
```

Add `archived INTEGER NOT NULL DEFAULT 0` to `workflow_definitions` through the existing `init_db()` column-upgrade pattern. Archive updates all versions for the workflow.

- [ ] **Step 6: Implement one publish path**

`publish_draft()` must:

1. load and validate the draft spec;
2. compute the current latest version;
3. compare it with `expected_latest_version` using equality, including `None` for first publish;
4. set draft spec version to `latest + 1` or `1`;
5. call existing `deploy_definition(..., auto_bump=False)`;
6. delete the draft only after the definition commits;
7. return the persisted definition.

Do not add a second checksum/versioning implementation.

- [ ] **Step 7: Enforce feed state at the DB boundary**

Add one private guard used by `enqueue_input_item()` and `update_input_item()`:

```python
def _require_open_feed(feed: WorkflowInputFeed) -> None:
    if feed.status != "open":
        raise ValueError(f"workflow input feed is {feed.status}: {feed.feed_id}")
```

`set_input_feed_status()` validates the transition map before update. A no-op transition returns the current feed idempotently; closed -> any different state fails.

- [ ] **Step 8: Add stable API error helpers and routes**

Keep FastAPI's existing `detail` wrapper, but every workflow API error detail is:

```python
{
    "code": code,
    "message": redact_sensitive(message),
    "field_errors": field_errors,
    "hint": hint,
}
```

Map version/state/history conflicts to 409, missing resources to 404, and validation to 400. Existing redaction remains mandatory.

`list_workflow_summaries()` is one grouped SQL/read pass plus current draft rows; do not issue per-workflow queries. `delete_definition(..., purge=False)` deletes draft-only workflows and workflows with no execution/feed history. When history exists it raises a stable conflict; `purge=True` uses the existing cascading delete transaction. Enable/disable delegates to existing `set_definition_enabled()`.

- [ ] **Step 9: Update old feed tests to the approved lifecycle**

Replace any test that enqueues while paused with:

1. enqueue while open;
2. pause;
3. prove no further writes/admission;
4. resume;
5. prove admission resumes;
6. close;
7. prove terminal behavior;
8. open a new feed and prove it has a different `feed_id`.

- [ ] **Step 10: Run GREEN and commit**

```bash
uv run --extra dev python -m pytest \
  tests/hermes_cli/test_workflows_db.py \
  tests/hermes_cli/test_workflows_dispatcher.py \
  tests/hermes_cli/test_workflows_e2e.py \
  tests/plugins/test_workflows_dashboard_plugin.py -q
git diff --check
git add hermes_cli/workflows_db.py plugins/workflows/dashboard/plugin_api.py tests/hermes_cli tests/plugins/test_workflows_dashboard_plugin.py
git commit -m "feat: add workflow draft and feed lifecycle"
```

---

### Task 4: Introduce Build, Run, and History Workspace Modes

**Files:**
- Create: `plugins/workflows/dashboard/src/workspace.js`
- Create: `plugins/workflows/dashboard/src/workspace.test.js`
- Modify: `plugins/workflows/dashboard/src/app.js`
- Modify: `plugins/workflows/dashboard/src/style.css`
- Delete: `tests/plugins/test_workflows_audit_regressions.py`
- Regenerate: `plugins/workflows/dashboard/dist/index.js`
- Regenerate: `plugins/workflows/dashboard/dist/style.css`

**Interfaces:**
- Produces: `WORKSPACE_MODES`, `modeForLocation(location)`, `locationForMode(workflowId, mode, selection)`.
- Build owns graph/editor; Run owns run/feed operations; History owns execution list/detail.

- [ ] **Step 1: Write workspace routing RED tests**

```javascript
import { describe, expect, it } from "vitest";
import { locationForMode, modeForLocation } from "./workspace.js";

it("defaults a workflow workspace to build", () => {
  expect(modeForLocation({ pathname: "/workflows/demo", search: "" })).toBe("build");
});

it("round-trips run and history selections", () => {
  expect(locationForMode("demo", "run", { feed: "wffeed_1" }))
    .toBe("/workflows/demo/run?feed=wffeed_1");
  expect(locationForMode("demo", "history", { execution: "wfexec_1" }))
    .toBe("/workflows/demo/history?execution=wfexec_1");
});
```

- [ ] **Step 2: Add jsdom RED tests for mode isolation and workflow navigation**

Render the workspace with simple child markers and assert only the active mode is present. The test must also assert native tab roles and `aria-selected`.

With a mocked workflow summary list, assert each row shows draft/published status, latest version, enabled/disabled state, latest execution status, and open feed count. Assert action visibility and payloads:

- **Duplicate** creates a new draft id and never copies published history.
- **Disable/Enable** calls the enabled route.
- **Archive/Restore** calls the archive route and removes/restores the row according to the current filter.
- **Delete** is available only without history; a 409 offers **Purge workflow and history** behind typed-name confirmation.
- A draft-only workflow opens Build and has Run disabled.

- [ ] **Step 3: Run RED**

```bash
npm run test:workflows --workspace web
```

- [ ] **Step 4: Implement the smallest workspace shell and workflow navigation**

Use native buttons with `role="tab"` and a `role="tablist"`. Keep history state in the URL through `history.pushState`; listen for `popstate`. Do not add React Router to the plugin.

Load `GET /workflows` into the existing sidebar rather than creating a second navigation tree. Keep search and archived toggle local. Implement Duplicate by cloning the selected draft/published spec, clearing `version`, assigning a unique id/name, and saving through the existing draft endpoint. Reuse API lifecycle routes for Enable, Archive, Delete, and Purge; do not mutate summary state optimistically before the server succeeds.

- [ ] **Step 5: Move existing render paths without rewriting them**

- Build: existing sidebar, palette, canvas, inspector, validation.
- Run: existing run form and continuous feed panel.
- History: existing execution list/timeline/node-run detail.
- Diagnostics: Manual Tick and dispatcher status under Run's diagnostics disclosure.

Delete the old bottom Execution tab once History owns that content. Do not render feed or history siblings underneath Build.

- [ ] **Step 6: Add layout CSS and explicit viewport rules**

Use CSS grid/flex only. Required assertions in `workspace.test.js` or Python asset test:

```text
@media (max-width: 1279px)
@media (max-width: 767px)
@media (max-height: 600px)
min-height: 240px
```

At short height, `.hermes-workflows-build-mode` owns the 240px minimum; Run and History scroll independently.

- [ ] **Step 7: Run GREEN and rebuild**

```bash
npm run test:workflows --workspace web
npm run build:workflows --workspace web
uv run --extra dev python -m pytest \
  tests/plugins/test_workflows_dashboard_assets.py \
  tests/plugins/test_workflows_dashboard_build.py -q
```

- [ ] **Step 8: Delete the temporary layout regression and commit**

The source-level mode-isolation and CSS tests now supersede `test_workflows_audit_regressions.py`; delete the file.

```bash
git diff --check
git add plugins/workflows/dashboard tests/plugins
git commit -m "feat: split workflow build run and history modes"
```

---

### Task 5: Implement AI-First Draft Creation, Refinement Review, and Publish

**Files:**
- Create: `plugins/workflows/dashboard/src/build.js`
- Create: `plugins/workflows/dashboard/src/build.test.js`
- Modify: `plugins/workflows/dashboard/src/app.js`
- Modify: `plugins/workflows/dashboard/src/api.js`
- Modify: `hermes_cli/workflows_assistant.py`
- Modify: `plugins/workflows/dashboard/plugin_api.py`
- Modify: `tests/hermes_cli/test_workflows_assistant.py`
- Modify: `tests/plugins/test_workflows_dashboard_plugin.py`
- Create: `tests/fixtures/workflows/assistant_responses.json`

**Interfaces:**
- Produces: `semanticWorkflowDiff(before, after) -> ChangeSection[]`.
- `ChangeSection`: `{ kind, summary, items }` for metadata, triggers/input, nodes, routing, runtime.
- AI draft response remains `{ spec, summary, assumptions, warnings }`; unresolved user choices use `warnings` with stable text rather than a new protocol.

- [ ] **Step 1: Write semantic-diff RED tests**

```javascript
import { expect, it } from "vitest";
import { semanticWorkflowDiff } from "./build.js";

it("summarizes trigger node and routing changes", () => {
  const before = {
    triggers: [{ id: "manual", type: "manual" }],
    nodes: { review: { type: "pass" } },
    edges: [],
  };
  const after = {
    triggers: [{ id: "manual", type: "manual", input_schema: { repo: { kind: "repo_path" } } }],
    nodes: { review: { type: "agent_task", profile: "reviewer", prompt: "Review" } },
    edges: [{ from: "review", to: "done" }],
  };
  expect(semanticWorkflowDiff(before, after).map((section) => section.kind))
    .toEqual(["triggers", "nodes", "routing"]);
});
```

- [ ] **Step 2: Write UI RED tests for candidate acceptance**

With mocked `createApi()`, assert:

- Generate shows summary/assumptions and saves a draft only after **Accept draft**.
- Refine shows semantic changes and leaves the existing draft unchanged until **Accept changes**.
- Reject leaves the draft unchanged.
- Publish sends `expected_latest_version` and clears the draft only on success.
- A 409 conflict preserves the draft and displays Reload/Review guidance.

- [ ] **Step 3: Write assistant/API RED tests**

Add Python tests that AI draft/refine responses include `summary`, `assumptions`, and `warnings`, and that invalid candidate output remains a typed assistant validation error suitable for **Repair with AI**.

Create `tests/fixtures/workflows/assistant_responses.json` with exactly two complete API response objects, `draft` and `refine`. Python endpoint tests, JavaScript UI tests, and Task 9 browser route mocks must read this one fixture rather than embedding three versions of the same candidate.

- [ ] **Step 4: Run RED**

```bash
npm run test:workflows --workspace web
uv run --extra dev python -m pytest \
  tests/hermes_cli/test_workflows_assistant.py \
  tests/plugins/test_workflows_dashboard_plugin.py -q
```

- [ ] **Step 5: Implement semantic diff with plain objects**

Use `Object.keys`, `JSON.stringify`, and stable sorting. Do not add a diff package. Compare only product-meaningful groups and suppress unchanged sections.

- [ ] **Step 6: Implement candidate state and durable draft calls**

Keep four states only:

```javascript
{
  savedDraft,
  workingDraft,
  candidateDraft,
  candidateSource, // "generate" | "refine" | "repair"
}
```

`workingDraft` is dirty when it differs from `savedDraft`. Refresh never overwrites dirty state. Accept candidate sets `workingDraft` then saves through the draft API. Reject clears candidate only.

- [ ] **Step 7: Reuse existing AI backend and add publish conflict input**

Do not introduce a second assistant. Extend request/response models only enough to carry constraints/input/output examples if provided, while preserving goal-only requests. Publish delegates to Task 3's `publish_draft()`.

- [ ] **Step 8: Add local undo for accepted draft changes**

Use a bounded array of the last 20 accepted `workingDraft` values in component state. No undo framework and no persisted history.

- [ ] **Step 9: Run GREEN, rebuild, and commit**

```bash
npm run test:workflows --workspace web
npm run build:workflows --workspace web
uv run --extra dev python -m pytest \
  tests/hermes_cli/test_workflows_assistant.py \
  tests/plugins/test_workflows_dashboard_plugin.py \
  tests/plugins/test_workflows_dashboard_build.py -q
git diff --check
git add hermes_cli/workflows_assistant.py plugins/workflows/dashboard tests
git commit -m "feat: add ai-first workflow draft review"
```

---

### Task 6: Complete Structured Editing for the Supported v1 Schema

**Files:**
- Modify: `plugins/workflows/dashboard/src/build.js`
- Modify: `plugins/workflows/dashboard/src/app.js`
- Create: `plugins/workflows/dashboard/src/editor-model.js`
- Create: `plugins/workflows/dashboard/src/editor-model.test.js`
- Modify: `tests/plugins/test_workflows_dashboard_assets.py`

**Interfaces:**
- Produces: `editorSections(spec) -> EditorSection[]`, `changeNodeType(spec, nodeId, nextType) -> { spec, removedFields }`, `conditionFromForm(form)`, `resultContractFromRows(rows)`.
- Supported triggers: manual, schedule.
- Supported nodes: pass, switch, agent_task, wait, parallel, join, fail.

- [ ] **Step 1: Write coverage RED tests**

```javascript
import { expect, it } from "vitest";
import { supportedEditorCoverage } from "./editor-model.js";

it("covers every supported v1 primitive", () => {
  expect(supportedEditorCoverage()).toEqual({
    triggers: ["manual", "schedule"],
    nodes: ["agent_task", "fail", "join", "parallel", "pass", "switch", "wait"],
    intakeModes: ["continuous", "single"],
  });
});
```

Add parameterized tests proving each node type round-trips through its form model without dropping supported fields.

- [ ] **Step 2: Write node conversion RED tests**

```javascript
it("previews and removes fields incompatible with the next node type", () => {
  const spec = { nodes: { work: { type: "agent_task", profile: "worker", prompt: "Do it", result_contract: { ok: "boolean" } } } };
  const result = changeNodeType(spec, "work", "pass");
  expect(result.removedFields.sort()).toEqual(["profile", "prompt", "result_contract"]);
  expect(result.spec.nodes.work).toEqual({ type: "pass", output: {} });
});
```

- [ ] **Step 3: Write unsupported-scope RED tests**

The editor must not offer batch, document upload, split strategy, item source, webhook, kanban_event, send_message, subworkflow, or loop controls. Importing such a draft must show the existing backend validation error, not silently omit fields.

- [ ] **Step 4: Run RED**

```bash
npm run test:workflows --workspace web
```

- [ ] **Step 5: Extract only pure form conversions**

Move existing input-row/schema and trigger-intake conversion logic from `app.js` to `editor-model.js`. Add missing pure conversions for condition presets and result-contract rows. Reuse the existing inspector renderers in `app.js`/`build.js`; do not create a component file per node type.

- [ ] **Step 6: Add guided condition and result-contract forms**

Supported condition operations are exactly those accepted by `workflows_spec._validate_condition_context`. Paths are explicit and preview the generated condition object. Result-contract tokens are the existing primitive tokens or non-empty `a|b` enum strings.

- [ ] **Step 7: Add confirmed node-type conversion**

The visible Cell Type control calls `changeNodeType()`, shows removed fields, and commits only after confirmation. Use the Dashboard's existing confirmation pattern; do not add a modal library.

- [ ] **Step 8: Keep Advanced YAML round-trip only**

Give the textarea `aria-label="Workflow definition YAML"`. Import replaces the working draft only after parse/validate succeeds. Export serializes the current working draft. Publish and Run never consume raw textarea content.

- [ ] **Step 9: Run GREEN and complete an editor marker audit**

```bash
npm run test:workflows --workspace web
npm run build:workflows --workspace web
uv run --extra dev python -m pytest tests/plugins/test_workflows_dashboard_assets.py tests/plugins/test_workflows_dashboard_build.py -q
```

Replace old string-presence assertions with source-level coverage assertions where a source test now exists.

- [ ] **Step 10: Commit**

```bash
git diff --check
git add plugins/workflows/dashboard tests/plugins
git commit -m "feat: complete structured workflow editing"
```

---

### Task 7: Build Run Mode, Enforce Feed Operations, and Clarify Diagnostics

**Files:**
- Create: `plugins/workflows/dashboard/src/run.js`
- Create: `plugins/workflows/dashboard/src/run.test.js`
- Modify: `plugins/workflows/dashboard/src/app.js`
- Modify: `plugins/workflows/dashboard/src/api.js`
- Modify: `plugins/workflows/dashboard/plugin_api.py`
- Modify: `hermes_cli/workflows_dispatcher.py`
- Modify: `tests/hermes_cli/test_workflows_dispatcher.py`
- Modify: `tests/plugins/test_workflows_dashboard_plugin.py`

**Interfaces:**
- Produces: `feedActions(status) -> string[]`, `shouldPollFeed(items) -> bool`, `fieldErrors(error) -> object`.
- Produces dispatcher APIs: existing `tick(limit) -> int` unchanged; new `tick_detailed(limit) -> TickReport`.
- `TickReport`: schedules admitted, feed items admitted, executions advanced, remaining queued, remaining running/waiting.

- [ ] **Step 1: Write Run-mode pure RED tests**

```javascript
import { expect, it } from "vitest";
import { feedActions, shouldPollFeed } from "./run.js";

it("shows only valid feed actions", () => {
  expect(feedActions("open")).toEqual(["pause", "close"]);
  expect(feedActions("paused")).toEqual(["resume", "close"]);
  expect(feedActions("closed")).toEqual(["start-new"]);
});

it("polls only active feed items", () => {
  expect(shouldPollFeed([{ status: "queued" }])).toBe(true);
  expect(shouldPollFeed([{ status: "succeeded" }, { status: "failed" }])).toBe(false);
});
```

- [ ] **Step 2: Write API/dispatcher RED tests for detailed ticks**

```python
def test_tick_detailed_separates_admission_from_execution(tmp_path, monkeypatch):
    # Arrange one ready feed item and one older queued execution.
    report = workflows_dispatcher.tick_detailed(limit=2)
    assert report.executions_advanced == 1
    assert report.feed_items_admitted == 1
    assert report.processed == 2
```

Update Dashboard `/tick` test to expect:

```python
{
    "schedules_admitted": 0,
    "feed_items_admitted": 1,
    "executions_advanced": 1,
    "remaining_queued": 0,
    "remaining_running_or_waiting": 0,
    "processed": 2,
}
```

- [ ] **Step 3: Write UI RED tests**

Assert:

- Run is disabled for draft-only workflows.
- Version selector lists published versions.
- Missing required input keeps the form open and renders `field_errors` beside fields.
- Continuous workflow renders feeds instead of the single-run form.
- Closed feed renders **Start new feed**, not Resume.
- Manual Tick appears only inside Diagnostics.
- Active items poll; terminal-only lists stop.

- [ ] **Step 4: Run RED**

```bash
npm run test:workflows --workspace web
uv run --extra dev python -m pytest \
  tests/hermes_cli/test_workflows_dispatcher.py \
  tests/plugins/test_workflows_dashboard_plugin.py -q
```

- [ ] **Step 5: Add one shared detailed tick implementation**

Refactor dispatcher internals so `_tick(limit) -> TickReport` is the only implementation. Keep compatibility:

```python
def tick(*, limit: int = 1) -> int:
    return _tick(limit=limit).processed


def tick_detailed(*, limit: int = 1) -> TickReport:
    return _tick(limit=limit)
```

Do not change gateway callers. Preserve existing queued-execution fairness.

- [ ] **Step 6: Implement schema-driven single-run form**

Reuse trigger `input_schema`. Client validation is advisory; submit uses existing server `start_manual_execution()`. On 400, map `field_errors` and preserve values. On success, navigate to History for the new execution.

- [ ] **Step 7: Implement continuous-feed workspace**

Show feed version/state/timestamps, valid actions, counters, item composer, filterable item list, item validation, and linked execution. Enforce Task 3 state rules in controls and backend. **Start new feed** always POSTs a new feed and selects its new id.

- [ ] **Step 8: Implement bounded polling**

Use one `setTimeout` loop per selected feed/execution, 2-second delay, stopped on unmount, selection change, or terminal-only state. Avoid overlapping requests by scheduling after completion.

- [ ] **Step 9: Run GREEN, rebuild, and commit**

```bash
npm run test:workflows --workspace web
npm run build:workflows --workspace web
uv run --extra dev python -m pytest \
  tests/hermes_cli/test_workflows_dispatcher.py \
  tests/plugins/test_workflows_dashboard_plugin.py \
  tests/plugins/test_workflows_dashboard_build.py -q
git diff --check
git add hermes_cli/workflows_dispatcher.py plugins/workflows/dashboard tests
git commit -m "feat: add workflow run and feed operations"
```

---

### Task 8: Add Workflow-Filtered History, Detail, Cancel, and Rerun

**Files:**
- Create: `plugins/workflows/dashboard/src/history.js`
- Create: `plugins/workflows/dashboard/src/history.test.js`
- Modify: `plugins/workflows/dashboard/src/app.js`
- Modify: `plugins/workflows/dashboard/src/api.js`
- Modify: `plugins/workflows/dashboard/plugin_api.py`
- Modify: `hermes_cli/workflows_db.py`
- Modify: `tests/hermes_cli/test_workflows_db.py`
- Modify: `tests/plugins/test_workflows_dashboard_plugin.py`

**Interfaces:**
- Extends `list_executions()` with `status`, `version`, `trigger_id`, `before`, and `limit` filters.
- Produces `GET /executions/{execution_id}/detail` returning execution, definition summary, node runs, and events in one redacted response.
- Produces `POST /executions/{execution_id}/rerun` creating a new execution pinned to the same workflow version and trigger with explicit input.

- [ ] **Step 1: Write DB/API filtering RED tests**

Create executions across two workflows and assert current-workflow filtering is the default UI request. Add DB tests for combined status/version/trigger/before filters and stable newest-first pagination.

- [ ] **Step 2: Write detail/redaction RED tests**

```python
def test_execution_detail_is_versioned_and_redacted(client):
    # Deploy/run a workflow with secret-looking input.
    detail = client.get(f"/api/plugins/workflows/executions/{execution_id}/detail")
    assert detail.status_code == 200
    body = detail.json()
    assert body["execution"]["version"] == 1
    assert body["definition"]["version"] == 1
    assert "super-secret" not in detail.text
    assert body["node_runs"]
    assert body["events"]
```

- [ ] **Step 3: Write History UI RED tests**

Assert:

- selected workflow id is always included in normal history requests;
- All Executions rows include workflow name and version;
- filters serialize into query parameters;
- Cancel appears only for queued/running/waiting;
- cancel is idempotent;
- rerun submits a new run and retains the original execution;
- selected execution graph decoration loads its exact definition version.

- [ ] **Step 4: Run RED**

```bash
npm run test:workflows --workspace web
uv run --extra dev python -m pytest tests/hermes_cli/test_workflows_db.py tests/plugins/test_workflows_dashboard_plugin.py -q
```

- [ ] **Step 5: Extend the existing DB query without a query builder**

Append fixed internal clauses for provided filters and parameterize every value. Use the existing newest-first order. `before` is `(created_at, execution_id)` cursor data encoded as two query parameters; do not add offset pagination.

- [ ] **Step 6: Reuse existing serializers for detail**

Compose `_execution_to_dict`, `_redact_node_run_for_display`, `_event_to_dict`, and `_definition_to_dict`. Do not create alternate raw serializers.

- [ ] **Step 7: Implement rerun through `start_manual_execution()`**

Require explicit input in the request. The UI may prefill only non-sensitive values from the redacted detail; it never retrieves raw secrets. Return the new execution and `source_execution_id`.

- [ ] **Step 8: Implement History mode**

Add workflow-scoped list, All Executions toggle, filters, stable detail URL, timeline, node attempts, linked Kanban task, feed/item navigation, Cancel, and Rerun. Remove duplicate node-run rendering from legacy `app.js`.

- [ ] **Step 9: Run GREEN, rebuild, and commit**

```bash
npm run test:workflows --workspace web
npm run build:workflows --workspace web
uv run --extra dev python -m pytest \
  tests/hermes_cli/test_workflows_db.py \
  tests/plugins/test_workflows_dashboard_plugin.py \
  tests/plugins/test_workflows_dashboard_assets.py \
  tests/plugins/test_workflows_dashboard_build.py -q
git diff --check
git add hermes_cli/workflows_db.py plugins/workflows/dashboard tests
git commit -m "feat: add workflow execution history"
```

---

### Task 9: Complete Accessibility, Responsive Behavior, and Browser E2E

**Files:**
- Modify: `plugins/workflows/dashboard/src/workspace.js`
- Modify: `plugins/workflows/dashboard/src/build.js`
- Modify: `plugins/workflows/dashboard/src/run.js`
- Modify: `plugins/workflows/dashboard/src/history.js`
- Modify: `plugins/workflows/dashboard/src/style.css`
- Create: `scripts/test-workflows-dashboard-e2e.sh`
- Create: `tests/plugins/test_workflows_dashboard_e2e_script.py`
- Modify: `tests/plugins/test_workflows_dashboard_assets.py`

**Interfaces:**
- Consumes: existing root `agent-browser` dependency and `hermes dashboard` CLI.
- Produces: one non-interactive browser smoke command that exits non-zero on a failed assertion and saves viewport screenshots under a caller-provided directory.

- [ ] **Step 1: Write source RED tests for accessibility**

Using jsdom and React DOM, assert:

- Build/Run/History use native tab semantics;
- disclosures are buttons with `aria-expanded`/`aria-controls`;
- YAML editor has `Workflow definition YAML` label;
- dialogs trap focus and restore it to the opener;
- icon-only controls have accessible names;
- operation results use `aria-live="polite"`;
- status text exists in addition to color classes.

- [ ] **Step 2: Write RED test for the browser script contract**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "test-workflows-dashboard-e2e.sh"


def test_workflow_browser_script_covers_release_viewports():
    text = SCRIPT.read_text(encoding="utf-8")
    for viewport in ("1440 900", "1280 576", "1024 768", "768 1024", "390 844"):
        assert viewport in text
    for marker in (
        "Generate draft",
        "Accept draft",
        "Publish",
        "Start Run",
        "Open Continuous Feed",
        "Pause",
        "Resume",
        "Close",
        "Start new feed",
        "Cancel Execution",
    ):
        assert marker in text
```

- [ ] **Step 3: Run RED**

```bash
npm run test:workflows --workspace web
uv run --extra dev python -m pytest tests/plugins/test_workflows_dashboard_e2e_script.py -q
```

- [ ] **Step 4: Replace clickable headings with native controls**

Use buttons inside headings. Implement Escape close and focus return for sheets/dialogs. Use existing CSS focus tokens. Do not add an accessibility component library.

- [ ] **Step 5: Finalize responsive CSS**

- Wide: navigation + workspace + optional inspector.
- Medium: collapsed workflow navigation; inspector sheet.
- Narrow: one pane and mode-local tabs/sheets.
- Short: Build editor minimum 240px; no sibling panel may shrink it.
- Palette: wrap when space permits; otherwise show edge fade and Previous/Next controls with accessible names.
- Respect `prefers-reduced-motion: reduce`.

- [ ] **Step 6: Create the browser smoke script**

The script must:

1. create a temporary `HERMES_HOME`;
2. start `uv run --extra dev hermes dashboard --port 9148 --no-open --skip-build` in the background;
3. wait for HTTP 200;
4. use an isolated `agent-browser --session workflows-e2e`;
5. install an `agent-browser network route` for only `/api/plugins/workflows/definitions/draft` and `/refine`, returning fixed valid candidates so browser CI never depends on provider credentials;
6. exercise prompt generation, accept, structured edit, publish, invalid/valid run, feed open/pause/resume/close/new, history, and cancel against the real local API/DB for every route except the two mocked AI calls;
7. iterate all five `agent-browser set viewport` sizes;
8. assert with `agent-browser is visible`, `is enabled`, `get count`, and `eval` geometry checks;
9. save one screenshot per viewport;
10. assert no console errors;
11. always close the browser and dashboard in a shell `trap`.

The mocked candidates must use the same JSON fixtures exercised by assistant/API unit tests. The live real-provider check remains Task 10 Step 9 and is intentionally separate from deterministic browser CI.

The short-height geometry assertion is:

```javascript
JSON.stringify((() => {
  const body = document.querySelector('.hermes-workflows-build-mode');
  const rect = body.getBoundingClientRect();
  return { height: rect.height, visible: rect.height >= 240 };
})())
```

Parse the JSON with Python stdlib in the shell script and fail unless `visible` is true.

- [ ] **Step 7: Run GREEN and the real browser E2E**

```bash
npm run test:workflows --workspace web
npm run build:workflows --workspace web
uv run --extra dev python -m pytest tests/plugins/test_workflows_dashboard_e2e_script.py -q
bash scripts/test-workflows-dashboard-e2e.sh /tmp/hermes-workflows-browser-proof
```

Expected: script exits 0, five screenshots exist, no console errors, and every scenario completes.

- [ ] **Step 8: Commit**

```bash
git diff --check
git add plugins/workflows/dashboard scripts/test-workflows-dashboard-e2e.sh tests/plugins
git commit -m "test: cover workflow dashboard end to end"
```

---

### Task 10: Close Cross-Surface Parity, Upgrade, Documentation, and Release Gates

**Files:**
- Modify: `hermes_cli/workflows.py`
- Modify: `hermes_cli/workflows_capabilities.py`
- Modify: `tools/workflow_tools.py`
- Modify: `tests/hermes_cli/test_workflow_cli.py`
- Modify: `tests/hermes_cli/test_workflows_capabilities.py`
- Modify: `tests/tools/test_workflow_tools.py`
- Create: `tests/hermes_cli/test_workflows_db_upgrade.py`
- Modify: `tests/plugins/test_plugin_dashboard_auth_contract.py`
- Modify: `website/docs/user-guide/features/workflows.md`
- Modify: `website/docs/reference/cli-commands.md`
- Modify: `MANIFEST.in` only if the new source/build files are absent from the wheel

**Interfaces:**
- CLI/model tools continue using existing validate/deploy/run/tick commands.
- Dashboard drafts are a UI lifecycle; CLI/model deploy remains direct immutable version deployment.
- All surfaces share validation, redaction, published-version, feed-state, and terminal-status semantics.

- [ ] **Step 1: Write parity RED tests**

Add table-driven tests that submit the same invalid spec/input through CLI helper, model-tool handler, and Dashboard API and assert the same core message/code. Add tests that model-tool execution detail remains redacted and that CLI/model deploy never mutates an existing version.

Add a profile-capability parity case: an `agent_task` referencing a missing profile must fail validation/publish/deploy on all three surfaces with `workflow_profile_not_found`; `default` and names returned by `profiles_mod.list_profiles()` pass. Implement one `require_available_profiles(spec, available_names)` helper in `workflows_capabilities.py` and call it at each surface boundary after `require_implemented_primitives()`. Do not put profile filesystem access into `workflows_spec.py` or `workflows_db.py`.

- [ ] **Step 2: Write DB upgrade RED test**

Create a pre-draft/pre-archive SQLite fixture with the old tables, run `wfdb.init_db()`, and assert:

- old definitions/executions remain readable;
- `workflow_drafts` exists;
- `archived` exists with default 0;
- opening a continuous feed still works;
- no old row count changes.

Build the fixture in test code with explicit SQL; do not commit a binary DB.

- [ ] **Step 3: Pin mounted Dashboard auth**

Extend the mounted plugin auth contract so unauthenticated `/api/plugins/workflows/definitions` returns 401 and authenticated SDK-style access succeeds. Reuse existing dashboard token fixtures; do not build a workflow-specific auth layer.

- [ ] **Step 4: Run RED**

```bash
uv run --extra dev python -m pytest \
  tests/hermes_cli/test_workflow_cli.py \
  tests/hermes_cli/test_workflows_capabilities.py \
  tests/tools/test_workflow_tools.py \
  tests/hermes_cli/test_workflows_db_upgrade.py \
  tests/plugins/test_plugin_dashboard_auth_contract.py -q
```

- [ ] **Step 5: Apply only parity fixes revealed by tests**

Route all surfaces through existing shared helpers. Do not expose Dashboard draft APIs as model tools or CLI commands; direct immutable deploy is sufficient for those expert surfaces.

- [ ] **Step 6: Update user documentation**

Document:

- AI-first Generate -> Review -> Publish lifecycle;
- immutable versions and Run version selection;
- Build, Run, History modes;
- single and continuous input;
- feed state table with closed terminal and Start new feed;
- structured editor coverage;
- Advanced YAML import/export/diagnostics role;
- cancellation, rerun, and filtered history;
- dispatcher diagnostics and the normal gateway path;
- unsupported batch/documents/true loops;
- rollback command `hermes config set workflow.dispatch_in_gateway false`.

Correct any remaining statement that says gateway dispatch defaults to false; runtime default is true.

- [ ] **Step 7: Verify package contents**

```bash
rm -rf /tmp/hermes-workflows-release-build
uv build --out-dir /tmp/hermes-workflows-release-build
python3 - <<'PY'
from pathlib import Path
import zipfile
wheel = next(Path('/tmp/hermes-workflows-release-build').glob('*.whl'))
with zipfile.ZipFile(wheel) as zf:
    names = set(zf.namelist())
required = {
    'plugins/workflows/dashboard/manifest.json',
    'plugins/workflows/dashboard/plugin_api.py',
    'plugins/workflows/dashboard/dist/index.js',
    'plugins/workflows/dashboard/dist/style.css',
}
missing = sorted(required - names)
assert not missing, missing
print('workflow package assets verified')
PY
```

Authored `src/` files do not need to ship if generated `dist/` is complete; do not expand the wheel without a runtime need.

- [ ] **Step 8: Run the authoritative release gate**

```bash
npm run test:workflows --workspace web
npm run build:workflows --workspace web
npm run lint:workflows --workspace web
npm run lint --workspace web
uv run --extra dev ruff check \
  hermes_cli/workflows.py \
  hermes_cli/workflows_assistant.py \
  hermes_cli/workflows_capabilities.py \
  hermes_cli/workflows_db.py \
  hermes_cli/workflows_dispatcher.py \
  hermes_cli/workflows_intake.py \
  hermes_cli/workflows_spec.py \
  tools/workflow_tools.py \
  plugins/workflows/dashboard/plugin_api.py
uv run --extra dev python -m compileall -q \
  hermes_cli/workflows.py \
  hermes_cli/workflows_assistant.py \
  hermes_cli/workflows_capabilities.py \
  hermes_cli/workflows_db.py \
  hermes_cli/workflows_dispatcher.py \
  hermes_cli/workflows_intake.py \
  hermes_cli/workflows_spec.py \
  tools/workflow_tools.py \
  plugins/workflows/dashboard/plugin_api.py
uv run --extra dev python -m pytest \
  tests/hermes_cli/test_workflows_spec.py \
  tests/hermes_cli/test_workflows_intake.py \
  tests/hermes_cli/test_workflows_db.py \
  tests/hermes_cli/test_workflows_db_upgrade.py \
  tests/hermes_cli/test_workflows_dispatcher.py \
  tests/hermes_cli/test_workflows_e2e.py \
  tests/hermes_cli/test_workflow_cli.py \
  tests/hermes_cli/test_workflows_capabilities.py \
  tests/tools/test_workflow_tools.py \
  tests/plugins/test_workflows_dashboard_plugin.py \
  tests/plugins/test_workflows_dashboard_assets.py \
  tests/plugins/test_workflows_dashboard_build.py \
  tests/plugins/test_workflows_dashboard_e2e_script.py \
  tests/plugins/test_plugin_dashboard_auth_contract.py -q
npm audit --workspace web --omit=dev --audit-level=high
bash scripts/test-workflows-dashboard-e2e.sh /tmp/hermes-workflows-browser-proof
uv build --out-dir /tmp/hermes-workflows-release-build
git diff --check
git status --short --branch
```

Expected: all commands pass; worktree is clean after committing generated assets.

- [ ] **Step 9: Run a live real-provider smoke test**

Using an isolated `HERMES_HOME` that links only existing provider configuration, verify:

1. Generate a workflow draft from a prompt.
2. Refine it.
3. Accept and publish it.
4. Run valid input.
5. Confirm terminal success in History.
6. Remove temporary config/auth links afterward.

Do not persist credentials or provider responses in test fixtures, logs, screenshots, or commits.

- [ ] **Step 10: Repeat the original audit acceptance gate**

Re-run the 2026-07-10 UX/UI, flow, and functionality audit against the release candidate. Release only when:

- all ten original findings are closed with regression evidence;
- no Critical or High findings remain;
- no source behavior requires YAML to author supported v1 capabilities;
- all five viewport screenshots are current;
- Build, Run, and History work keyboard-only;
- CLI, model tools, API, Dashboard, and dispatcher agree on validation/status semantics.

- [ ] **Step 11: Commit**

```bash
git add hermes_cli tools plugins/workflows/dashboard tests scripts website/docs MANIFEST.in
git commit -m "docs: finalize workflows v1 release"
```

If `MANIFEST.in` was unchanged, omit it from `git add`.

---

## Audit Traceability

| 2026-07-10 finding | Closing task | Required evidence |
|---|---:|---|
| High: continuous-feed panel collapses Build editor | 4, 9 | Mode-isolation tests plus 1280x576 geometry >= 240px |
| High: trigger identity/status updates lose graph nodes | 2 | Pure graph membership tests plus browser run/reload flow |
| Medium: stale Kanban provenance regression | 0 | Focused E2E suite green |
| Medium: raw API envelopes shown to users | 2, 3 | Recursive formatter tests and typed API envelopes |
| Medium: no usable responsive model | 4, 9 | Five viewport E2E screenshots and assertions |
| Medium: clickable headings lack native semantics | 9 | jsdom keyboard/focus tests |
| Medium: Advanced YAML lacks accessible name | 6, 9 | Source render assertion and browser accessibility snapshot |
| Medium: clipped palette lacks discoverability | 9 | wrap/overflow controls tested at narrow viewports |
| Low: Manual Tick result is ambiguous | 7 | structured TickReport API/UI tests |
| Low: execution history defaults to global scope | 8 | workflow-filter query and All Executions opt-in tests |

Every finding must have both an automated regression and release-candidate browser evidence before the repeat audit can mark it closed.

## Execution Order and Review Gates

Execute Tasks 0-10 serially. Do not parallelize tasks that touch `src/app.js`, `plugin_api.py`, or `workflows_db.py`. A reviewer may reject one task without invalidating later contracts, but later tasks assume all earlier task interfaces exist.

After each task:

1. Review correctness against that task's tests and interfaces.
2. Review Ponytail compliance: reuse existing helpers, remove superseded paths, and reject unused scaffolding.
3. Run the task's focused tests independently.
4. Confirm no unrelated files changed.
5. Commit before starting the next task.

## Explicitly Skipped

- No frontend framework migration.
- No new state manager, router, form library, diff package, accessibility library, or browser dependency.
- No server-side draft revision history; one current draft per workflow plus a 20-entry local undo stack covers v1.
- No offset pagination; stable cursor filtering is sufficient.
- No v2 attachment, batch, or loop scaffolding.

Add any skipped item only after a measured v1 limitation or an approved v2 design requires it.
