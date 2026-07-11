import { describe, expect, it } from "vitest";
import {
  WORKSPACE_MODES,
  locationForMode,
  modeForLocation,
  renderWorkflowSummary,
  renderWorkspaceTabs,
} from "./workspace.js";

// ---- URL routing round-trips (plan lines 696–710) ----------------------------

describe("workspace routing", () => {
  it("exposes build/run/history modes", () => {
    expect(WORKSPACE_MODES.slice().sort()).toEqual(["build", "history", "run"]);
  });

  it("defaults a workflow workspace to build", () => {
    expect(modeForLocation({ pathname: "/workflows/demo", search: "" })).toBe("build");
    expect(modeForLocation({ pathname: "/workflows/demo/", search: "" })).toBe("build");
    expect(modeForLocation({ pathname: "/workflows/demo", search: "?" })).toBe("build");
  });

  it("parses /run and /history modes from the pathname", () => {
    expect(modeForLocation({ pathname: "/workflows/demo/run", search: "" })).toBe("run");
    expect(modeForLocation({ pathname: "/workflows/demo/history", search: "" })).toBe("history");
  });

  it("falls back to build on unknown mode segments", () => {
    expect(modeForLocation({ pathname: "/workflows/demo/banana", search: "" })).toBe("build");
  });

  it("round-trips run and history selections", () => {
    expect(locationForMode("demo", "run", { feed: "wffeed_1" }))
      .toBe("/workflows/demo/run?feed=wffeed_1");
    expect(locationForMode("demo", "history", { execution: "wfexec_1" }))
      .toBe("/workflows/demo/history?execution=wfexec_1");
  });

  it("round-trips build mode without query string", () => {
    expect(locationForMode("demo", "build")).toBe("/workflows/demo");
  });

  it("round-trips back through modeForLocation for every supported mode", () => {
    for (const mode of WORKSPACE_MODES) {
      const selection = mode === "run" ? { feed: "wffeed_1" } : mode === "history" ? { execution: "wfexec_1" } : {};
      const next = locationForMode("demo", mode, selection);
      const parsed = new URL("http://x" + next);
      const search = parsed.search ? "?" + parsed.search.slice(1) : "";
      expect(modeForLocation({ pathname: parsed.pathname, search })).toBe(mode);
    }
  });

  it("ignores query keys not relevant to the chosen mode", () => {
    expect(locationForMode("demo", "run", { execution: "wfexec_1", feed: "wffeed_1" }))
      .toBe("/workflows/demo/run?feed=wffeed_1");
    expect(locationForMode("demo", "history", { feed: "wffeed_1", execution: "wfexec_1" }))
      .toBe("/workflows/demo/history?execution=wfexec_1");
  });
});

// ---- Tab semantics + mode isolation -----------------------------------------

function renderTabsFor(mode, extra) {
  const node = renderWorkspaceTabs(Object.assign({ workflowId: "demo", mode, onSelect: () => {} }, extra || {}));
  document.body.appendChild(node);
  return node;
}

describe("workspace tabs", () => {
  it("uses native buttons with role=tab inside a role=tablist container", () => {
    const node = renderTabsFor("build");
    expect(node.getAttribute("role")).toBe("tablist");
    const tabs = node.querySelectorAll('[role="tab"]');
    expect(tabs.length).toBe(WORKSPACE_MODES.length);
    tabs.forEach((tab) => {
      expect(tab.tagName.toLowerCase()).toBe("button");
    });
  });

  it("marks the active mode with aria-selected=true and others with false", () => {
    for (const mode of WORKSPACE_MODES) {
      const node = renderTabsFor(mode);
      const tabs = node.querySelectorAll('[role="tab"]');
      tabs.forEach((tab) => {
        const isActive = tab.getAttribute("data-workspace-mode") === mode;
        expect(tab.getAttribute("aria-selected")).toBe(isActive ? "true" : "false");
      });
    }
  });

  it("invokes onSelect with the picked mode when a tab is clicked", () => {
    let picked = null;
    const node = renderTabsFor("build", { onSelect: (next) => { picked = next; } });
    const historyTab = node.querySelector('[data-workspace-mode="history"]');
    historyTab.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(picked).toBe("history");
  });

  it("disables the run tab for draft-only workflows", () => {
    const node = renderTabsFor("build", { runDisabled: true });
    const runTab = node.querySelector('[data-workspace-mode="run"]');
    expect(runTab.hasAttribute("disabled")).toBe(true);
    expect(runTab.getAttribute("aria-disabled")).toBe("true");
  });
});

// ---- Workflow summary fields on the workspace list --------------------------

function buildWorkflowSummary() {
  return [
    { workflow_id: "demo", name: "Demo", status: "draft", version: null, enabled: false, latest_execution_status: null, open_feed_count: 0 },
    { workflow_id: "ops", name: "Ops", status: "published", version: 3, enabled: true, latest_execution_status: "succeeded", open_feed_count: 2 },
    { workflow_id: "broken", name: "Broken", status: "published", version: 7, enabled: false, latest_execution_status: "failed", open_feed_count: 1 },
  ];
}

describe("workflow navigation summary", () => {
  it("renders each row with name, status, version, enabled, latest execution, and open feed count", () => {
    const list = renderWorkflowSummary({
      workflows: buildWorkflowSummary(),
      activeWorkflowId: "ops",
      onSelect: () => {},
    });
    document.body.appendChild(list);

    const rows = list.querySelectorAll('[data-workflow-id]');
    expect(rows.length).toBe(3);

    const demo = list.querySelector('[data-workflow-id="demo"]');
    expect(demo.textContent).toMatch(/draft/i);
    expect(demo.textContent).toMatch(/v—|—|none/i);
    expect(demo.textContent).toMatch(/disabled|off/i);

    const ops = list.querySelector('[data-workflow-id="ops"]');
    expect(ops.textContent).toMatch(/published/i);
    expect(ops.textContent).toMatch(/v3/);
    expect(ops.textContent).toMatch(/enabled|on/i);
    expect(ops.textContent).toMatch(/succeeded/i);
    expect(ops.textContent).toMatch(/2/);
    expect(ops.getAttribute("aria-current")).toBe("true");

    const broken = list.querySelector('[data-workflow-id="broken"]');
    expect(broken.textContent).toMatch(/failed/i);
    expect(broken.textContent).toMatch(/v7/);
    expect(broken.textContent).toMatch(/disabled|off/i);
    expect(broken.textContent).toMatch(/1/);
  });
});

// ---- Responsive CSS invariants (plan lines 749–756) --------------------------

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const SRC_CSS = readFileSync(
  path.join(path.dirname(fileURLToPath(import.meta.url)), "style.css"),
  "utf8",
);

describe("workspace responsive CSS", () => {
  it("declares the three required viewport breakpoints", () => {
    expect(SRC_CSS).toMatch(/@media\s*\(max-width:\s*1279px\)/);
    expect(SRC_CSS).toMatch(/@media\s*\(max-width:\s*767px\)/);
    expect(SRC_CSS).toMatch(/@media\s*\(max-height:\s*600px\)/);
  });

  it("keeps the build mode at least 240px tall on short viewports", () => {
    expect(SRC_CSS).toMatch(/\.hermes-workflows-build-mode\s*\{[^}]*min-height:\s*240px/);
  });

  it("lets Run and History scroll independently of the editor", () => {
    expect(SRC_CSS).toMatch(/\.hermes-workflows-(run-mode|history-mode)[^{}]*\{[^}]*overflow-y:\s*auto/);
  });
});