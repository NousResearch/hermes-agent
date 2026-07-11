// Build / Run / History workspace mode routing and tab primitives.
//
// Ponytail: routes live here because the plugin already keeps mode state in the
// URL (initialExecutionIdFromLocation); centralising the parse/serialise pair
// lets the React tree reuse one source of truth instead of re-deriving paths
// from string slicing at every call site.
//
// upgrade path: if a fourth mode ever needs query-shaped state beyond
// {feed,execution}, replace `selection` with a `URLSearchParams` builder.

export const WORKSPACE_MODES = ["build", "run", "history"];
const WORKSPACE_PATH_PREFIX = "/workflows/";
const MODE_TO_QUERY_KEY = { run: "feed", history: "execution" };

function isMode(value) {
  return WORKSPACE_MODES.indexOf(value) !== -1;
}

function trimSlashes(value) {
  return String(value || "").replace(/^\/+|\/+$/g, "");
}

function parseLocation(location) {
  if (!location || typeof location !== "object") {
    return { workflowId: "", mode: "build", search: "" };
  }
  const rawPath = String(location.pathname || "");
  const trimmed = trimSlashes(rawPath).replace(/^\/+/, "");
  const segments = trimmed.split("/").filter(Boolean);
  // Accept either "/workflows/<id>" or "/workflows/<id>/<mode>" — anything else
  // routes to the build default so the host app can still navigate home.
  let workflowId = "";
  let modeSegment = "";
  if (segments.length >= 2 && segments[0] === "workflows") {
    workflowId = segments[1];
    modeSegment = segments[2] || "";
  }
  const search = String(location.search || "");
  const mode = isMode(modeSegment) ? modeSegment : "build";
  return { workflowId, mode, search };
}

export function modeForLocation(location) {
  return parseLocation(location).mode;
}

function selectionForMode(mode, selection) {
  const key = MODE_TO_QUERY_KEY[mode];
  if (!key) return "";
  const value = selection && typeof selection === "object" ? selection[key] : "";
  return value ? String(value) : "";
}

export function locationForMode(workflowId, mode, selection) {
  const id = trimSlashes(workflowId);
  const safeMode = isMode(mode) ? mode : "build";
  const base = WORKSPACE_PATH_PREFIX + id + (safeMode === "build" ? "" : "/" + safeMode);
  const value = selectionForMode(safeMode, selection);
  return value ? base + "?" + MODE_TO_QUERY_KEY[safeMode] + "=" + encodeURIComponent(value) : base;
}

const MODE_LABELS = { build: "Build", run: "Run", history: "History" };

function safeText(value) {
  if (value === null || value === undefined || value === "") return "—";
  return String(value);
}

// Render the workspace tablist. The dashboard wraps this in its own React
// createElement at runtime; for unit tests we return a plain DOM node so the
// jsdom suite can assert role / aria-selected without React on the test page.
export function renderWorkspaceTabs({ workflowId, mode, onSelect, runDisabled }) {
  const safeMode = isMode(mode) ? mode : "build";
  const list = document.createElement("div");
  list.setAttribute("role", "tablist");
  list.setAttribute("aria-label", "Workflow workspace modes");
  list.className = "hermes-workflows-workspace-tabs";
  WORKSPACE_MODES.forEach((entry) => {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("role", "tab");
    button.setAttribute("data-workspace-mode", entry);
    button.setAttribute("aria-selected", entry === safeMode ? "true" : "false");
    button.setAttribute("aria-controls", "hermes-workflows-mode-" + entry);
    button.className = "hermes-workflows-workspace-tab" + (entry === safeMode ? " is-active" : "");
    button.textContent = MODE_LABELS[entry] || entry;
    if (entry === "run" && runDisabled) {
      button.disabled = true;
      button.setAttribute("aria-disabled", "true");
      button.title = "Run is disabled until the workflow is published";
    }
    if (typeof onSelect === "function") {
      button.addEventListener("click", function () {
        if (button.disabled) return;
        onSelect(entry, { workflowId: workflowId, location: locationForMode(workflowId, entry) });
      });
    }
    list.appendChild(button);
  });
  return list;
}

function statusBadge(workflow) {
  const status = String(workflow && workflow.status || "draft").toLowerCase();
  return status === "published" ? "published" : "draft";
}

function executionBadge(workflow) {
  return workflow && workflow.latest_execution_status
    ? String(workflow.latest_execution_status).toLowerCase()
    : "no runs";
}

function feedBadge(workflow) {
  const count = workflow && Number.isFinite(workflow.open_feed_count) ? workflow.open_feed_count : 0;
  return count + " open";
}

// Render the workflow summary list. Same plain-DOM strategy as the tablist:
// keeps the unit suite self-contained while the React tree uses the same data.
export function renderWorkflowSummary({ workflows, activeWorkflowId, onSelect }) {
  const list = document.createElement("div");
  list.className = "hermes-workflows-workflow-summary";
  list.setAttribute("role", "list");
  (workflows || []).forEach(function (workflow) {
    const row = document.createElement("button");
    row.type = "button";
    row.setAttribute("role", "listitem");
    row.setAttribute("data-workflow-id", safeText(workflow && workflow.workflow_id));
    const isActive = workflow && workflow.workflow_id === activeWorkflowId;
    if (isActive) row.setAttribute("aria-current", "true");
    row.className = "hermes-workflows-workflow-summary-row" + (isActive ? " is-selected" : "");
    const fields = [
      ["hermes-workflows-workflow-summary-name", "name", workflow && workflow.name],
      ["hermes-workflows-workflow-summary-status", "status", statusBadge(workflow)],
      ["hermes-workflows-workflow-summary-version", "version", workflow && workflow.version != null ? "v" + workflow.version : "v—"],
      ["hermes-workflows-workflow-summary-enabled", "enabled", workflow && workflow.enabled ? "enabled" : "disabled"],
      ["hermes-workflows-workflow-summary-execution", "execution", executionBadge(workflow)],
      ["hermes-workflows-workflow-summary-feeds", "feeds", feedBadge(workflow)],
    ];
    fields.forEach(function (entry) {
      const span = document.createElement("span");
      span.className = entry[0];
      span.setAttribute("data-field", entry[1]);
      span.textContent = safeText(entry[2]);
      row.appendChild(span);
    });
    if (typeof onSelect === "function") {
      row.addEventListener("click", function () { onSelect(workflow); });
    }
    list.appendChild(row);
  });
  return list;
}