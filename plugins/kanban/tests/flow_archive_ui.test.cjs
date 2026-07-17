const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function loadArchiveUiApi() {
  const bundle = fs.readFileSync(
    path.join(__dirname, "..", "dashboard", "dist", "index.js"),
    "utf8",
  );
  const start = bundle.indexOf("  function workflowIslandAction");
  const end = bundle.indexOf("\n  function WorkflowArchiveDialog", start);
  assert.ok(start >= 0 && end > start, "archive UI helpers must be discoverable");
  return vm.runInNewContext(
    `(function () { ${bundle.slice(start, end)}; return { workflowIslandAction, workflowArchiveCanSubmit, workflowArchiveCounts }; })()`,
    { Object, Array },
  );
}

test("synthetic Unlinked group never exposes a workflow action", () => {
  const api = loadArchiveUiApi();
  assert.equal(api.workflowIslandAction({ isUnlinked: true, archiveId: null }), null);
});

test("real active and archived groups expose archive and restore respectively", () => {
  const api = loadArchiveUiApi();
  assert.equal(api.workflowIslandAction({ isUnlinked: false, archiveId: null }), "archive");
  assert.equal(api.workflowIslandAction({ isUnlinked: false, archiveId: "wa_1" }), "restore");
});

test("strong confirmation requires acknowledgement and blocks repeat submit", () => {
  const api = loadArchiveUiApi();
  const preview = { counts: { active: 2, running: 1 } };
  assert.equal(api.workflowArchiveCanSubmit(preview, false, false), false);
  assert.equal(api.workflowArchiveCanSubmit(preview, false, true), true);
  assert.equal(api.workflowArchiveCanSubmit(preview, true, true), false);
});

test("dialog counts omit zero categories but retain canonical total", () => {
  const api = loadArchiveUiApi();
  assert.deepEqual(
    JSON.parse(JSON.stringify(api.workflowArchiveCounts({
      counts: { total: 4, active: 2, running: 1, review: 0, done: 2, archived: 0 },
    }))),
    [
      { key: "total", label: "Total", value: 4 },
      { key: "active", label: "Active", value: 2 },
      { key: "running", label: "Running", value: 1 },
      { key: "done", label: "Done", value: 2 },
    ],
  );
});