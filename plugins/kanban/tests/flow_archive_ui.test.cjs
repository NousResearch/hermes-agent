const assert = require("node:assert/strict");
const test = require("node:test");
const api = require("../dashboard/flow_helpers.js");

test("synthetic Unlinked group never exposes a workflow action", () => {
  assert.equal(api.workflowIslandAction({ isUnlinked: true, archiveId: null }), null);
});

test("real active and archived groups expose archive and restore respectively", () => {
  assert.equal(api.workflowIslandAction({ isUnlinked: false, archiveId: null }), "archive");
  assert.equal(api.workflowIslandAction({ isUnlinked: false, archiveId: "wa_1" }), "restore");
});

test("strong confirmation requires acknowledgement and blocks repeat submit", () => {
  const preview = { counts: { active: 2, running: 1 } };
  assert.equal(api.workflowArchiveCanSubmit(preview, false, false), false);
  assert.equal(api.workflowArchiveCanSubmit(preview, false, true), true);
  assert.equal(api.workflowArchiveCanSubmit(preview, true, true), false);
});

test("dialog counts omit zero categories but retain canonical total", () => {
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
