const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function loadRequestGate() {
  const bundlePath = path.join(__dirname, "..", "dashboard", "dist", "index.js");
  const bundle = fs.readFileSync(bundlePath, "utf8");
  const start = bundle.indexOf("  function beginLayoutSettingsRequest");
  const end = bundle.indexOf("\n  function KanbanPage()", start);
  assert.ok(start >= 0 && end > start, "layout request-gate source must be discoverable");
  const source = bundle.slice(start, end);
  return vm.runInNewContext(
    `(function () { ${source}; return { beginLayoutSettingsRequest, isCurrentLayoutSettingsRequest }; })()`,
    { Object },
  );
}

function loadChangeLayoutPreset(dependencies) {
  const bundlePath = path.join(__dirname, "..", "dashboard", "dist", "index.js");
  const bundle = fs.readFileSync(bundlePath, "utf8");
  const start = bundle.indexOf("    const changeLayoutPreset = useCallback");
  const end = bundle.indexOf("\n\n    // --- load list of boards", start);
  assert.ok(start >= 0 && end > start, "layout PATCH callback source must be discoverable");
  const source = bundle.slice(start, end);
  return vm.runInNewContext(
    `(function () {
      const layoutPreset = "balanced-horizontal";
      ${source}
      return changeLayoutPreset;
    })()`,
    Object.assign({ JSON, useCallback: (callback) => callback }, dependencies),
  );
}

function applyIfCurrent(api, boardRef, requestIdRef, request, update) {
  if (api.isCurrentLayoutSettingsRequest(
    boardRef,
    requestIdRef,
    request.requestBoard,
    request.requestId,
  )) update();
}

test("stale GET completion cannot update after a board switch", () => {
  const api = loadRequestGate();
  const boardRef = { current: "alpha" };
  const requestIdRef = { current: 0 };
  const alphaGet = api.beginLayoutSettingsRequest(boardRef, requestIdRef);
  let preset = "balanced-horizontal";

  boardRef.current = "beta";
  requestIdRef.current += 1;
  applyIfCurrent(api, boardRef, requestIdRef, alphaGet, () => {
    preset = "compact";
  });

  assert.equal(preset, "balanced-horizontal");
});

test("only the latest overlapping PATCH may update save state", () => {
  const api = loadRequestGate();
  const boardRef = { current: "alpha" };
  const requestIdRef = { current: 0 };
  const firstPatch = api.beginLayoutSettingsRequest(boardRef, requestIdRef);
  const secondPatch = api.beginLayoutSettingsRequest(boardRef, requestIdRef);
  const updates = [];

  applyIfCurrent(api, boardRef, requestIdRef, firstPatch, () => updates.push("stale error"));
  applyIfCurrent(api, boardRef, requestIdRef, firstPatch, () => updates.push("stale saved"));
  applyIfCurrent(api, boardRef, requestIdRef, secondPatch, () => updates.push("latest saved"));

  assert.deepEqual(updates, ["latest saved"]);
});

test("failed current PATCH keeps the optimistic preset and exposes error state", async () => {
  const presets = [];
  const states = [];
  const errors = [];
  const changeLayoutPreset = loadChangeLayoutPreset({
    API: "/api/plugins/kanban",
    SDK: { fetchJSON: () => Promise.reject(new Error("disk full")) },
    beginLayoutSettingsRequest: () => ({ requestBoard: "alpha", requestId: 1 }),
    board: "alpha",
    currentBoardRef: { current: "alpha" },
    isCurrentLayoutSettingsRequest: () => true,
    layoutRequestIdRef: { current: 1 },
    parseApiErrorMessage: (error) => error.message,
    setError: (message) => errors.push(message),
    setLayoutPreset: (preset) => presets.push(preset),
    setLayoutSettingsState: (state) => states.push(state),
    withBoard: (url) => url,
  });

  changeLayoutPreset("compact");
  await new Promise(setImmediate);

  assert.deepEqual(presets, ["compact"]);
  assert.deepEqual(states, ["saving", "error"]);
  assert.deepEqual(errors, ["Layout update failed: disk full"]);
});
