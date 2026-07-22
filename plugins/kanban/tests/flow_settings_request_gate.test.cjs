const assert = require("node:assert/strict");
const test = require("node:test");
const api = require("../dashboard/flow_helpers.js");

function applyIfCurrent(api, boardRef, requestIdRef, request, update) {
  if (api.isCurrentLayoutSettingsRequest(
    boardRef,
    requestIdRef,
    request.requestBoard,
    request.requestId,
  )) update();
}

test("stale GET completion cannot update after a board switch", () => {
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

test("invalidating a request prevents its completion from updating layout state", () => {
  const boardRef = { current: "alpha" };
  const requestIdRef = { current: 0 };
  const request = api.beginLayoutSettingsRequest(boardRef, requestIdRef);

  api.invalidateLayoutSettingsRequests(requestIdRef);

  assert.equal(
    api.isCurrentLayoutSettingsRequest(
      boardRef, requestIdRef, request.requestBoard, request.requestId,
    ),
    false,
  );
});
