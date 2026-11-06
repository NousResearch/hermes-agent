// Behavioral probe for the bulk "Complete" button: runs the dashboard's real applyBulk and
// requestCompletionSummary callbacks, extracted from the shipped bundle (no build step, so the
// bundle IS the source), with stubbed React/SDK dependencies. The backend refuses to mark a
// card done without a result, so a bulk done must ask for a summary and send it. Exits 0 and
// prints "PASS" when it does. Run via: node kanban_bulk_complete_probe.js <path-to-bundle>
const fs = require("fs");

const src = fs.readFileSync(process.argv[2], "utf8");

function fail(msg) {
  console.error("FAIL: " + msg);
  process.exit(1);
}

// Index just past the bracket closing the one at `open`. Skips string literals and
// comments so brackets inside them are not counted.
function matchClose(text, open) {
  const closer = { "(": ")", "{": "}", "[": "]" };
  const stack = [];
  for (let i = open; i < text.length; i++) {
    const c = text[i];
    if (c === "\"" || c === "'" || c === "`") {
      for (i++; i < text.length && text[i] !== c; i++) if (text[i] === "\\") i++;
    } else if (c === "/" && text[i + 1] === "/") {
      i = text.indexOf("\n", i);
      if (i === -1) break;
    } else if (c === "/" && text[i + 1] === "*") {
      i = text.indexOf("*/", i + 2);
      if (i === -1) break;
      i++;
    } else if (closer[c]) {
      stack.push(closer[c]);
    } else if (c === ")" || c === "}" || c === "]") {
      if (stack.pop() !== c) break;
      if (stack.length === 0) return i + 1;
    }
  }
  fail("unbalanced brackets extracting from offset " + open);
}

function extract(marker, openChar) {
  const start = src.indexOf(marker);
  if (start === -1) fail(marker + " not found in bundle");
  return src.slice(start, matchClose(src, src.indexOf(openChar, start)));
}

// Stand-ins for what the callbacks close over inside the board page component.
const board = "default";
let selectedIds = new Set();
const requests = [];
let prompts = 0;
let promptAnswer = null;
const API = "/api/plugins/kanban";
const t = {};
function tx(_t, _key, fallback) { return fallback; }
function useCallback(fn) { return fn; }
global.window = {
  prompt() { prompts++; return promptAnswer; },
  alert() {},
};
const kanbanDialogs = { request() { return Promise.resolve({ confirmed: true }); } };
const SDK = {
  fetchJSON(url, opts) {
    requests.push({ url: url, body: JSON.parse(opts.body) });
    return Promise.resolve({ results: [] });
  },
};
function loadBoard() {}
function setBoardData() {}
function setFailedIds() {}
function setLastSelectedId() {}
function setSelectedIds(v) { selectedIds = typeof v === "function" ? v(selectedIds) : v; }
function setError(e) { fail("callback reported an error: " + e); }

eval(extract("function withBoard", "{"));
eval(extract("function dialogLabelForCount", "{"));
const requestCompletionSummary = eval("(function () { " + extract("const requestCompletionSummary = useCallback(", "(") + "; return requestCompletionSummary; })()");
const applyBulk = eval("(function () { " + extract("const applyBulk = useCallback(", "(") + "; return applyBulk; })()");

async function settle() {
  for (let i = 0; i < 5; i++) await new Promise((resolve) => setImmediate(resolve));
}

(async function main() {
  // Complete with a summary: the summary reaches the bulk request.
  selectedIds = new Set(["t_one", "t_two"]);
  promptAnswer = "  shipped both  ";
  applyBulk({ status: "done" }, "Mark 2 task(s) as done?");
  await settle();
  if (prompts !== 1) fail("bulk complete asked for a summary " + prompts + " time(s), expected 1");
  if (requests.length !== 1) fail("expected 1 bulk request, got " + JSON.stringify(requests));
  const body = requests[0].body;
  if (body.status !== "done") fail("bulk request status is " + JSON.stringify(body.status));
  if (JSON.stringify(body.ids) !== JSON.stringify(["t_one", "t_two"])) fail("wrong ids " + JSON.stringify(body.ids));
  if (body.summary !== "shipped both" || body.result !== "shipped both") {
    fail("bulk complete did not send the summary: " + JSON.stringify(body));
  }

  // Cancelling the summary prompt sends nothing.
  requests.length = 0;
  prompts = 0;
  selectedIds = new Set(["t_three"]);
  promptAnswer = null;
  applyBulk({ status: "done" }, "Mark 1 task(s) as done?");
  await settle();
  if (requests.length !== 0) fail("cancelled summary still sent " + JSON.stringify(requests));

  // Other bulk moves do not ask for a summary.
  selectedIds = new Set(["t_four"]);
  applyBulk({ status: "ready" });
  await settle();
  if (prompts !== 1) fail("expected only the cancelled prompt, got " + prompts);
  if (requests.length !== 1 || requests[0].body.summary !== undefined) {
    fail("non-done bulk move changed: " + JSON.stringify(requests));
  }

  console.log("PASS");
})().catch(function (e) { fail(String((e && e.stack) || e)); });
