// Behavioral probe for #29347: runs the dashboard's real deleteTask and deleteSelected
// callbacks, extracted verbatim from the shipped bundle (no build step, so the bundle IS the
// source), with stubbed React/SDK dependencies, and checks that every DELETE they send
// carries the selected board. Without ?board= the backend falls back to its own board
// resolution and looks the task up in the wrong database. Exits 0 and prints "PASS" when
// every request is board-scoped. Run via: node kanban_delete_board_probe.js <path-to-bundle>
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
let board = null;
let selectedIds = new Set();
const requests = [];
const API = "/api/plugins/kanban";
const t = {};
const FALLBACK_TRASH = { confirm: "Delete task?" };
function tx(_t, _key, fallback) { return fallback; }
function useCallback(fn) { return fn; }
const kanbanDialogs = { request() { return Promise.resolve({ confirmed: true }); } };
const SDK = {
  fetchJSON(url, opts) {
    requests.push({ url: url, method: (opts && opts.method) || "GET" });
    return Promise.resolve({});
  },
};
function loadBoard() { return Promise.resolve(); }
function setSelectedIds(v) { selectedIds = typeof v === "function" ? v(selectedIds) : v; }
function setError(e) { fail("callback reported an error: " + e); }

eval(extract("function withBoard", "{"));
const deleteTask = eval("(function () { " + extract("const deleteTask = useCallback(", "(") + "; return deleteTask; })()");
const deleteSelected = eval("(function () { " + extract("const deleteSelected = useCallback(", "(") + "; return deleteSelected; })()");

async function settle() {
  for (let i = 0; i < 5; i++) await new Promise((resolve) => setImmediate(resolve));
}

function expectScopedDeletes(label, ids) {
  if (requests.length !== ids.length) {
    fail(label + ": expected " + ids.length + " request(s), got " + JSON.stringify(requests));
  }
  requests.forEach(function (r, n) {
    const u = new URL(r.url, "http://dashboard.invalid");
    if (r.method !== "DELETE") fail(label + ": expected DELETE, got " + r.method + " " + r.url);
    if (u.pathname !== API + "/tasks/" + ids[n]) fail(label + ": wrong task URL " + r.url);
    if (u.searchParams.get("board") !== board) {
      fail(label + ": DELETE " + r.url + " does not target the selected board " + JSON.stringify(board));
    }
  });
  requests.length = 0;
}

(async function main() {
  board = "side-project";

  await deleteTask("t_one");
  await settle();
  expectScopedDeletes("deleteTask", ["t_one"]);

  selectedIds = new Set(["t_two", "t_three"]);
  deleteSelected(2);
  await settle();
  expectScopedDeletes("deleteSelected", ["t_two", "t_three"]);

  console.log("PASS");
})().catch(function (e) { fail(String((e && e.stack) || e)); });
