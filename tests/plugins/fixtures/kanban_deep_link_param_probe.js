// Behavioral probe for the /kanban deep-link params: extracts readUrlParam() from the
// shipped dashboard bundle (no build step — the bundle IS the source) and runs it against
// stubbed window.location values, then checks that the board and task-drawer state are
// seeded from ?board= and ?task=.
//
// The bundle is the source of truth for the dashboard surface, so the probe reads the
// shipped file directly rather than a re-implementation that could drift from it.
// Exits 0 and prints "PASS" on success. Run via:
//   node kanban_deep_link_param_probe.js <path-to-bundle>
const fs = require("fs");

const src = fs.readFileSync(process.argv[2], "utf8");

const start = src.indexOf("function readUrlParam");
if (start === -1) { console.error("readUrlParam not found in bundle"); process.exit(1); }
const bodyStart = src.indexOf("{", start);
let depth = 0, end = bodyStart;
for (; end < src.length; end++) {
  if (src[end] === "{") depth++;
  else if (src[end] === "}") { depth--; if (depth === 0) break; }
}
global.window = { location: { search: "" } };
eval(src.slice(start, end + 1));

function check(search, name, expected) {
  window.location.search = search;
  const got = readUrlParam(name);
  if (got !== expected) {
    console.error(`FAIL: readUrlParam(${JSON.stringify(name)}) with ${JSON.stringify(search)} = ${JSON.stringify(got)}, expected ${JSON.stringify(expected)}`);
    process.exit(1);
  }
}

// The three deep-link params share one reader.
check("?board=alpha", "board", "alpha");
check("?task=t_123", "task", "t_123");
check("?search=urgent", "search", "urgent");

// Several params in one query string.
check("?board=alpha&task=t_9&search=x", "task", "t_9");
check("?board=alpha&task=t_9&search=x", "board", "alpha");

// Absent param -> null (the "not configured" case, not an empty string).
check("", "board", null);
check("?other=1", "task", null);

// A blank value is not a value: it must not seed a board or a task drawer.
check("?board=", "board", null);
check("?task=", "task", null);

// Surrounding whitespace is trimmed, so a stray space cannot name a different board.
check("?board=%20alpha%20", "board", "alpha");

console.log("PASS");
