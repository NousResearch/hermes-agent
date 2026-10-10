// Behavioral probe for the /kanban?search=<query> deep link: extracts readUrlParam() from the
// shipped dashboard bundle (no build step — the bundle IS the source) and runs it against stubbed
// window.location values, then checks the search filter's initial state is seeded from it.
// Exits 0 and prints "PASS" on success. Run via: node kanban_search_param_probe.js <path-to-bundle>
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
check("?search=deploy%20bug", "search", "deploy bug");
check("?board=ops&search=t_123", "search", "t_123");
check("?search=+hello+", "search", "hello"); // '+' decodes to space, then trimmed
check("?search=", "search", null);
check("?search=%20%20", "search", null);
check("", "search", null);
check("?board=ops", "search", null);

// A throwing location (sandboxed iframe, odd host) must not break the board.
global.window = {};
Object.defineProperty(window, "location", { get() { throw new Error("blocked"); } });
if (readUrlParam("search") !== null) { console.error("FAIL: throwing location not handled"); process.exit(1); }

// The board's search filter state must be seeded from the URL param.
if (!/const \[search, setSearch\] = useState\(function \(\) \{ return readUrlParam\("search"\) \|\| ""; \}\);/.test(src)) {
  console.error("FAIL: search filter state is not initialised from readUrlParam(\"search\")");
  process.exit(1);
}
console.log("PASS");
