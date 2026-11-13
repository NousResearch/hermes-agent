// Behavioral probe for isImeComposing() in the kanban dashboard bundle: extracts the function
// from the shipped bundle (no build step — the bundle IS the source) and checks the events it
// must treat as part of an IME composition. Exits 0 and prints "PASS" when a composing Enter or
// Escape (isComposing, or keyCode 229 on a plain or React-style event) is claimed and an ordinary
// Enter or Escape is not. Run via: node kanban_ime_probe.js <path-to-bundle>
const fs = require("fs");

const src = fs.readFileSync(process.argv[2], "utf8");
const start = src.indexOf("function isImeComposing");
if (start === -1) { console.error("isImeComposing not found in bundle"); process.exit(1); }
const bodyStart = src.indexOf("{", start);
let depth = 0, end = bodyStart;
for (; end < src.length; end++) {
  if (src[end] === "{") depth++;
  else if (src[end] === "}") { depth--; if (depth === 0) break; }
}
eval(src.slice(start, end + 1));

const cases = [
  [{ key: "Enter", isComposing: true }, true, "composing Enter"],
  [{ key: "Escape", isComposing: true }, true, "composing Escape"],
  [{ key: "Enter", keyCode: 229 }, true, "commit Enter after compositionend (keyCode 229)"],
  [{ key: "Enter", nativeEvent: { isComposing: true } }, true, "React event, composing"],
  [{ key: "Enter", nativeEvent: { keyCode: 229 } }, true, "React event, keyCode 229"],
  [{ key: "Enter", isComposing: false, keyCode: 13 }, false, "plain Enter"],
  [{ key: "Escape", nativeEvent: { isComposing: false, keyCode: 27 } }, false, "plain Escape"],
];
for (const [event, expected, label] of cases) {
  if (isImeComposing(event) !== expected) {
    console.error(`FAIL: ${label}: expected ${expected}`);
    process.exit(1);
  }
}
console.log("PASS");
