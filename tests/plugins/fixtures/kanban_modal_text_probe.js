// Behavioral probe for the kanban task modal's text helpers: extracts
// renderMarkdown (with escapeHtml/renderInline) verbatim from the shipped
// dashboard bundle (no build step: the bundle IS the source) and runs it.
// Input is JSON on stdin: {"markdown": [src, ...]}
// Output is JSON on stdout: {"markdown": [html, ...]}.
// Run via: node kanban_modal_text_probe.js <path-to-bundle>
const fs = require("fs");

const src = fs.readFileSync(process.argv[2], "utf8");
function extractFunction(name) {
  const start = src.indexOf(`function ${name}(`);
  if (start === -1) throw new Error(`${name} not found in bundle`);
  let depth = 0, end = src.indexOf("{", start);
  for (; end < src.length; end++) {
    if (src[end] === "{") depth++;
    else if (src[end] === "}") { depth--; if (depth === 0) break; }
  }
  return src.slice(start, end + 1);
}

const code = [
  extractFunction("escapeHtml"),
  extractFunction("renderInline"),
  extractFunction("renderMarkdown"),
  "module.exports = { renderMarkdown };",
].join("\n");
const mod = { exports: {} };
new Function("module", code)(mod);

const input = JSON.parse(fs.readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify({
  markdown: (input.markdown || []).map(mod.exports.renderMarkdown),
}));
