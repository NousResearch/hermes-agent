// Behavioral probe for the kanban task modal's text helpers: extracts
// renderMarkdown (with escapeHtml/renderInline) and taskSummary (with
// ADMIN_SUMMARY_RE) verbatim from the shipped dashboard bundle (no build step:
// the bundle IS the source) and runs them. Input is JSON on stdin:
//   {"markdown": [src, ...], "summaries": [latest_summary, ...]}
// Output is JSON on stdout: {"markdown": [html, ...], "summaries": [value|null, ...]}.
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
function extractLine(prefix) {
  const start = src.indexOf(prefix);
  if (start === -1) throw new Error(`${prefix} not found in bundle`);
  return src.slice(start, src.indexOf("\n", start));
}

const code = [
  extractFunction("escapeHtml"),
  extractFunction("renderInline"),
  extractFunction("renderMarkdown"),
  extractLine("const ADMIN_SUMMARY_RE"),
  extractFunction("taskSummary"),
  "module.exports = { renderMarkdown, taskSummary };",
].join("\n");
const mod = { exports: {} };
new Function("module", code)(mod);

const input = JSON.parse(fs.readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify({
  markdown: (input.markdown || []).map(mod.exports.renderMarkdown),
  summaries: (input.summaries || []).map(function (s) {
    return mod.exports.taskSummary({ latest_summary: s });
  }),
}));
