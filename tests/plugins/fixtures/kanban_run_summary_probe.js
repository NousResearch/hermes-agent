// Behavioral probe for the run-history summary rendering: extracts RunHistorySection()
// from the shipped dashboard bundle (no build step — the bundle IS the source) and runs
// it with a minimal React `h` stub, then walks the returned element tree.
// Contract: a run's `summary` is agent-written markdown, so it must reach the drawer's
// markdown renderer (MarkdownBlock) instead of being interpolated as a plain string —
// otherwise links, `**bold**` and inline code stay literal. `enabled` must forward the
// caller's renderMarkdown toggle so the drawer's "render markdown" switch still applies.
// Run via: node kanban_run_summary_probe.js <path-to-bundle>
const fs = require("fs");

const bundlePath = process.argv[2];
const src = fs.readFileSync(bundlePath, "utf8");
const start = src.indexOf("function RunHistorySection(");
if (start === -1) {
  console.error("RunHistorySection not found in bundle");
  process.exit(1);
}
const bodyStart = src.indexOf("{", start);
let depth = 0;
let end = bodyStart;
for (; end < src.length; end++) {
  if (src[end] === "{") depth++;
  else if (src[end] === "}") {
    depth--;
    if (depth === 0) break;
  }
}
const fnSrc = src.slice(start, end + 1);

// --- React-shaped stubs -----------------------------------------------------
function h(type, props) {
  const children = Array.prototype.slice.call(arguments, 2);
  return { type, props: props || {}, children };
}
const tx = (_t, _path, fallback) => fallback;
const cn = (...parts) => parts.filter(Boolean).join(" ");
const useState = initial => [initial, () => {}];
const useEffect = () => {};
const useI18n = () => ({ t: {} });
const timeAgo = () => "1m";
function MarkdownBlock(props) {
  return { type: MarkdownBlock, props: props || {}, children: [] };
}

const factory = new Function(
  "h", "tx", "cn", "useState", "useEffect", "useI18n", "timeAgo", "MarkdownBlock",
  `${fnSrc}\nreturn RunHistorySection;`,
);
const RunHistorySection = factory(h, tx, cn, useState, useEffect, useI18n, timeAgo, MarkdownBlock);

// --- helpers ----------------------------------------------------------------
function walk(node, visit) {
  if (!node || typeof node !== "object") return;
  if (Array.isArray(node)) {
    node.forEach(child => walk(child, visit));
    return;
  }
  visit(node);
  (node.children || []).forEach(child => walk(child, visit));
}
function findByClass(tree, className) {
  let found = null;
  walk(tree, node => {
    if (!found && node.props && node.props.className === className) found = node;
  });
  return found;
}

const failures = [];
function check(name, ok, detail) {
  if (!ok) failures.push(`${name}${detail ? `: ${detail}` : ""}`);
}

const SUMMARY = "Shipped **PR #123** — see [the diff](https://example.invalid/pr/123)";
const run = { id: "r1", started_at: 1000, ended_at: 1060, summary: SUMMARY, outcome: "done" };

// 1. Markdown on (default): the summary reaches MarkdownBlock, verbatim.
const onTree = RunHistorySection({ runs: [run] });
const onSummary = findByClass(onTree, "hermes-kanban-run-summary");
check("summary wrapper rendered", Boolean(onSummary));
const onMd = onSummary && (onSummary.children || []).find(c => c && c.type === MarkdownBlock);
check("summary goes through MarkdownBlock", Boolean(onMd));
if (onMd) {
  check("source is the agent text", onMd.props.source === SUMMARY, `got ${JSON.stringify(onMd.props.source)}`);
  check("enabled forwards renderMarkdown (true)", onMd.props.enabled !== false, `got ${JSON.stringify(onMd.props.enabled)}`);
}

// 2. The drawer's markdown toggle must reach it: renderMarkdown:false disables the renderer.
const offTree = RunHistorySection({ runs: [run], renderMarkdown: false });
const offSummary = findByClass(offTree, "hermes-kanban-run-summary");
const offMd = offSummary && (offSummary.children || []).find(c => c && c.type === MarkdownBlock);
check("toggle reaches the summary", Boolean(offMd) && offMd.props.enabled === false,
  offMd ? `enabled=${JSON.stringify(offMd.props.enabled)}` : "no MarkdownBlock");

// 3. A run without a summary still renders no summary block.
const bare = RunHistorySection({ runs: [{ id: "r2", started_at: 1000, ended_at: 1005 }] });
check("no summary -> no block", findByClass(bare, "hermes-kanban-run-summary") === null);

if (failures.length) {
  console.error(`FAIL\n- ${failures.join("\n- ")}`);
  process.exit(1);
}
console.log("PASS");
