// Behavioral probe for the dashboard markdown XSS guard: extracts the markdown
// helpers + MarkdownBlock from the shipped bundle (no build step — the bundle IS
// the source) and runs them. Exits 0 and prints "PASS" when
//   1. sanitizeMarkdownHtml strips non-allowlisted tags, event-handler attributes
//      and non-http(s)/mailto hrefs from raw HTML;
//   2. MarkdownBlock routes its HTML through the sanitizer before
//      dangerouslySetInnerHTML (proved by swapping renderMarkdown for an identity
//      function that passes raw HTML straight through);
//   3. ordinary markdown still renders (bold, safe links).
// Run via: node kanban_markdown_sanitize_probe.js <path-to-bundle>
const fs = require("fs");

const src = fs.readFileSync(process.argv[2], "utf8");
const start = src.indexOf("function escapeHtml(");
const blockStart = src.indexOf("function MarkdownBlock(", start);
if (start === -1 || blockStart === -1) {
  console.error("markdown helpers / MarkdownBlock not found in bundle");
  process.exit(1);
}
const bodyStart = src.indexOf("{", blockStart);
let depth = 0;
let end = bodyStart;
for (; end < src.length; end++) {
  if (src[end] === "{") depth++;
  else if (src[end] === "}") { depth--; if (depth === 0) break; }
}
const code = src.slice(start, end + 1);

const h = (tag, props, ...children) => ({ tag, props, children });
eval(code); // sloppy-mode direct eval: function declarations land in this scope

const failures = [];
const check = (cond, msg) => { if (!cond) failures.push(msg); };
const DANGER = [/<img/i, /<script/i, /<iframe/i, /<svg/i, /\son\w+\s*=/i, /javascript:/i];
const assertClean = (html, label) => {
  for (const re of DANGER) check(!re.test(html), `${label}: ${re} survived in ${JSON.stringify(html)}`);
};

const RAW =
  '<img src=x onerror=alert(1)><script>alert(1)</script><iframe src="https://evil"></iframe>' +
  '<svg onload=alert(1)></svg><a href="javascript:alert(1)" onclick="steal()">y</a>' +
  '<p onmouseover="steal()">z</p><a href="https://ok.example/">ok</a>';

// 1. The sanitizer itself.
const cleaned = sanitizeMarkdownHtml(RAW);
assertClean(cleaned, "sanitizeMarkdownHtml");
check(cleaned.includes("<p>z</p>"), `allowlisted <p> lost: ${cleaned}`);
check(cleaned.includes('href="https://ok.example/"'), `safe https href lost: ${cleaned}`);

// 2. MarkdownBlock must sanitize whatever the renderer produces.
const realRender = renderMarkdown;
renderMarkdown = (s) => s;
const wired = MarkdownBlock({ source: RAW }).props.dangerouslySetInnerHTML.__html;
assertClean(wired, "MarkdownBlock(raw renderer)");
renderMarkdown = realRender;

// 3. Ordinary markdown still renders through the real renderer.
const md = MarkdownBlock({ source: "**bold** and [link](https://example.com)" }).props.dangerouslySetInnerHTML.__html;
check(md.includes("<strong>bold</strong>"), `bold lost: ${md}`);
check(md.includes('href="https://example.com"'), `link lost: ${md}`);

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log("PASS");
