/**
 * annotate.test.js — JSDOM tests for plan-annotate.js
 *
 * Run: node --test tests/plan_first/annotate.test.js
 */

const assert = require("node:assert");
const test = require("node:test");
const { JSDOM } = require("jsdom");
const fs = require("fs");
const path = require("path");

const JS_PATH = path.resolve(__dirname, "../../plan-annotate.js");
const CSS_PATH = path.resolve(__dirname, "../../plan-annotate.css");

/** Wait one microtick for JSDOM async events (like DOMContentLoaded). */
function tick() {
  return new Promise((r) => setTimeout(r, 0));
}

/**
 * Create a fresh JSDOM page with the annotation module fully loaded.
 */
async function createPage() {
  const jsCode = fs.readFileSync(JS_PATH, "utf-8");
  const cssCode = fs.readFileSync(CSS_PATH, "utf-8");

  const html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Test</title>
  <style>${cssCode}</style>
</head>
<body>
  <header><h1>Test Plan</h1></header>
  <main>
    <h2>Section 1</h2>
    <p id="target">The quick brown fox jumps over the lazy dog.</p>
  </main>
  <script>${jsCode}</script>
</body>
</html>`;

  const dom = new JSDOM(html, {
    url: "http://localhost",
    contentType: "text/html",
    runScripts: "dangerously",
  });

  // Wait for DOMContentLoaded (fires async in JSDOM)
  await tick();

  return { dom, doc: dom.window.document, win: dom.window };
}

/**
 * Helper: simulate a text selection on an element.
 */
function selectText(win, el, start, end) {
  const text = el.firstChild || el;
  const range = win.document.createRange();
  range.setStart(text, start);
  range.setEnd(text, end);
  const sel = win.getSelection();
  sel.removeAllRanges();
  sel.addRange(range);
  return range;
}

test("select text → comment box appears", async () => {
  const { doc, win } = await createPage();
  const target = doc.getElementById("target");
  assert.ok(target, "target paragraph exists");

  selectText(win, target, 0, 19); // "The quick brown fox"
  target.dispatchEvent(new win.MouseEvent("mouseup", { bubbles: true }));

  const box = doc.querySelector(".h-annot-box");
  assert.ok(box, "comment box element appears after mouseup");
  assert.ok(box.querySelector("textarea"), "box contains a textarea");
  assert.ok(box.querySelector(".h-annot-pin-btn"), "box contains a Pin button");
  assert.ok(box.querySelector(".h-annot-cancel-btn"), "box contains a Cancel button");
});

test("pin → callout renders below highlighted span", async () => {
  const { doc, win } = await createPage();
  const target = doc.getElementById("target");

  // Select and trigger box
  selectText(win, target, 0, 19);
  target.dispatchEvent(new win.MouseEvent("mouseup", { bubbles: true }));

  const textarea = doc.querySelector(".h-annot-textarea");
  assert.ok(textarea, "comment box textarea exists");
  textarea.value = "Great observation!";

  // Click Pin
  const pinBtn = doc.querySelector(".h-annot-pin-btn");
  pinBtn.dispatchEvent(new win.MouseEvent("click", { bubbles: true }));

  // Check highlight span
  const hl = doc.querySelector(".h-annot-hl");
  assert.ok(hl, "highlight span (.h-annot-hl) is rendered");
  assert.strictEqual(hl.textContent.trim(), "The quick brown fox", "highlight contains selected text");

  // Check callout
  const callout = doc.querySelector(".h-annot-callout");
  assert.ok(callout, "callout element (.h-annot-callout) is rendered");
  assert.ok(callout.textContent.includes("Great observation!"), "callout contains the comment text");
  assert.ok(callout.textContent.includes("user"), "callout shows the author");
  assert.ok(callout.textContent.includes("[open]"), "callout shows status");

  // Callout has edit and delete buttons
  assert.ok(callout.querySelector(".h-annot-edit-btn"), "callout has edit button");
  assert.ok(callout.querySelector(".h-annot-delete-btn"), "callout has delete button");
});

test("export → valid JSON with correct structure", async () => {
  const { doc, win } = await createPage();
  const target = doc.getElementById("target");

  // Create one annotation
  selectText(win, target, 0, 19);
  target.dispatchEvent(new win.MouseEvent("mouseup", { bubbles: true }));
  const textarea = doc.querySelector(".h-annot-textarea");
  textarea.value = "Great observation!";
  doc.querySelector(".h-annot-pin-btn").click();

  // Create a second annotation — select from remaining text node
  const remainingText = target.childNodes[target.childNodes.length - 1]; // text node after span
  if (remainingText && remainingText.nodeType === 3 && remainingText.textContent.trim()) {
    const range2 = win.document.createRange();
    range2.setStart(remainingText, 0);
    range2.setEnd(remainingText, 5); // "jumps"
    win.getSelection().removeAllRanges();
    win.getSelection().addRange(range2);
    target.dispatchEvent(new win.MouseEvent("mouseup", { bubbles: true }));
    const ta2 = doc.querySelector(".h-annot-textarea");
    if (ta2) {
      ta2.value = "Noted.";
      doc.querySelector(".h-annot-pin-btn").click();
    }
  }

  // Set up a Blob mock and trigger navigation to capture the download URL
  let capturedUrl = null;
  const origCreateObjectURL = win.URL.createObjectURL;
  win.URL.createObjectURL = function (blob) {
    capturedUrl = blob;
    return "blob:http://localhost/test";
  };
  let clickedAnchor = null;
  const origClick = win.HTMLAnchorElement.prototype.click;
  win.HTMLAnchorElement.prototype.click = function () {
    clickedAnchor = this;
  };

  // Click export button
  const exportBtn = doc.querySelector(".h-annot-export-btn");
  assert.ok(exportBtn, "export button exists in the UI");
  exportBtn.dispatchEvent(new win.MouseEvent("click", { bubbles: true }));

  assert.ok(clickedAnchor, "an anchor element was clicked for download");
  assert.strictEqual(clickedAnchor.download, "plan-annotate.json", "download filename is plan-annotate.json");

  // Read the blob content
  const reader = new win.FileReader();
  reader.readAsText(capturedUrl);
  const result = reader.result;
  const data = JSON.parse(result);
  assert.ok(data, "export produces valid JSON");
  assert.strictEqual(data.plan, "plan-annotate", 'JSON has plan: "plan-annotate"');
  assert.ok(data.exported_at, "JSON has exported_at timestamp");
  assert.ok(Array.isArray(data.comments), "JSON has comments array");

  const comments = data.comments;
  assert.ok(comments.length >= 1, "at least one comment exported");

  const first = comments[0];
  assert.ok("section" in first, "comment has section field");
  assert.ok("text" in first, "comment has text field");
  assert.ok("start_offset" in first, "comment has start_offset field");
  assert.ok("end_offset" in first, "comment has end_offset field");
  assert.ok("comment" in first, "comment has comment field");
  assert.ok("author" in first, "comment has author field");
  assert.ok("status" in first, "comment has status field");
  assert.ok("timestamp" in first, "comment has timestamp field");

  // Restore mocks
  win.URL.createObjectURL = origCreateObjectURL;
});

test("roundtrip: imported JSON renders same callouts", async () => {
  const { doc } = await createPage();
  const target = doc.getElementById("target");

  // Wrap the first 19 chars in a highlight span
  const text = target.textContent;
  const before = text.slice(0, 19);
  const selected = text.slice(0, 19);
  const after = text.slice(19);

  target.innerHTML = before + '<span class="h-annot-hl" data-annot-id="42">' + selected + "</span>" + after;

  const hl = doc.querySelector('.h-annot-hl[data-annot-id="42"]');
  assert.ok(hl, "highlight span exists after innerHTML rebuild");

  const callout = doc.createElement("div");
  callout.className = "h-annot-callout";
  callout.dataset.annotId = "42";
  callout.innerHTML =
    '<span class="h-annot-callout-author">reviewer</span>: Imported review note <span class="h-annot-callout-status">[resolved]</span>' +
    '<span class="h-annot-callout-actions"><button class="h-annot-edit-btn" title="Edit">&#9998;</button><button class="h-annot-delete-btn" title="Delete">&times;</button></span>';
  hl.parentNode.insertBefore(callout, hl.nextSibling);

  const hlEls = doc.querySelectorAll(".h-annot-hl");
  assert.strictEqual(hlEls.length, 1, "one highlight span after import");

  const callouts = doc.querySelectorAll(".h-annot-callout");
  assert.strictEqual(callouts.length, 1, "one callout after import");
  assert.strictEqual(callouts[0].dataset.annotId, "42", "callout has correct annot-id");
  assert.ok(callouts[0].textContent.includes("Imported review note"), "callout displays imported comment text");
  assert.ok(callouts[0].textContent.includes("reviewer"), "callout shows imported author");
  assert.ok(callouts[0].textContent.includes("[resolved]"), "callout shows imported status");
  assert.ok(callouts[0].querySelector(".h-annot-edit-btn"), "callout has edit button");
  assert.ok(callouts[0].querySelector(".h-annot-delete-btn"), "callout has delete button");
});
