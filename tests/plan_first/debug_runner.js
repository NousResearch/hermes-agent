const { JSDOM } = require("jsdom");
const fs = require("fs");
const path = require("path");

const jsCode = fs.readFileSync(path.join(__dirname, "..", "..", "plan-annotate.js"), "utf-8");
const cssCode = fs.readFileSync(path.join(__dirname, "..", "..", "plan-annotate.css"), "utf-8");

const html = `<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>T</title><style>${cssCode}</style></head>
<body>
<header><h1>Plan</h1></header>
<p id="t">hello world</p>
<script>${jsCode}</script>
</body>
</html>`;

const dom = new JSDOM(html, { url: "http://localhost", runScripts: "dangerously" });
const doc = dom.window.document;
const win = dom.window;

console.log("SYNC check - readyState:", doc.readyState);
console.log("SYNC check - Export button:", !!doc.querySelector(".h-annot-export-btn"));
console.log("SYNC check - body listeners:", typeof doc.body.onmouseup);

// Check again after a tick
setTimeout(() => {
  console.log("\nASYNC check (tick) - readyState:", doc.readyState);
  console.log("ASYNC check - Export button:", !!doc.querySelector(".h-annot-export-btn"));

  // Test mouseup
  const p = doc.getElementById("t");
  const range = doc.createRange();
  range.setStart(p.firstChild, 0);
  range.setEnd(p.firstChild, 5);
  win.getSelection().removeAllRanges();
  win.getSelection().addRange(range);
  console.log("Selected:", JSON.stringify(win.getSelection().toString()));
  win.dispatchEvent(new win.MouseEvent("mouseup", { bubbles: true }));
  console.log("Box:", !!doc.querySelector(".h-annot-box"));

  process.exit(0);
}, 50);
