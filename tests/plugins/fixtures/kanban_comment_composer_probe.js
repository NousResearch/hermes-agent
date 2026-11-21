// Behavioral regression for Kanban TaskDrawer's multiline comment composer.
// Execute the real shipped component with inert React-shaped hooks, then inspect
// its element tree and the matching plugin stylesheet.
// Usage: node kanban_comment_composer_probe.js <index.js> <style.css>
const fs = require("fs");

const bundle = fs.readFileSync(process.argv[2], "utf8");
const styles = fs.readFileSync(process.argv[3], "utf8");
const start = bundle.indexOf("function TaskDrawer(");
if (start < 0) throw new Error("TaskDrawer not found");
const opening = bundle.indexOf("{", start);
let depth = 0;
let end = -1;
for (let i = opening; i < bundle.length; i++) {
  if (bundle[i] === "{") depth++;
  if (bundle[i] === "}" && --depth === 0) {
    end = i;
    break;
  }
}
if (end < 0) throw new Error("TaskDrawer body not terminated");
const functionText = bundle.slice(start, end + 1);

const h = (type, props, ...children) => ({ type, props: props || {}, children });
const tx = (_translations, _key, fallback) => fallback;
const useI18n = () => ({ t: {} });
const useCallback = fn => fn;
const useEffect = () => {};
const comment = "First paragraph\nSecond paragraph";
let hook = 0;
const useState = initial => {
  const index = hook++;
  // Existing TaskDrawer hook order: data, loading, error, patchError,
  // newComment. Effects are deliberately inert in this isolated rendering.
  const state = index === 0 ? { task: { status: "ready" } }
    : index === 1 ? false : index === 4 ? comment : initial;
  return [state, () => {}];
};
const posts = [];
const SDK = {
  fetchJSON(url, options) {
    posts.push({ url, options });
    return Promise.resolve({});
  },
};
const withBoard = url => url;
function TaskDetail() {}
function Button() {}
const TaskDrawer = new Function(
  "h", "tx", "useI18n", "useCallback", "useEffect", "useState",
  "SDK", "withBoard", "API", "TaskDetail", "Button",
  functionText + "\nreturn TaskDrawer;",
)(h, tx, useI18n, useCallback, useEffect, useState,
  SDK, withBoard, "/api/plugins/kanban", TaskDetail, Button);

const failures = [];
const check = (name, ok) => { if (!ok) failures.push(name); };
function findByClass(element, cls) {
  if (!element || typeof element !== "object") return null;
  if (Array.isArray(element)) {
    for (const child of element) {
      const found = findByClass(child, cls);
      if (found) return found;
    }
    return null;
  }
  if ((element.props || {}).className === cls) return element;
  return findByClass(element.children || [], cls);
}
const drawer = TaskDrawer({
  taskId: "K-1", boardSlug: "default", onClose() {}, onRefresh() {},
});
const row = findByClass(drawer, "hermes-kanban-drawer-comment-row");
const editor = row && row.children.find(child => child && child.type === "textarea");
check("native multiline textarea", Boolean(editor));
if (editor) {
  check("three visible rows", editor.props.rows === 3);
  check("multiline draft preserved", editor.props.value === comment);
  check("style class applied", editor.props.className.split(/\s+/).includes("hermes-kanban-comment-input"));

  let prevented = false;
  editor.props.onKeyDown({
    key: "Enter", shiftKey: true, preventDefault() { prevented = true; },
  });
  check("Shift+Enter leaves native newline enabled", !prevented && posts.length === 0);

  editor.props.onKeyDown({
    key: "Enter", shiftKey: false, preventDefault() { prevented = true; },
  });
  check("Enter submits once", prevented && posts.length === 1);
  if (posts.length === 1) {
    check("multiline content sent intact",
      posts[0].options.method === "POST" &&
      JSON.parse(posts[0].options.body).body === comment);
  }
}

function styleBlock(selector) {
  const start = styles.indexOf(selector + " {");
  if (start < 0) return "";
  const opening = styles.indexOf("{", start);
  const closing = styles.indexOf("}", opening);
  return styles.slice(opening + 1, closing);
}
const editorCSS = styleBlock(".hermes-kanban-comment-input");
const rowCSS = styleBlock(".hermes-kanban-drawer-comment-row");
check("comment editor has vertical resize", editorCSS.includes("resize: vertical;"));
check("comment editor has height cap", editorCSS.includes("max-height: 40vh;"));
check("comment editor scrolls beyond cap", editorCSS.includes("overflow-y: auto;"));
check("send button sits on bottom edge", rowCSS.includes("align-items: flex-end;"));
if (failures.length) {
  console.error("FAIL\n- " + failures.join("\n- "));
  process.exit(1);
}
console.log("PASS");
