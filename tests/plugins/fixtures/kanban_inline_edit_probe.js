// Behavioral probe for the drawer meta-row inline editors (#125714): extracts
// AssigneeEditor / PriorityEditor / ModelEditor from the shipped dashboard bundle
// (no build step — the bundle IS the source) and drives them through real
// render/dispatch cycles with a minimal hooks runtime. A click-away (blur without
// Enter) must commit a changed value, an unchanged or empty field must just close
// the row, and Escape keeps cancel semantics — no patch, and no double patch when
// blur follows Enter. Exits non-zero on the first failure. Run via:
//   node kanban_inline_edit_probe.js <path-to-bundle>
const fs = require("fs");

const bundlePath = process.argv[2];
const src = fs.readFileSync(bundlePath, "utf8");

function extractFn(name) {
  const start = src.indexOf(`function ${name}(`);
  if (start === -1) throw new Error(`${name} not found in bundle`);
  const bodyStart = src.indexOf("{", start);
  let depth = 0, end = bodyStart;
  for (; end < src.length; end++) {
    if (src[end] === "{") depth++;
    else if (src[end] === "}") { depth--; if (depth === 0) break; }
  }
  if (src.slice(start, end + 1).includes(`function ${name}(`) === false) throw new Error("brace scan error");
  return src.slice(start, end + 1);
}

function makeHooks() {
  const slots = [];
  let cursor = 0;
  return {
    reset() { cursor = 0; },
    useState(initial) {
      const idx = cursor++;
      if (!(idx in slots)) slots[idx] = typeof initial === "function" ? initial() : initial;
      return [slots[idx], (v) => { slots[idx] = typeof v === "function" ? v(slots[idx]) : v; }];
    },
    useEffect() { /* not needed: the probe seeds catalog state directly */ },
    useRef(initial) {
      const idx = cursor++;
      if (!(idx in slots)) slots[idx] = { current: initial };
      return slots[idx];
    },
  };
}

const h = function (type, props) {
  return { type, props: props || {}, children: Array.prototype.slice.call(arguments, 2) };
};
const useI18n = () => ({ t: {} });
const tx = (t, key, fallback) => fallback;
const cn = (...xs) => xs.filter(Boolean).join(" ");
const Input = "input";

function bind(fnName, hooks, extraBindings) {
  const parts = ["h", "useState", "useEffect", "useRef", "useI18n", "tx", "cn", "Input"];
  const vals = [h, hooks.useState, hooks.useEffect, hooks.useRef, useI18n, tx, cn, Input];
  for (const k of Object.keys(extraBindings || {})) { parts.push(k); vals.push(extraBindings[k]); }
  const factory = new Function(...parts, `${fnName}Src`, `"use strict";\n${extractFn(fnName)}\nreturn ${fnName};`);
  return factory(...vals, extractFn(fnName));
}

// A synchronous thenable so .then(setEditing(false)) applies before the next assert.
const syncThen = () => ({ then(cb) { cb(); return this; }, catch() { return this; } });

let failures = 0;
function check(label, cond, detail) {
  if (cond) return;
  console.error(`FAIL: ${label}${detail ? " — " + detail : ""}`);
  failures++;
}

// Edit renders: div > [label span, value span (onClick opens)] — editing renders
// div > [label span, Input]. Detect the row kind by the second child.
const secondChild = (row) => row.children[1];

// Hooks state lives in per-scenario slots; the cursor must reset before every
// render so each call re-reads the same slots instead of allocating new ones.
// React re-renders on every setState; the probe mirrors that by re-rendering
// after every change so handlers always see the current state.
function openEditor(Component, hooks, props) {
  hooks.reset();
  const staticRow = Component(props);
  secondChild(staticRow).props.onClick();
  hooks.reset();
  return Component(props); // editing mode
}

function rerender(Component, hooks, props) {
  hooks.reset();
  return Component(props);
}

// Simulate typing: change event, then the re-render React would schedule, then
// hand back the fresh input element.
function typeInto(Component, hooks, props, value) {
  hooks.reset();
  const cur = Component(props);
  secondChild(cur).props.onChange({ target: { value } });
  return rerender(Component, hooks, props);
}

function scenario(name, fn) {
  try {
    fn();
    console.log(`ok - ${name}`);
  } catch (e) {
    console.error(`FAIL: ${name} — ${e.message}`);
    failures++;
  }
}

// ---------------- AssigneeEditor ----------------
scenario("assignee: blur commits a changed value", () => {
  const hooks = makeHooks();
  const AssigneeEditor = bind("AssigneeEditor", hooks);
  const patches = [];
  const props = { task: { assignee: null }, onPatch: (p) => { patches.push(p); return syncThen(); } };
  openEditor(AssigneeEditor, hooks, props);
  const input = secondChild(typeInto(AssigneeEditor, hooks, props, "default"));
  input.props.onBlur();
  check("patch issued on blur", JSON.stringify(patches) === '[{"assignee":"default"}]', JSON.stringify(patches));
  const after = rerender(AssigneeEditor, hooks, props);
  check("row closed after blur commit", secondChild(after).props.onClick !== undefined, "still editing");
});

scenario("assignee: blur with unchanged value closes without patch", () => {
  const hooks = makeHooks();
  const AssigneeEditor = bind("AssigneeEditor", hooks);
  const patches = [];
  const props = { task: { assignee: "ann" }, onPatch: (p) => { patches.push(p); return syncThen(); } };
  const editing = openEditor(AssigneeEditor, hooks, props);
  secondChild(editing).props.onBlur();
  check("no patch when unchanged", patches.length === 0, JSON.stringify(patches));
});

scenario("assignee: blur with empty value closes without unassigning", () => {
  const hooks = makeHooks();
  const AssigneeEditor = bind("AssigneeEditor", hooks);
  const patches = [];
  const props = { task: { assignee: "ann" }, onPatch: (p) => { patches.push(p); return syncThen(); } };
  openEditor(AssigneeEditor, hooks, props);
  const input = secondChild(typeInto(AssigneeEditor, hooks, props, ""));
  input.props.onBlur();
  check("no unassign patch on empty blur", patches.length === 0, JSON.stringify(patches));
});

scenario("assignee: Escape then blur never patches", () => {
  const hooks = makeHooks();
  const AssigneeEditor = bind("AssigneeEditor", hooks);
  const patches = [];
  const props = { task: { assignee: null }, onPatch: (p) => { patches.push(p); return syncThen(); } };
  openEditor(AssigneeEditor, hooks, props);
  const input = secondChild(typeInto(AssigneeEditor, hooks, props, "default"));
  input.props.onKeyDown({ key: "Escape", preventDefault() {} });
  input.props.onBlur();
  check("no patch after escape+blur", patches.length === 0, JSON.stringify(patches));
});

scenario("assignee: Enter then blur does not double-patch", () => {
  const hooks = makeHooks();
  const AssigneeEditor = bind("AssigneeEditor", hooks);
  const patches = [];
  const props = { task: { assignee: null }, onPatch: (p) => { patches.push(p); return syncThen(); } };
  openEditor(AssigneeEditor, hooks, props);
  const input = secondChild(typeInto(AssigneeEditor, hooks, props, "default"));
  input.props.onKeyDown({ key: "Enter", preventDefault() {} });
  input.props.onBlur();
  check("single patch for Enter+blur", patches.length === 1 && patches[0].assignee === "default", JSON.stringify(patches));
});

// ---------------- PriorityEditor ----------------
scenario("priority: blur commits a changed value", () => {
  const hooks = makeHooks();
  const PriorityEditor = bind("PriorityEditor", hooks);
  const patches = [];
  const props = { task: { priority: 0 }, onPatch: (p) => { patches.push(p); return syncThen(); } };
  openEditor(PriorityEditor, hooks, props);
  const input = secondChild(typeInto(PriorityEditor, hooks, props, "3"));
  input.props.onBlur();
  check("priority patch on blur", JSON.stringify(patches) === '[{"priority":3}]', JSON.stringify(patches));
});

scenario("priority: blur with unchanged value closes without patch", () => {
  const hooks = makeHooks();
  const PriorityEditor = bind("PriorityEditor", hooks);
  const patches = [];
  const props = { task: { priority: 2 }, onPatch: (p) => { patches.push(p); return syncThen(); } };
  const editing = openEditor(PriorityEditor, hooks, props);
  secondChild(editing).props.onBlur();
  check("no patch when unchanged", patches.length === 0, JSON.stringify(patches));
});

// ---------------- ModelEditor (free-text fallback) ----------------
// Seed the catalog cache so editing opens straight into the free-text branch.
scenario("model free-text: blur commits a changed model", () => {
  const hooks = makeHooks();
  const ModelEditor = bind("ModelEditor", hooks, {
    _modelCatalogCache: { providers: [] },
    fetchModelCatalog: () => Promise.resolve({ providers: [] }),
  });
  const patches = [];
  const props = { task: { model_override: null, provider_override: null }, onPatch: (p) => { patches.push(p); return syncThen(); } };
  openEditor(ModelEditor, hooks, props);
  hooks.reset();
  const fresh = ModelEditor(props);
  check("free-text branch reached", secondChild(fresh).props.onBlur !== undefined && secondChild(fresh).props.placeholder !== undefined, "editing row is not the free-text input");
  const input = secondChild(typeInto(ModelEditor, hooks, props, "gpt-4o"));
  input.props.onBlur();
  check("model patch on blur", JSON.stringify(patches) === '[{"model_override":"gpt-4o"}]', JSON.stringify(patches));
});

scenario("model free-text: blur with empty value closes without clearing override", () => {
  const hooks = makeHooks();
  const ModelEditor = bind("ModelEditor", hooks, {
    _modelCatalogCache: { providers: [] },
    fetchModelCatalog: () => Promise.resolve({ providers: [] }),
  });
  const patches = [];
  const props = { task: { model_override: "qwen3", provider_override: null }, onPatch: (p) => { patches.push(p); return syncThen(); } };
  const editing = openEditor(ModelEditor, hooks, props);
  const input = secondChild(editing);
  input.props.onBlur();
  check("no clear patch on empty blur", patches.length === 0, JSON.stringify(patches));
});

scenario("model free-text: Escape then blur never patches", () => {
  const hooks = makeHooks();
  const ModelEditor = bind("ModelEditor", hooks, {
    _modelCatalogCache: { providers: [] },
    fetchModelCatalog: () => Promise.resolve({ providers: [] }),
  });
  const patches = [];
  const props = { task: { model_override: null, provider_override: null }, onPatch: (p) => { patches.push(p); return syncThen(); } };
  openEditor(ModelEditor, hooks, props);
  const input = secondChild(typeInto(ModelEditor, hooks, props, "gpt-4o"));
  input.props.onKeyDown({ key: "Escape", preventDefault() {} });
  input.props.onBlur();
  check("no patch after escape+blur", patches.length === 0, JSON.stringify(patches));
});

if (failures > 0) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
console.log("PASS");
