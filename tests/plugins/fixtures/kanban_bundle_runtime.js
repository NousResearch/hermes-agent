// Mounted-runtime harness for the Kanban dashboard bundle. The bundle has no build step, so it
// IS the program: this EXECUTES it (require) against a minimal React-compatible host (function
// components + the five hooks the plugin SDK exposes, class components) and a stub SDK whose
// GET responses are real API payloads the Python test captured. It then drives the UI the way
// a user does (click "+", type, submit; open a card, edit the due date, Save / Clear) and prints
// every mutating request the bundle sent as JSON. The bundle's source text is never inspected.
//
//   node kanban_bundle_runtime.js <bundle.js> <responses.json> <scenario>
//   scenario ("|"-separated, since datetime values contain ":"):
//     create|<column>|<title>|<datetime-local>   edit|<task id>|<datetime-local>
//     clear|<task id>                            empty-save|<task id>
"use strict";

const fs = require("fs");

const [bundlePath, responsesPath, scenario] = process.argv.slice(2);
const responses = JSON.parse(fs.readFileSync(responsesPath, "utf8"));
const requests = [];
const errors = [];

// ---------------------------------------------------------------- tiny React host ----------
function flatten(children, out) {
  for (const c of children) {
    if (Array.isArray(c)) flatten(c, out);
    else if (c !== null && c !== undefined && c !== false && c !== true) out.push(c);
  }
  return out;
}

function createElement(type, props, ...children) {
  const p = Object.assign({}, props || {});
  const flat = flatten(children, []);
  if (flat.length) p.children = flat.length === 1 ? flat[0] : flat;
  return { $el: true, type, key: p.key, props: p };
}

class Component {
  constructor(props) { this.props = props; this.state = {}; }
  setState(update) {
    Object.assign(this.state, typeof update === "function" ? update(this.state) : update);
    scheduleRender();
  }
}

const instances = new Map();   // path -> { hooks: [], inst }
let seen = new Set();
let current = null;            // { slot, i }
let pendingEffects = [];
let dirty = false;

function depsChanged(a, b) {
  return !a || !b || a.length !== b.length || a.some((v, i) => !Object.is(v, b[i]));
}

function hookSlot(init) {
  const i = current.i++;
  if (current.slot.hooks.length <= i) current.slot.hooks.push(init());
  return current.slot.hooks[i];
}

const hooks = {
  useState(initial) {
    const h = hookSlot(() => ({ value: typeof initial === "function" ? initial() : initial }));
    if (!h.set) {
      h.set = (v) => { h.value = typeof v === "function" ? v(h.value) : v; scheduleRender(); };
    }
    return [h.value, h.set];
  },
  useRef(initial) { return hookSlot(() => ({ current: initial })); },
  useMemo(fn, deps) {
    const h = hookSlot(() => ({ deps: undefined, value: undefined, init: false }));
    if (!h.init || depsChanged(h.deps, deps)) { h.value = fn(); h.deps = deps; h.init = true; }
    return h.value;
  },
  useCallback(fn, deps) { return hooks.useMemo(() => fn, deps); },
  useEffect(fn, deps) {
    const h = hookSlot(() => ({ deps: undefined, cleanup: null, init: false }));
    if (!h.init || depsChanged(h.deps, deps)) {
      h.init = true;
      h.deps = deps;
      pendingEffects.push(() => {
        if (h.cleanup) h.cleanup();
        const c = fn();
        h.cleanup = typeof c === "function" ? c : null;
      });
    }
  },
};

function renderNode(node, path) {
  if (node === null || node === undefined || typeof node === "boolean") return null;
  if (typeof node === "string" || typeof node === "number") return { text: String(node) };
  if (Array.isArray(node)) {
    return { type: "#fragment", props: {}, children: node.map((c, i) => renderNode(c, `${path}/${c && c.key != null ? "k" + c.key : i}`)) };
  }
  const { type, props } = node;
  if (typeof type === "string") {
    const kids = props.children === undefined ? [] : [].concat(props.children);
    return { type, props, children: kids.map((c, i) => renderNode(c, `${path}/${c && c.key != null ? "k" + c.key : i}`)) };
  }
  const id = `${path}:${type.name || "anon"}`;
  seen.add(id);
  if (!instances.has(id)) instances.set(id, { hooks: [] });
  const slot = instances.get(id);
  if (type.prototype && type.prototype.render) {
    if (!slot.inst) slot.inst = new type(props);
    slot.inst.props = props;
    return renderNode(slot.inst.render(), id);
  }
  const prev = current;
  current = { slot, i: 0 };
  try { return renderNode(type(props), id); } finally { current = prev; }
}

let Page = null;
let tree = null;
function renderAll() {
  dirty = false;
  seen = new Set();
  tree = renderNode(createElement(Page, null), "root");
  for (const [id, slot] of instances) {
    if (seen.has(id)) continue;
    for (const h of slot.hooks) if (h && h.cleanup) h.cleanup();
    instances.delete(id);
  }
  const effects = pendingEffects;
  pendingEffects = [];
  for (const run of effects) {
    try { run(); } catch (e) { errors.push(String(e && e.stack || e)); }
  }
}
function scheduleRender() { dirty = true; }

async function settle() {
  for (let round = 0; round < 200; round++) {
    await new Promise((r) => setImmediate(r));
    if (dirty) renderAll();
    else if (round > 5) return;
  }
  throw new Error("UI never settled");
}

function findAll(pred, node = tree, out = []) {
  if (!node || node.text !== undefined) return out;
  if (pred(node)) out.push(node);
  for (const c of node.children || []) findAll(pred, c, out);
  return out;
}
function textOf(node) {
  if (!node) return "";
  if (node.text !== undefined) return node.text;
  return (node.children || []).map(textOf).join("");
}
function only(list, what) {
  if (list.length !== 1) throw new Error(`expected exactly one ${what}, found ${list.length}`);
  return list[0];
}
const cls = (node) => String(node.props.className || "");

// ---------------------------------------------------------------- host + stub SDK ----------
function pathOf(url) { return url.split("?")[0]; }

const SDK = {
  React: { createElement, Component, ...hooks },
  hooks,
  components: Object.fromEntries(
    ["Card", "CardContent", "Badge", "Button", "Input", "Label", "Select", "SelectOption"].map((n) => [n, n])),
  utils: { cn: (...a) => a.filter(Boolean).join(" "), timeAgo: () => "" },
  fetchJSON(url, opts) {
    const method = (opts && opts.method) || "GET";
    if (method !== "GET") {
      requests.push({ method, path: pathOf(url), body: opts.body ? JSON.parse(opts.body) : null });
      return Promise.resolve(responses.mutation || {});
    }
    const hit = responses.get[pathOf(url)];
    return hit === undefined
      ? Promise.reject(new Error(`404: ${pathOf(url)}`))
      : Promise.resolve(JSON.parse(JSON.stringify(hit)));
  },
  authedFetch: () => Promise.reject(new Error("not used")),
  buildWsUrl: () => new Promise(() => {}),  // live updates are out of scope: never connect
};

const noop = () => {};
global.window = {
  __HERMES_PLUGIN_SDK__: SDK,
  __HERMES_PLUGINS__: { register: (_name, component) => { Page = component; } },
  localStorage: { getItem: () => null, setItem: noop, removeItem: noop },
  addEventListener: noop, removeEventListener: noop,
  confirm: () => true, prompt: () => null, alert: noop,
};
global.document = {
  body: { appendChild: noop, removeChild: noop },
  addEventListener: noop, removeEventListener: noop,
  createElement: () => ({ style: {}, setAttribute: noop, appendChild: noop, remove: noop }),
  querySelector: () => null, querySelectorAll: () => [], elementFromPoint: () => null,
};
global.requestAnimationFrame = (fn) => setTimeout(fn, 0);

// ---------------------------------------------------------------- scenarios ----------------
const evt = (extra) => Object.assign({ preventDefault: noop, stopPropagation: noop }, extra);

async function openCard(taskId) {
  const card = only(findAll((n) => cls(n).split(" ").includes("hermes-kanban-card")
    && String(n.props["aria-label"] || "").includes(taskId)), "card");
  card.props.onClick(evt({ shiftKey: false, ctrlKey: false, metaKey: false }));
  await settle();
}

function dueRow() {
  return only(findAll((n) => cls(n).includes("hermes-kanban-meta-row")
    && n.children.length > 0 && textOf(n.children[0]) === "Due"), "Due row");
}

async function editDue(taskId, value, button) {
  await openCard(taskId);
  const row = dueRow();
  only(findAll((n) => cls(n).includes("hermes-kanban-editable"), row), "due value").props.onClick(evt());
  await settle();
  if (value !== null) {
    only(findAll((n) => n.type === "Input" && n.props.type === "datetime-local", dueRow()), "due input")
      .props.onChange({ target: { value } });
    await settle();
  }
  only(findAll((n) => n.type === "Button" && textOf(n) === button, dueRow()), `${button} button`).props.onClick(evt());
  await settle();
}

async function create(column, title, due) {
  const columns = responses.get["/api/plugins/kanban/board"].columns.map((c) => c.name);
  const adds = findAll((n) => n.type === "button" && cls(n).includes("hermes-kanban-column-add"));
  if (adds.length !== columns.length) throw new Error(`expected ${columns.length} column add buttons, found ${adds.length}`);
  adds[columns.indexOf(column)].props.onClick(evt());
  await settle();
  const form = only(findAll((n) => n.type === "form" && cls(n).includes("hermes-kanban-create-dialog")), "create form");
  only(findAll((n) => n.type === "textarea", form), "title box").props.onChange({ target: { value: title } });
  await settle();
  only(findAll((n) => n.type === "Input" && n.props.type === "datetime-local"), "due input").props.onChange({ target: { value: due } });
  await settle();
  only(findAll((n) => n.type === "form" && cls(n).includes("hermes-kanban-create-dialog")), "create form")
    .props.onSubmit(evt());
  await settle();
}

async function main() {
  require(bundlePath);
  if (!Page) throw new Error("bundle did not register a page");
  renderAll();
  await settle();
  const [kind, ...args] = scenario.split("|");
  if (kind === "create") await create(args[0], args[1], args[2]);
  else if (kind === "edit") await editDue(args[0], args[1], "Save");
  else if (kind === "empty-save") await editDue(args[0], "", "Save");
  else if (kind === "clear") await editDue(args[0], null, "Clear");
  else throw new Error(`unknown scenario ${kind}`);
}

main()
  .catch((e) => errors.push(String(e && e.stack || e)))
  .finally(() => {
    process.stdout.write(JSON.stringify({ requests, errors }));
    process.exit(0);
  });
