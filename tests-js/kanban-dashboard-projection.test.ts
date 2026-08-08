import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import path from 'node:path'

import { test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..')
const DEFAULT_BUNDLE_PATH = path.join(REPO_ROOT, 'plugins', 'kanban', 'dashboard', 'dist', 'index.js')

const BUNDLE_PATH = process.env.KANBAN_DASHBOARD_BUNDLE_PATH || DEFAULT_BUNDLE_PATH

const behavioralProbe = String.raw`
const assert = require('node:assert/strict');
const bundlePath = process.argv[1];

class Component {
  constructor(props) {
    this.props = props;
    this.state = {};
  }
}

let activeRenderer = null;
const Fragment = Symbol('Fragment');
const React = {
  Component,
  Fragment,
  createElement(type, props, ...children) {
    const nextProps = Object.assign({}, props || {});
    if (children.length === 1) nextProps.children = children[0];
    else if (children.length > 1) nextProps.children = children;
    return { type, props: nextProps };
  },
};

function sameDeps(left, right) {
  return !!left && !!right && left.length === right.length &&
    left.every((value, index) => Object.is(value, right[index]));
}

function flatten(items, output) {
  for (const item of items) {
    if (Array.isArray(item)) flatten(item, output);
    else if (item !== null && item !== undefined && item !== false && item !== true) output.push(item);
  }
  return output;
}

function hostNode(type, props, children, existing) {
  if (existing) {
    existing.type = type;
    existing.props = props;
    existing.children = children;
    if (props.ref && typeof props.ref === 'object') props.ref.current = existing;
    if (typeof props.ref === 'function') props.ref(existing);
    return existing;
  }
  const listeners = new Map();
  const node = {
    type,
    props,
    children,
    scrollWidth: 0,
    clientWidth: 0,
    scrollLeft: 0,
    style: {},
    classList: { add() {}, remove() {} },
    addEventListener(name, handler) { listeners.set(name, handler); },
    removeEventListener(name) { listeners.delete(name); },
    hasListener(name) { return listeners.has(name); },
    getBoundingClientRect() { return { bottom: 100 }; },
    closest() { return null; },
    getAttribute(name) { return this.props[name]; },
  };
  if (props.ref && typeof props.ref === 'object') props.ref.current = node;
  if (typeof props.ref === 'function') props.ref(node);
  return node;
}

class Renderer {
  constructor(root) {
    this.root = root;
    this.hooks = new Map();
    this.hosts = new Map();
    this.effects = [];
    this.dirty = true;
    this.tree = null;
    this.currentPath = '';
    this.hookIndex = 0;
  }

  hookSlot(kind, initialValue) {
    const slots = this.hooks.get(this.currentPath) || [];
    this.hooks.set(this.currentPath, slots);
    const index = this.hookIndex++;
    if (!slots[index]) slots[index] = { kind, value: initialValue };
    assert.equal(slots[index].kind, kind, 'hook order changed at ' + this.currentPath);
    return slots[index];
  }

  useState(initialValue) {
    const slot = this.hookSlot(
      'state',
      typeof initialValue === 'function' ? initialValue() : initialValue,
    );
    const renderer = this;
    return [slot.value, function setState(update) {
      const next = typeof update === 'function' ? update(slot.value) : update;
      if (!Object.is(next, slot.value)) {
        slot.value = next;
        renderer.dirty = true;
      }
    }];
  }

  useRef(initialValue) {
    return this.hookSlot('ref', { current: initialValue }).value;
  }

  useMemo(factory, deps) {
    const slot = this.hookSlot('memo', undefined);
    if (!sameDeps(slot.deps, deps)) {
      slot.value = factory();
      slot.deps = deps;
    }
    return slot.value;
  }

  useCallback(callback, deps) {
    return this.useMemo(function () { return callback; }, deps);
  }

  useEffect(effect, deps) {
    const slot = this.hookSlot('effect', undefined);
    if (!sameDeps(slot.deps, deps)) {
      slot.deps = deps;
      this.effects.push(effect);
    }
  }

  renderNode(element, pathName) {
    if (element === null || element === undefined || element === false || element === true) return null;
    if (typeof element === 'string' || typeof element === 'number') return String(element);
    if (Array.isArray(element)) {
      return flatten(element, []).map((child, index) => this.renderNode(child, pathName + '.' + index));
    }
    if (element.type === Fragment) {
      return this.renderNode(element.props.children, pathName + '.fragment');
    }
    if (typeof element.type === 'function') {
      if (element.type.prototype instanceof Component) {
        const instance = new element.type(element.props);
        return this.renderNode(instance.render(), pathName + '.class');
      }
      const previousRenderer = activeRenderer;
      const previousPath = this.currentPath;
      const previousIndex = this.hookIndex;
      activeRenderer = this;
      this.currentPath = pathName;
      this.hookIndex = 0;
      const rendered = element.type(element.props);
      activeRenderer = previousRenderer;
      this.currentPath = previousPath;
      this.hookIndex = previousIndex;
      return this.renderNode(rendered, pathName + '.rendered');
    }
    const props = Object.assign({}, element.props);
    const rawChildren = flatten([props.children], []);
    delete props.children;
    const children = rawChildren.map((child, index) => this.renderNode(child, pathName + '.' + index));
    const existing = this.hosts.get(pathName);
    const node = hostNode(element.type, props, children, existing);
    this.hosts.set(pathName, node);
    return node;
  }

  render() {
    this.effects = [];
    this.dirty = false;
    this.tree = this.renderNode(this.root, 'root');
    const effects = this.effects.slice();
    this.effects = [];
    for (const effect of effects) effect();
  }

  async flush() {
    for (let turn = 0; turn < 30; turn += 1) {
      if (this.dirty) this.render();
      await new Promise((resolve) => setImmediate(resolve));
      if (!this.dirty && this.effects.length === 0) return this.tree;
    }
    throw new Error('renderer did not settle');
  }
}

const hooks = {
  useState(initialValue) { return activeRenderer.useState(initialValue); },
  useRef(initialValue) { return activeRenderer.useRef(initialValue); },
  useMemo(factory, deps) { return activeRenderer.useMemo(factory, deps); },
  useCallback(callback, deps) { return activeRenderer.useCallback(callback, deps); },
  useEffect(effect, deps) { return activeRenderer.useEffect(effect, deps); },
};

function walk(node, visitor) {
  if (node === null || node === undefined) return;
  if (Array.isArray(node)) {
    for (const child of node) walk(child, visitor);
    return;
  }
  if (typeof node === 'string') return;
  visitor(node);
  walk(node.children, visitor);
}

function textContent(node) {
  if (node === null || node === undefined) return '';
  if (Array.isArray(node)) return node.map(textContent).join('');
  if (typeof node === 'string') return node;
  return textContent(node.children);
}

function findAll(root, predicate) {
  const matches = [];
  walk(root, function (node) {
    if (predicate(node)) matches.push(node);
  });
  return matches;
}

function one(root, predicate, description) {
  const matches = findAll(root, predicate);
  assert.equal(matches.length, 1, description + ' (found ' + matches.length + ')');
  return matches[0];
}

function task(id, title, status, diagnostics) {
  return {
    id,
    title,
    status,
    priority: 0,
    assignee: 'kai',
    tenant: null,
    comment_count: 0,
    link_counts: { parents: 0, children: 0 },
    created_at: '2026-07-31T00:00:00Z',
    presentation_diagnostics: diagnostics || [],
  };
}

const canonicalBoard = {
  columns: [{
    name: 'todo',
    tasks: [task('task-canonical', 'Canonical task', 'todo')],
  }],
  tenants: [],
  assignees: ['kai'],
  latest_event_id: 1,
};

const projectedBoard = {
  columns: [{
    name: 'queue-view',
    label: 'Curated Queue',
    helper: 'Tasks selected by the presentation policy',
    read_only: true,
    tasks: [task('task-projected', 'Projected task', 'todo', [{
      kind: 'malformed_needs_input',
      message: 'Needs-input metadata is malformed',
    }])],
  }],
  tenants: [],
  assignees: ['kai'],
  latest_event_id: 2,
  presentation: {
    mode: 'projection',
    error: 'projection schema is invalid',
  },
};

let boardResponse = canonicalBoard;
const sdk = {
  React,
  components: {
    Card: 'section',
    CardContent: 'div',
    Badge: 'span',
    Button: 'button',
    Input: 'input',
    Label: 'label',
    Select: 'select',
    SelectOption: 'option',
    Checkbox: 'input',
  },
  hooks,
  utils: {
    cn: (...values) => values.filter(Boolean).join(' '),
    timeAgo: () => 'now',
  },
  useI18n: () => ({ t: { kanban: null }, locale: 'en' }),
  fetchJSON(url) {
    if (url.includes('/config')) return Promise.resolve({ render_markdown: false });
    if (url.includes('/boards')) return Promise.resolve({
      boards: [{ slug: 'default', name: 'Default' }],
      current: 'default',
    });
    if (/\/board(?:\?|$)/.test(url)) return Promise.resolve(boardResponse);
    if (url.includes('/profiles')) return Promise.resolve({ profiles: [] });
    if (url.includes('/orchestration')) return Promise.resolve({ mode: 'manual' });
    return Promise.resolve({});
  },
  buildWsUrl: () => Promise.resolve('ws://example.invalid/events'),
};

const localStorage = {
  getItem() { return 'default'; },
  setItem() {},
  removeItem() {},
};
const eventTarget = { addEventListener() {}, removeEventListener() {} };
global.window = Object.assign({}, eventTarget, {
  __HERMES_PLUGIN_SDK__: sdk,
  __HERMES_PLUGINS__: {
    register(name, page) {
      assert.equal(name, 'kanban');
      global.KanbanPage = page;
    },
  },
  localStorage,
  confirm: () => true,
  prompt: () => 'done',
  alert() {},
  location: { href: 'http://localhost/' },
});
global.document = Object.assign({}, eventTarget, {
  querySelectorAll: () => [],
  createElement: () => hostNode('div', {}, []),
  body: { appendChild() {}, removeChild() {} },
});
global.ResizeObserver = class ResizeObserver {
  observe() {}
  disconnect() {}
};
global.WebSocket = class WebSocket {
  close() {}
};
global.requestAnimationFrame = (callback) => callback();

require(bundlePath);
assert.equal(typeof global.KanbanPage, 'function', 'bundle must register the Kanban page');

async function renderBoard(board) {
  boardResponse = board;
  const renderer = new Renderer(React.createElement(global.KanbanPage));
  renderer.render();
  await renderer.flush();
  return renderer;
}

(async function main() {
  const canonical = await renderBoard(canonicalBoard);
  let tree = canonical.tree;
  const writableColumn = one(
    tree,
    (node) => node.props['data-kanban-column'] === 'todo',
    'canonical column should render',
  );
  assert.equal(typeof writableColumn.props.onDragOver, 'function');
  assert.equal(typeof writableColumn.props.onDrop, 'function');
  assert.equal(
    writableColumn.hasListener('hermes-kanban:drop'),
    true,
    'canonical column should accept the touch-drop event',
  );

  const createButton = one(
    tree,
    (node) => node.type === 'button' && node.props.title === 'Create task in this column',
    'canonical column should expose its create-task control',
  );
  assert.ok(findAll(tree, (node) => String(node.props['aria-label'] || '').startsWith('Select all tasks in ')).length > 0);
  assert.ok(findAll(tree, (node) => node.props['aria-label'] === 'Select task task-canonical').length > 0);

  const writableCard = one(
    tree,
    (node) => node.props['data-task-id'] === 'task-canonical',
    'canonical task card should render',
  );
  assert.equal(writableCard.props.draggable, true);
  assert.equal(typeof writableCard.props.onDragStart, 'function');
  assert.equal(
    writableCard.hasListener('pointerdown'),
    true,
    'canonical card should register touch dragging',
  );
  assert.equal(
    findAll(tree, (node) => node.props['data-kanban-trash'] === 'true').length,
    1,
    'canonical board should render its trash drop zone',
  );

  createButton.props.onClick();
  await canonical.flush();
  tree = canonical.tree;
  assert.ok(
    findAll(tree, (node) => node.type === 'form' && textContent(node).includes('New task')).length > 0,
    'canonical create control should open the task form',
  );

  const projected = await renderBoard(projectedBoard);
  tree = projected.tree;
  const readOnlyColumn = one(
    tree,
    (node) => node.props['data-kanban-column'] === 'queue-view',
    'projected column should render',
  );
  assert.ok(textContent(readOnlyColumn).includes('Curated Queue'));
  assert.ok(textContent(readOnlyColumn).includes('Tasks selected by the presentation policy'));
  assert.equal(
    findAll(readOnlyColumn, (node) => node.props.title === 'Tasks selected by the presentation policy').length,
    1,
    'projected helper should also label the column header',
  );
  assert.equal(typeof readOnlyColumn.props.onDragOver, 'undefined');
  assert.equal(typeof readOnlyColumn.props.onDrop, 'undefined');
  assert.equal(
    readOnlyColumn.hasListener('hermes-kanban:drop'),
    false,
    'projected column must not accept the touch-drop event',
  );
  assert.equal(
    findAll(readOnlyColumn, (node) => node.props.title === 'Create task in this column').length,
    0,
  );
  assert.equal(
    findAll(readOnlyColumn, (node) => String(node.props['aria-label'] || '').startsWith('Select ')).length,
    0,
    'projected columns should suppress column and task checkboxes',
  );

  const readOnlyCard = one(
    tree,
    (node) => node.props['data-task-id'] === 'task-projected',
    'projected task card should render',
  );
  assert.equal(readOnlyCard.props.draggable, false);
  assert.equal(typeof readOnlyCard.props.onDragStart, 'undefined');
  assert.equal(
    readOnlyCard.hasListener('pointerdown'),
    false,
    'projected card must not register touch dragging',
  );
  assert.equal(findAll(tree, (node) => node.props['data-kanban-trash'] === 'true').length, 0);

  const taskAlert = one(
    tree,
    (node) => node.props.role === 'alert' && textContent(node).includes('Needs-input metadata is malformed'),
    'task presentation diagnostic should render as an alert',
  );
  assert.equal(taskAlert.props.title, 'malformed_needs_input: Needs-input metadata is malformed');

  const configAlert = one(
    tree,
    (node) => node.props.role === 'alert' && textContent(node).includes('Board presentation configuration is invalid:'),
    'presentation configuration error should render as an alert',
  );
  assert.ok(textContent(configAlert).includes('projection schema is invalid'));

  process.exit(0);
})().catch((error) => {
  console.error(error && error.stack ? error.stack : error);
  process.exit(1);
});
`

test('kanban bundle renders canonical controls and enforces projected read-only diagnostics', () => {
  const child = spawnSync(process.execPath, ['-e', behavioralProbe, BUNDLE_PATH], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    timeout: 30_000,
  })

  assert.equal(
    child.status,
    0,
    `behavioral bundle probe failed\nstdout:\n${child.stdout}\nstderr:\n${child.stderr}`,
  )
})
