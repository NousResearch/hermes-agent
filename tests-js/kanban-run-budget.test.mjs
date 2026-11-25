// @vitest-environment jsdom
import { afterEach, expect, test } from 'vitest';
import * as React from 'react';
import { createRoot } from 'react-dom/client';

const { document, window, Event, MouseEvent, KeyboardEvent, HTMLInputElement, HTMLTextAreaElement } = globalThis;
let root;
afterEach(async () => {
  if (root) await React.act(async () => root.unmount());
  document.body.replaceChildren();
  delete window.__HERMES_PLUGIN_SDK__;
  delete window.__HERMES_PLUGINS__;
});

// Executes the whole shipped plugin, not extracted functions or source assertions.
test('dashboard creates, raises and clears absolute worker-run caps', async () => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const writes = [];
  const task = { id: 't_fixture', title: 'Capped fixture', status: 'blocked',
    priority: 0, workspace_kind: 'scratch', max_worker_runs: 1, worker_runs_used: 1,
    age: {}, diagnostics: [] };
  let Page;
  const primitive = (tag) => function Component(raw) {
    const props = { ...raw };
    for (const key of ['children', 'variant', 'size', 'onValueChange']) delete props[key];
    return React.createElement(tag, { ...props,
      ...(raw.onValueChange ? { onChange: (event) => raw.onValueChange(event.target.value) } : {}) }, raw.children);
  };
  window.__HERMES_PLUGIN_SDK__ = {
    React, hooks: React,
    components: { Card: primitive('div'), CardContent: primitive('div'), Badge: primitive('span'),
      Button: primitive('button'), Input: primitive('input'), Label: primitive('label'),
      Select: primitive('select'), SelectOption: primitive('option') },
    utils: { cn: (...parts) => parts.filter(Boolean).join(' '), timeAgo: () => 'now' },
    buildWsUrl: async () => 'ws://localhost/test-only',
    fetchJSON: async (url, options = {}) => {
      const path = new URL(url, 'http://localhost').pathname;
      if (options.method === 'PATCH') {
        const payload = JSON.parse(options.body); writes.push(payload);
        Object.assign(task, payload);
        if (payload.clear_max_worker_runs) task.max_worker_runs = null;
        return { task };
      }
      if (options.method === 'POST' && path.endsWith('/tasks')) {
        writes.push(JSON.parse(options.body)); return { task };
      }
      if (path.endsWith('/config')) return { lane_by_profile: false };
      if (path.endsWith('/boards')) return { current: 'default', boards: [{ slug: 'default', name: 'Fixture' }] };
      if (path.endsWith('/board')) return { columns: [{ name: 'blocked', tasks: [task] }],
        tenants: [], assignees: [], counts: { blocked: 1 } };
      if (path.endsWith('/tasks/t_fixture')) return { task, comments: [], events: [], runs: [],
        attachments: [], links: { parents: [], children: [] }, child_results: [] };
      return {};
    },
  };
  window.__HERMES_PLUGINS__ = { register: (_, component) => { Page = component; } };
  window.WebSocket = class { close() {} };
  await import('../plugins/kanban/dashboard/dist/index.js');
  const container = document.createElement('div'); document.body.append(container);
  root = createRoot(container);
  await React.act(async () => root.render(React.createElement(Page)));
  const click = async (element) => {
    expect(element).toBeTruthy();
    await React.act(async () => element.dispatchEvent(new MouseEvent('click', { bubbles: true })));
  };
  const change = async (element, value) => {
    await React.act(async () => {
      const proto = element.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(proto, 'value').set.call(element, value);
      element.dispatchEvent(new Event('input', { bubbles: true }));
    });
  };
  const enter = async (element) => React.act(async () => {
    element.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
  });

  await click(container.querySelector('.hermes-kanban-column-add'));
  const form = container.querySelector('form');
  await change(form.querySelector('textarea'), 'One run');
  await change(form.querySelector('input[type=number]'), '1');
  await React.act(async () => form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
  expect(writes.at(-1).max_worker_runs).toBe(1);

  await click(container.querySelector('.hermes-kanban-card'));
  expect(container.textContent).toContain('1/1 run IDs used');
  await click(container.querySelector('button[title^="Edit absolute cap"]'));
  let editor = container.querySelector('input[title^="Absolute run-ID cap"]');
  await change(editor, '4'); await enter(editor);
  expect(writes.at(-1)).toEqual({ max_worker_runs: 4 });
  expect(container.textContent).toContain('1/4 run IDs used');
  await click(container.querySelector('button[title^="Edit absolute cap"]'));
  editor = container.querySelector('input[title^="Absolute run-ID cap"]');
  await change(editor, ''); await enter(editor);
  expect(writes.at(-1)).toEqual({ clear_max_worker_runs: true });
  expect(container.textContent).toContain('1/unlimited run IDs used');
});
