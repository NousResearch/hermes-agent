/* Exercise the shipped no-build plugin through its public registration and real React DOM. */
const { test, afterEach } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { JSDOM } = require('jsdom');
const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
global.window = dom.window;
global.document = dom.window.document;
global.HTMLElement = dom.window.HTMLElement;
global.IS_REACT_ACT_ENVIRONMENT = true;
const React = require('react');
const { render, screen, fireEvent, cleanup } = require('@testing-library/react');
afterEach(cleanup);

async function openTask(origin) {
  const task = { id: 'fixture-task', title: 'Open fixture card', status: 'ready', assignee: 'default',
    workspace_kind: 'scratch', priority: 0, origin };
  const components = {};
  for (const name of ['Card', 'CardContent', 'Badge', 'Button', 'Input', 'Label', 'Select', 'SelectOption']) {
    const tag = { Button: 'button', Input: 'input', Label: 'label', Select: 'select', SelectOption: 'option' }[name] || 'div';
    components[name] = ({ children, onValueChange, ...props }) => React.createElement(tag,
      onValueChange ? { ...props, onChange: (event) => onValueChange(event.target.value) } : props, children);
  }
  let Page;
  window.__HERMES_PLUGINS__ = { register: (id, component) => { assert.equal(id, 'kanban'); Page = component; } };
  window.__HERMES_PLUGIN_SDK__ = {
    React, components, hooks: React, buildWsUrl: async (path) => 'ws://localhost' + path,
    utils: { cn: (...parts) => parts.filter(Boolean).join(' '), timeAgo: () => 'now' },
    fetchJSON: async (url) => {
      if (url.includes('/tasks/fixture-task/log')) return { exists: false };
      if (url.includes('/tasks/fixture-task')) return { task, comments: [], events: [], runs: [] };
      if (url.includes('/home-channels')) return { home_channels: [] };
      if (url.includes('/boards')) return { boards: [{ slug: 'default', name: 'Default' }] };
      if (url.includes('/board')) return { columns: [{ name: 'ready', tasks: [task] }], assignees: ['default'], tenants: [], latest_event_id: 0 };
      return {};
    },
  };
  // Evaluate the artifact as the host does; no source-pattern assertions or private exports.
  vm.runInNewContext(fs.readFileSync(`${__dirname}/dist/index.js`, 'utf8'), {
    window, document, console, URLSearchParams, localStorage: window.localStorage,
    setTimeout, clearTimeout, setInterval, clearInterval,
    WebSocket: class { close() {} },
  });
  assert.ok(Page);
  render(React.createElement(Page));
  fireEvent.click(await screen.findByText('Open fixture card'));
  await screen.findByText('Workspace');
}

test('drawer paints conversation anchors and quoted prompt as text, constructing its own local link', async () => {
  await openTask({ profile: 'fixture profile', session_id: 'session&fixture', message_id: 'native-fixture',
    message_row_id: 42, platform: 'telegram', chat_id: 'fixture-group', thread_id: 'fixture-topic',
    prompt_excerpt: '<img src=x onerror=alert(1)> quoted fixture', session_link: 'javascript:alert(1)' });
  const link = await screen.findByRole('link', { name: 'fixture profile / session&fixture' });
  assert.equal(link.getAttribute('href'), '/chat?resume=session%26fixture&profile=fixture%20profile');
  assert.ok(screen.getByText('native-fixture (session row 42)'));
  assert.ok(screen.getByText('telegram / fixture-group / fixture-topic'));
  assert.ok(screen.getByText('<img src=x onerror=alert(1)> quoted fixture'));
  assert.equal(document.querySelector('img'), null);
});

test('legacy drawer without provenance remains usable', async () => {
  await openTask(null);
  assert.equal(screen.queryByText('Origin session'), null);
  assert.ok(screen.getByText('Workspace'));
});
