import assert from 'node:assert/strict';
import test from 'node:test';
import { buildNavigationTree } from '../src/lib/navigation-tree.ts';

const pages = [
  { path: 'getting-started/quickstart.md', url: '/getting-started/quickstart', data: { title: 'Quickstart', sidebar_label: 'First conversation' } },
  { path: 'developer-guide/plugins/index.md', url: '/developer-guide/plugins', data: { title: 'Plugins' } },
];

test('curated virtual groups retain order, labels, index and stable keys', () => {
  const tree = buildNavigationTree([
    { type: 'link', label: 'Browse Skills', href: 'https://hermes-agent.nousresearch.com/marketplace?kind=skills' },
    { type: 'category', label: 'Start here', key: 'start', collapsed: false, items: ['getting-started/quickstart'] },
    { type: 'category', label: 'Extensions', items: [], link: { type: 'doc', id: 'developer-guide/plugins/index' } },
  ], pages, 'zh-Hans');
  // Off-site links are not localized and open as external links.
  assert.equal(tree.children[0].url, 'https://hermes-agent.nousresearch.com/marketplace?kind=skills');
  assert.equal(tree.children[0].external, true);
  assert.equal(tree.children[1].name, 'Start here');
  assert.equal(tree.children[1].$id, 'start');
  assert.equal(tree.children[1].defaultOpen, true);
  assert.equal(tree.children[1].children[0].name, 'First conversation');
  assert.equal(tree.children[2].index.url, '/developer-guide/plugins');
});

test('a missing curated reference fails rather than silently disappearing', () => {
  assert.throws(() => buildNavigationTree(['missing-page'], pages, 'en'), /missing-page/);
});
