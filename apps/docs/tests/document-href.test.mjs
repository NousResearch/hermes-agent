import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { resolveDocumentLink } from '../src/lib/content.mjs';
import { documentHref } from '../src/lib/document-href.ts';
import fragmentTranslations from '../src/lib/fragment-translations.json' with { type: 'json' };

const translations = {
  '/guide': { install: '安装', advanced: null },
  '/other': { configure: '配置' },
};

test('Chinese fragment links use verified section mappings and preserve query strings', () => {
  assert.equal(documentHref('/guide?mode=1#install', 'zh-Hans', '/docs/start/', translations), '/zh-Hans/guide?mode=1#%E5%AE%89%E8%A3%85');
  assert.equal(documentHref('#install', 'zh-Hans', '/docs/zh-Hans/guide/', translations), '#%E5%AE%89%E8%A3%85');
  assert.equal(documentHref('/guide#install', 'en', '/docs/start/', translations), '/guide#install');
  assert.equal(documentHref('/guide#%E5%AE%89%E8%A3%85', 'zh-Hans', '/docs/start/', translations), '/zh-Hans/guide#%E5%AE%89%E8%A3%85');
});

test('a missing translated section uses the exact English destination within the current host', () => {
  assert.equal(documentHref('/guide?mode=1#advanced', 'zh-Hans', '/docs/start/', translations), '/guide?mode=1#advanced');
  assert.equal(documentHref('#advanced', 'zh-Hans', '/docs/zh-Hans/guide/', translations), '/guide/#advanced');
  // English source shown at a zh-Hans fallback URL follows the same rule.
  assert.equal(documentHref('/guide#advanced', 'zh-Hans', '/docs/zh-Hans/en-only/', translations), '/guide#advanced');
  assert.equal(documentHref('/docs/guide/#advanced', 'zh-Hans', '/docs/zh-Hans/start/', translations), '/guide/#advanced');
});

test('extensionless configuration links reach the translated prompt-cache section', () => {
  const contentDir = fileURLToPath(new URL('../content/', import.meta.url));
  const source = fileURLToPath(new URL('../content/zh-Hans/user-guide/features/overview.md', import.meta.url));
  const resolved = resolveDocumentLink(source, '../configuration#prompt-caching', contentDir);
  assert.equal(documentHref(resolved, 'zh-Hans', '/docs/zh-Hans/user-guide/features/overview/', fragmentTranslations),
    '/zh-Hans/user-guide/configuration#prompt-%E7%BC%93%E5%AD%98');
});

test('English fallback content retains a connector section omitted from Chinese', () => {
  const contentDir = fileURLToPath(new URL('../content/', import.meta.url));
  const source = fileURLToPath(new URL('../content/en/user-guide/features/tool-search.md', import.meta.url));
  const resolved = resolveDocumentLink(source, './tool-gateway.mdx#multiple-connector-accounts', contentDir);
  assert.equal(documentHref(resolved, 'zh-Hans', '/docs/zh-Hans/user-guide/features/tool-search/', fragmentTranslations),
    '/user-guide/features/tool-gateway#multiple-connector-accounts');
});

test('ordinary pages, unknown fragments, explicit IDs and external links keep their handling', () => {
  assert.equal(documentHref('/guide', 'zh-Hans', '/docs/start/', translations), '/zh-Hans/guide');
  assert.equal(documentHref('#authored-id', 'zh-Hans', '/docs/zh-Hans/guide/', translations), '#authored-id');
  assert.equal(documentHref('/guide#unknown', 'zh-Hans', '/docs/start/', translations), '/zh-Hans/guide#unknown');
  assert.equal(documentHref('https://elsewhere.test/guide#install', 'zh-Hans', '/docs/start/', translations), 'https://elsewhere.test/guide#install');
  assert.equal(documentHref('mailto:docs@example.com', 'zh-Hans', '/docs/start/', translations), 'mailto:docs@example.com');
  assert.equal(documentHref('/guide#%zz', 'zh-Hans', '/docs/start/', translations), '/zh-Hans/guide#%zz');
});
