// Real filesystem fixtures for internal-page and fragment validation.
// Run from apps/docs: node --test tests/check-export-links.test.mjs
//
// Follow-up hardening for the same checker: same-origin absolute and
// protocol-relative links must be validated like local paths (the old checker
// skipped every full-URL href, so its same-origin "happy path" test proved
// nothing), encoded boundary escapes must fail even when the escaped file
// exists on disk, and malformed targets (%00) must produce diagnostics
// instead of an uncaught filesystem crash.

import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { checkExportLinks } from '../scripts/check-export-links.mjs';

const PAGE = (body) => `<!doctype html><html><head><title>p</title></head><body>${body}</body></html>`;

/** A tiny bilingual export: two pages per route, plus real artifacts. */
function buildExport(t, { pageBodies } = {}) {
  const root = mkdtempSync(path.join(tmpdir(), 'hermes-export-links-'));
  const appDir = path.join(root, 'apps', 'docs');
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const outDir = path.join(appDir, 'out');
  const routes = ['/', '/getting-started/quickstart', '/zh-Hans/getting-started/quickstart'];
  const bodies = pageBodies ?? {
    '/': PAGE('<a href="/docs/getting-started/quickstart#install">good</a><a href="#top">frag</a><h1 id="top">Top</h1>'),
    '/getting-started/quickstart': PAGE('<h2 id="install">Install</h2><a name="legacy">legacy anchor</a><a href="/docs/llms.txt">llms</a><a href="mailto:x@example.com">mail</a><a href="myapp://deep">custom</a>'),
    '/zh-Hans/getting-started/quickstart': PAGE('<h2 id="install">安装</h2>'),
  };
  for (const [route, body] of Object.entries(bodies)) {
    const file = path.join(outDir, route, 'index.html');
    mkdirSync(path.dirname(file), { recursive: true });
    writeFileSync(file, body);
  }
  writeFileSync(path.join(outDir, 'llms.txt'), 'LLM docs\n');
  mkdirSync(path.join(outDir, 'img'), { recursive: true });
  writeFileSync(path.join(outDir, 'img', 'logo@2x.png'), 'PNG2');
  mkdirSync(path.join(outDir, 'img', 'empty-dir'));
  return { appDir, outDir, routes };
}

function run(appDir, routes) {
  const errors = [];
  checkExportLinks(appDir, errors.push.bind(errors), { routes });
  return errors;
}

test('checks declared fragment translations even when no page currently links to them', (t) => {
  const { appDir, outDir, routes } = buildExport(t);
  const mapFile = path.join(appDir, 'src/lib/fragment-translations.json');
  mkdirSync(path.dirname(mapFile), { recursive: true });
  writeFileSync(mapFile, JSON.stringify({ '/getting-started/quickstart': { install: '安装', legacy: null } }));
  writeFileSync(path.join(outDir, 'getting-started/quickstart/index.html'), PAGE('<h2 id="install">Install</h2><a name="legacy"></a>'));
  writeFileSync(path.join(outDir, 'zh-Hans/getting-started/quickstart/index.html'), PAGE('<h2 id="安装">安装</h2>'));
  assert.deepEqual(run(appDir, routes), []);
  writeFileSync(mapFile, JSON.stringify({ '/getting-started/quickstart': { install: '不存在', missingEnglish: null } }));
  const errors = run(appDir, routes);
  assert.equal(errors.length, 2);
  assert.ok(errors.some((error) => error.includes('不存在') && error.includes('fragment translation')));
  assert.ok(errors.some((error) => error.includes('missingEnglish') && error.includes('fragment translation')));
});

test('accepts page links, on-page fragments, artifacts, mailto and custom protocols', (t) => {
  const { appDir, routes } = buildExport(t);
  assert.deepEqual(run(appDir, routes), []);
});

test('rejects a link to a page missing from the export', (t) => {
  const { appDir, routes, outDir } = buildExport(t, {
    pageBodies: {
      '/': PAGE('<a href="/docs/user-guide/s3-missing-extensionless/">missing page</a>'),
      '/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
      '/zh-Hans/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
    },
  });
  const errors = run(appDir, routes);
  assert.equal(errors.length, 1);
  assert.match(errors[0], /s3-missing-extensionless/);
  assert.match(errors[0], /missing from the export/);
  assert.match(errors[0], /page \//);
});

test('rejects .. and percent-encoded escapes from the /docs base path', (t) => {
  const { appDir, routes } = buildExport(t, {
    pageBodies: {
      '/': PAGE('<a href="/docs/../outside/">dotdot</a><a href="/docs/%2e%2e/encoded">encoded</a>'),
      '/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
      '/zh-Hans/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
    },
  });
  const errors = run(appDir, routes);
  assert.equal(errors.length, 2);
  assert.match(errors[0], /base path/);
  assert.match(errors[1], /base path/);
});

test('rejects invalid percent-encoding in a link path', (t) => {
  const { appDir, routes } = buildExport(t, {
    pageBodies: {
      '/': PAGE('<a href="/docs/%zz">bad</a>'),
      '/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
      '/zh-Hans/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
    },
  });
  const errors = run(appDir, routes);
  assert.equal(errors.length, 1);
  assert.match(errors[0], /percent-encoding/);
});

test('rejects missing on-page and cross-page anchors (the planted review case)', (t) => {
  const { appDir, routes } = buildExport(t, {
    pageBodies: {
      '/': PAGE('<a href="#planted-missing-hash-1">self</a>'
        + '<a href="/docs/getting-started/quickstart#planted-missing-hash-2">cross</a>'),
      '/getting-started/quickstart': PAGE('<h1 id="install">Install</h1>'),
      '/zh-Hans/getting-started/quickstart': PAGE('<h1 id="install">安装</h1>'),
    },
  });
  const errors = run(appDir, routes);
  assert.equal(errors.length, 2);
  assert.match(errors[0], /planted-missing-hash-1/);
  assert.match(errors[0], /missing from its own page/);
  assert.match(errors[1], /planted-missing-hash-2/);
  assert.match(errors[1], /missing anchor/);
});

test('accepts legacy <a name> anchors as link targets', (t) => {
  const { appDir, routes } = buildExport(t, {
    pageBodies: {
      '/': PAGE('<a href="/docs/getting-started/quickstart#legacy">legacy</a>'),
      '/getting-started/quickstart': PAGE('<a name="legacy">legacy anchor</a>'),
      '/zh-Hans/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
    },
  });
  assert.deepEqual(run(appDir, routes), []);
});

test('accepts encoded artifact segments and same-origin absolute URLs', (t) => {
  const { appDir, routes } = buildExport(t, {
    pageBodies: {
      '/': PAGE('<a href="/docs/img/logo%402x.png">encoded</a>'
        + '<a href="https://hermes-agent.nousresearch.com/docs/llms.txt">absolute</a>'),
      '/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
      '/zh-Hans/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
    },
  });
  assert.deepEqual(run(appDir, routes), []);
});

test('rejects a directory without index.html as a link target', (t) => {
  const { appDir, routes } = buildExport(t, {
    pageBodies: {
      '/': PAGE('<a href="/docs/img/empty-dir">dir</a>'),
      '/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
      '/zh-Hans/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
    },
  });
  const errors = run(appDir, routes);
  assert.equal(errors.length, 1);
  assert.match(errors[0], /img\/empty-dir/);
  assert.match(errors[0], /missing from the export/);
});

test('deduplicates one broken link repeated across many pages', (t) => {
  const { appDir, routes } = buildExport(t, {
    pageBodies: {
      '/': PAGE('<a href="/docs/gone/">x</a>'),
      '/getting-started/quickstart': PAGE('<a href="/docs/gone/">x</a>'),
      '/zh-Hans/getting-started/quickstart': PAGE('<a href="/docs/gone/">x</a>'),
    },
  });
  const errors = run(appDir, routes);
  assert.equal(errors.length, 1);
  assert.match(errors[0], /pages \/, \/getting-started\/quickstart, \/zh-Hans\/getting-started\/quickstart/);
});

test('skips routes whose HTML is missing (checkExportedPages owns that)', (t) => {
  const { appDir, routes, outDir } = buildExport(t);
  rmSync(path.join(outDir, 'zh-Hans', 'getting-started', 'quickstart', 'index.html'));
  assert.deepEqual(run(appDir, routes), []);
});

test('accepts percent-encoded on-page and cross-page anchors (decoded once)', (t) => {
  const { appDir, routes } = buildExport(t, {
    pageBodies: {
      '/': PAGE('<h2 id="注册新账号命令">注册</h2>'
        + '<a href="#%E6%B3%A8%E5%86%8C%E6%96%B0%E8%B4%A6%E5%8F%B7%E5%91%BD%E4%BB%A4">self</a>'
        + '<a href="/docs/getting-started/quickstart#%E5%AE%89%E8%A3%85">cross</a>'),
      '/getting-started/quickstart': PAGE('<h2 id="安装">安装</h2>'),
      '/zh-Hans/getting-started/quickstart': PAGE('<h2 id="安装">安装</h2>'),
    },
  });
  assert.deepEqual(run(appDir, routes), []);
});

test('rejects invalid percent-encoding in a link fragment', (t) => {
  const { appDir, routes } = buildExport(t, {
    pageBodies: {
      '/': PAGE('<a href="#%zz">bad</a>'),
      '/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
      '/zh-Hans/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
    },
  });
  const errors = run(appDir, routes);
  assert.equal(errors.length, 1);
  assert.match(errors[0], /percent-encoding/);
});

test('returns without diagnostics when ctx.routes is absent', (t) => {
  const { appDir } = buildExport(t);
  const errors = [];
  checkExportLinks(appDir, errors.push.bind(errors), {});
  assert.deepEqual(errors, []);
});

test('rejects same-origin absolute links to missing pages and anchors', (t) => {
  const { appDir, routes } = buildExport(t, {
    pageBodies: {
      '/': PAGE('<a href="https://hermes-agent.nousresearch.com/docs/user-guide/gone-page/">gone</a>'
        + '<a href="https://hermes-agent.nousresearch.com/docs/getting-started/quickstart#nope">anchor</a>'),
      '/getting-started/quickstart': PAGE('<h1 id="install">Install</h1>'),
      '/zh-Hans/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
    },
  });
  const errors = run(appDir, routes);
  assert.equal(errors.length, 2);
  assert.match(errors[0], /gone-page/);
  assert.match(errors[0], /missing from the export/);
  assert.match(errors[1], /nope/);
  assert.match(errors[1], /missing anchor/);
});

test('checks protocol-relative same-origin links and still skips other origins', (t) => {
  const { appDir, routes } = buildExport(t, {
    pageBodies: {
      '/': PAGE('<a href="//hermes-agent.nousresearch.com/docs/getting-started/quickstart#install">good</a>'
        + '<a href="//hermes-agent.nousresearch.com/docs/user-guide/gone-page/">gone</a>'
        + '<a href="//example.com/docs/user-guide/gone-page/">cross</a>'),
      '/getting-started/quickstart': PAGE('<h2 id="install">Install</h2>'),
      '/zh-Hans/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
    },
  });
  const errors = run(appDir, routes);
  assert.equal(errors.length, 1);
  assert.match(errors[0], /gone-page/);
  assert.match(errors[0], /missing from the export/);
});

test('rejects encoded-slash escapes even when the target file exists outside the export', (t) => {
  const { appDir, routes } = buildExport(t, {
    pageBodies: {
      '/': PAGE('<a href="/docs/img%2f..%2f..%2fleaked-outside.html">leak</a>'),
      '/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
      '/zh-Hans/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
    },
  });
  writeFileSync(path.join(appDir, 'leaked-outside.html'), PAGE('<h1 id="leak">Outside the export</h1>'));
  const errors = run(appDir, routes);
  assert.equal(errors.length, 1);
  assert.match(errors[0], /leaked-outside/);
  assert.match(errors[0], /outside the export directory/);
});

test('reports a null byte in a link path instead of crashing', (t) => {
  const { appDir, routes } = buildExport(t, {
    pageBodies: {
      '/': PAGE('<a href="/docs/%00leaked">nul</a>'),
      '/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
      '/zh-Hans/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
    },
  });
  const errors = run(appDir, routes);
  assert.equal(errors.length, 1);
  assert.match(errors[0], /null byte/);
});

test('allows same-origin full-URL links to product paths outside the docs base', (t) => {
  const { appDir, routes } = buildExport(t, {
    pageBodies: {
      '/': PAGE('<a href="https://hermes-agent.nousresearch.com/">home</a>'
        + '<a href="https://hermes-agent.nousresearch.com/desktop">desktop</a>'
        + '<a href="//hermes-agent.nousresearch.com/desktop">relative-scheme</a>'),
      '/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
      '/zh-Hans/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
    },
  });
  assert.deepEqual(run(appDir, routes), []);
});

test('resolves same-origin links to the docs root and to a legacy alias page', (t) => {
  const { appDir, routes, outDir } = buildExport(t, {
    pageBodies: {
      '/': PAGE('<a href="https://hermes-agent.nousresearch.com/docs">root</a>'
        + '<a href="https://hermes-agent.nousresearch.com/docs/quickstart">alias</a>'),
      '/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
      '/zh-Hans/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
    },
  });
  mkdirSync(path.join(outDir, 'quickstart'), { recursive: true });
  writeFileSync(path.join(outDir, 'quickstart', 'index.html'), PAGE('<h1 id="redirect">Redirect</h1>'));
  assert.deepEqual(run(appDir, routes), []);
});

test('rejects a same-origin link to a missing legacy alias page', (t) => {
  const { appDir, routes } = buildExport(t, {
    pageBodies: {
      '/': PAGE('<a href="https://hermes-agent.nousresearch.com/docs/quickstart/">alias</a>'),
      '/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
      '/zh-Hans/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
    },
  });
  const errors = run(appDir, routes);
  assert.equal(errors.length, 1);
  assert.match(errors[0], /quickstart/);
  assert.match(errors[0], /missing from the export/);
});

test('lists one repeated broken link on a single page only once in its examples', (t) => {
  const { appDir, routes } = buildExport(t, {
    pageBodies: {
      '/': PAGE('<a href="/docs/gone/">a</a><a href="/docs/gone/">b</a><a href="/docs/gone/">c</a>'),
      '/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
      '/zh-Hans/getting-started/quickstart': PAGE('<h1 id="x">x</h1>'),
    },
  });
  const errors = run(appDir, routes);
  assert.equal(errors.length, 1);
  assert.match(errors[0], /\(page \/\)$/);
});
