/**
 * Tests for electron/open-target.cjs.
 *
 * Run with: node --test electron/open-target.test.cjs
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * classifyOpenTarget is the pure decision behind main.cjs::openExternalUrl —
 * it decides whether an href is a local file (shell.openPath), a web URL
 * (shell.openExternal), or should be rejected. This is the main-process half
 * of the "chat file link is a silent dead end" fix.
 */

const test = require('node:test')
const assert = require('node:assert/strict')

const { classifyOpenTarget } = require('./open-target.cjs')

test('rejects empty / nullish input', () => {
  assert.deepEqual(classifyOpenTarget(''), { kind: 'reject' })
  assert.deepEqual(classifyOpenTarget('   '), { kind: 'reject' })
  assert.deepEqual(classifyOpenTarget(null), { kind: 'reject' })
  assert.deepEqual(classifyOpenTarget(undefined), { kind: 'reject' })
})

test('classifies http/https/mailto as web', () => {
  assert.deepEqual(classifyOpenTarget('https://example.com/a'), {
    kind: 'web',
    url: 'https://example.com/a'
  })
  assert.equal(classifyOpenTarget('http://example.com').kind, 'web')
  assert.equal(classifyOpenTarget('mailto:me@example.com').kind, 'web')
})

test('classifies a file:// URL as file, preserving encoded special chars', () => {
  // The renderer now percent-encodes path segments; we must NOT lose them.
  const res = classifyOpenTarget('file:///tmp/weird%23name%3F.png')
  assert.equal(res.kind, 'file')
  assert.equal(res.path, 'file:///tmp/weird%23name%3F.png')
})

test('classifies a file:// URL with spaces as file', () => {
  const res = classifyOpenTarget('file:///Users/c/Application%20Support/x.png')
  assert.equal(res.kind, 'file')
  assert.equal(res.path, 'file:///Users/c/Application%20Support/x.png')
})

test('classifies a scheme-less RELATIVE path as file (raw, for main to resolve)', () => {
  // The screenshot case: the renderer hands these over raw because
  // file://<relative> mis-parses the filename as a URL host.
  assert.deepEqual(classifyOpenTarget('hermes-support-slack.png'), {
    kind: 'file',
    path: 'hermes-support-slack.png'
  })
  assert.deepEqual(classifyOpenTarget('out/report.png'), {
    kind: 'file',
    path: 'out/report.png'
  })
})

test('classifies a ~ path as file (raw, for main to expand home)', () => {
  assert.deepEqual(classifyOpenTarget('~/Desktop/foo.png'), {
    kind: 'file',
    path: '~/Desktop/foo.png'
  })
})

test('classifies an absolute POSIX path as file', () => {
  assert.deepEqual(classifyOpenTarget('/var/log/x.txt'), {
    kind: 'file',
    path: '/var/log/x.txt'
  })
})

test('rejects an unknown URL scheme (cannot masquerade as a path)', () => {
  // Guards against e.g. custom-protocol abuse: a real scheme://... that we
  // don't allow must reject, not fall through to openPath.
  assert.deepEqual(classifyOpenTarget('bitbrowser://open?x=1'), { kind: 'reject' })
  assert.deepEqual(classifyOpenTarget('javascript://alert(1)'), { kind: 'reject' })
})

test('trims surrounding whitespace before classifying', () => {
  assert.equal(classifyOpenTarget('  https://example.com  ').kind, 'web')
  assert.deepEqual(classifyOpenTarget('  ./a.png  '), { kind: 'file', path: './a.png' })
})
