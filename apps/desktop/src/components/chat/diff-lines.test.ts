import assert from 'node:assert/strict'

import { test } from 'vitest'

import { parseDiff, stripDiffFileHeaders } from './diff-lines'

const SINGLE_FILE = [
  'diff --git a/src/a.ts b/src/a.ts',
  'index 1111111..2222222 100644',
  '--- a/src/a.ts',
  '+++ b/src/a.ts',
  '@@ -1,2 +1,3 @@',
  ' keep',
  '-gone',
  '+added',
  '+also added',
  ''
].join('\n')

// What reviewDiff now returns for a collapsed `dir/` row: the per-file all-add
// diffs concatenated, exactly as git prints a multi-file diff.
const MULTI_FILE = [
  'diff --git a/newdir/one.txt b/newdir/one.txt',
  'new file mode 100644',
  'index 0000000..1111111',
  '--- /dev/null',
  '+++ b/newdir/one.txt',
  '@@ -0,0 +1,2 @@',
  '+first',
  '+first again',
  'diff --git a/newdir/sub/two.txt b/newdir/sub/two.txt',
  'new file mode 100644',
  'index 0000000..2222222',
  '--- /dev/null',
  '+++ b/newdir/sub/two.txt',
  '@@ -0,0 +1 @@',
  '+second',
  ''
].join('\n')

test('parseDiff renders a single-file diff without a filename heading', () => {
  // Git's trailing newline leaves one empty context row; ignore that here.
  const lines = parseDiff(SINGLE_FILE).filter(line => line.text !== '')

  // The panel header already names the file — no redundant heading row.
  assert.deepEqual(
    lines.map(line => [line.kind, line.text]),
    [
      ['context', 'keep'],
      ['remove', 'gone'],
      ['add', 'added'],
      ['add', 'also added']
    ]
  )
})

test('parseDiff labels each file in a multi-file diff', () => {
  const texts = parseDiff(MULTI_FILE).map(line => line.text)

  assert.ok(texts.includes('newdir/one.txt'), 'first file is named')
  assert.ok(texts.includes('newdir/sub/two.txt'), 'second file is named')
})

test('parseDiff keeps every file body in a multi-file diff', () => {
  const adds = parseDiff(MULTI_FILE)
    .filter(line => line.kind === 'add')
    .map(line => line.text)

  assert.deepEqual(adds, ['first', 'first again', 'second'])
})

test('parseDiff drops inter-file header noise instead of showing it as context', () => {
  const texts = parseDiff(MULTI_FILE).map(line => line.text)

  // These used to leak into the previous file's hunk as context lines, because
  // the header strip only ran on the leading block.
  for (const noise of [
    'diff --git a/newdir/sub/two.txt b/newdir/sub/two.txt',
    'new file mode 100644',
    '--- /dev/null'
  ]) {
    assert.ok(!texts.includes(noise), `header noise not rendered: ${noise}`)
  }
})

test('parseDiff falls back to raw lines for a payload with no hunks', () => {
  assert.deepEqual(
    parseDiff('some plain text').map(line => line.text),
    ['some plain text']
  )
})

test('stripDiffFileHeaders still strips the leading header block', () => {
  assert.equal(stripDiffFileHeaders(SINGLE_FILE).split('\n')[0], '@@ -1,2 +1,3 @@')
})
