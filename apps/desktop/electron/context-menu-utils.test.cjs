const assert = require('node:assert/strict')
const test = require('node:test')

const {
  FILE_TREE_CONTEXT_SELECTOR,
  buildFileTreeContextLookupScript,
  normalizeFileTreeContext
} = require('./context-menu-utils.cjs')

test('buildFileTreeContextLookupScript reads file-tree row metadata at a point', () => {
  const script = buildFileTreeContextLookupScript(12, 34)

  assert.match(script, /document\.elementFromPoint\(12, 34\)/)
  assert.match(script, /closest/)
  assert.ok(script.includes(FILE_TREE_CONTEXT_SELECTOR))
  assert.match(script, /data-hermes-file-tree-path/)
  assert.match(script, /data-hermes-file-tree-is-directory/)
})

test('buildFileTreeContextLookupScript sanitizes invalid coordinates', () => {
  const script = buildFileTreeContextLookupScript(Number.NaN, 'bad')

  assert.match(script, /document\.elementFromPoint\(0, 0\)/)
})

test('normalizeFileTreeContext returns a clean path payload', () => {
  assert.deepEqual(
    normalizeFileTreeContext({ isDirectory: false, name: 'README.md', path: ' /repo/README.md ' }),
    { isDirectory: false, name: 'README.md', path: '/repo/README.md' }
  )
})

test('normalizeFileTreeContext rejects missing or blank paths', () => {
  assert.equal(normalizeFileTreeContext(null), null)
  assert.equal(normalizeFileTreeContext({ path: '' }), null)
  assert.equal(normalizeFileTreeContext({ path: '   ' }), null)
})
