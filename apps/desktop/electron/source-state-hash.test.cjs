const assert = require('node:assert/strict')
const test = require('node:test')

const {
  EMPTY_DESKTOP_SOURCE_STATE_HASH,
  desktopSourceStateMaterial,
  hashText
} = require('./source-state-hash.cjs')

test('desktop source state material is stable and includes untracked file entries', () => {
  const diff = 'diff --git a/apps/desktop/src/a.ts b/apps/desktop/src/a.ts\n'
  const untracked = ['apps/desktop/src/new.ts\t12\tabc123']

  assert.equal(hashText(desktopSourceStateMaterial('', [])), EMPTY_DESKTOP_SOURCE_STATE_HASH)
  assert.equal(
    hashText(desktopSourceStateMaterial(diff, untracked)),
    hashText(desktopSourceStateMaterial(diff, untracked))
  )
  assert.notEqual(
    hashText(desktopSourceStateMaterial(diff, [])),
    hashText(desktopSourceStateMaterial(diff, untracked))
  )
})
