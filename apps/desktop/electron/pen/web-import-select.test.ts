import assert from 'node:assert/strict'

import { test } from 'vitest'

import { ancestorSelector, importedNodes, parseTopLevelNodes, resolveImportMode } from './web-import-select'

test('a selector always means a selection import, whatever mode says', () => {
  assert.equal(resolveImportMode({ selector: '#hero', mode: 'page' }), 'selection')
  assert.equal(resolveImportMode({ mode: 'selection' }), 'selection')
  assert.equal(resolveImportMode({}), 'page')
})

test('ancestor selectors climb exactly `steps` levels from the picked element', () => {
  const pick = 'h1.lede'

  assert.equal(ancestorSelector(pick, 0), pick)
  assert.equal(ancestorSelector(pick, 1), '*:has(> h1.lede)')
  assert.equal(ancestorSelector(pick, 3), '*:has(> * > * > h1.lede)')
  assert.equal(ancestorSelector(pick, -1), undefined)
})

// The editor's `execute` response as pen.dev returns it: boilerplate, then
// Print lines with trailing whitespace.
const EXECUTE_RESPONSE = [
  'OK',
  '',
  'Global variables (e.g. root) carry over to subsequent calls!',
  '',
  '## Print output',
  'hermes-node ["frame0","Frame",0]  ',
  'hermes-node ["RtPii","www.pen.dev",1]',
  ''
].join('\n')

test('the probe lines come back as nodes, the starter frame reading as empty', () => {
  assert.deepEqual(parseTopLevelNodes(EXECUTE_RESPONSE), [
    { empty: true, id: 'frame0', name: 'Frame' },
    { empty: false, id: 'RtPii', name: 'www.pen.dev' }
  ])
  assert.deepEqual(parseTopLevelNodes('OK\n'), [])
})

test('an import is the top-level nodes that were not there before', () => {
  const before = parseTopLevelNodes(EXECUTE_RESPONSE)
  const after = [...before, { empty: false, id: 'dhydB', name: 'div' }]

  assert.deepEqual(importedNodes(before, after), [{ empty: false, id: 'dhydB', name: 'div' }])
  assert.deepEqual(importedNodes(before, before), [])
})
