import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { test } from 'vitest'

import { emitDesktopKeys, flattenKeys, renderKeysFile, SURFACE } from './i18n-keys.mjs'

const repoRoot = resolve(import.meta.dirname, '../../..')

test('flattenKeys lists every leaf as a sorted dotted path, functions included, arrays as one leaf', () => {
  assert.deepEqual(
    flattenKeys({ z: { count: n => `${n}`, a: 'x', 'dotted.key': 'y' }, list: ['a'], b: 'c' }),
    ['b', 'list', 'z.a', 'z.count', 'z.dotted.key']
  )
  assert.deepEqual(flattenKeys({ object: () => ({ before: 'x' }), throwing: () => { throw new Error('nope') }, text: () => 'text' }), ['text'])
  assert.equal(renderKeysFile(['a']), `{\n  "surface": "${SURFACE}",\n  "keys": [\n    "a"\n  ]\n}\n`)
})

test('the committed locales/_keys.desktop.json matches the English catalog (run `npm run i18n:keys`)', async () => {
  const out = join(repoRoot, 'locales', `_keys.${SURFACE}.json`)
  const committed = JSON.parse(readFileSync(out, 'utf8'))
  assert.equal(committed.surface, SURFACE)
  assert.ok(committed.keys.includes('common.save'))
  assert.ok(committed.keys.includes('catalog.results'), 'function-valued entries are keys too')
  await emitDesktopKeys({ source: repoRoot, out, check: true })
})
