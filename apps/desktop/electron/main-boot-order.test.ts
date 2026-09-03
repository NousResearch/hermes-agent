import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// Regression guard for the boot crash introduced by c401756a6: the pool-limits
// read runs at module evaluation and logs through rememberLog(), which pushes
// into hermesLog. esbuild lowers both top-level consts to `var`, so ordering
// them the wrong way around does not throw a ReferenceError at build time --
// it ships a packaged app that dies on every launch with
// "TypeError: Cannot read properties of undefined (reading 'push')".
// Nothing else in the file catches this, so assert the order on the source.
test('pool limits are read after the log buffer they write into is initialized', () => {
  const source = fs.readFileSync(path.join(__dirname, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')

  const logBufferInit = source.indexOf('\nconst hermesLog = []')
  const poolLimitsInit = source.indexOf('\nlet poolLimits = readPersistedPoolLimits()')

  assert.notEqual(logBufferInit, -1, 'hermesLog must be declared at module scope in main.ts')
  assert.notEqual(poolLimitsInit, -1, 'poolLimits must be initialized at module scope in main.ts')
  assert.ok(
    logBufferInit < poolLimitsInit,
    'readPersistedPoolLimits() logs through rememberLog(): it must run after hermesLog is initialized'
  )
})
