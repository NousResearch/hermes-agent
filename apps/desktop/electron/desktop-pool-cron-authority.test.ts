import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

const mainSource = fs.readFileSync(path.join(__dirname, 'main.ts'), 'utf8')

test('pooled Desktop profile backends are marked as non-authoritative cron workers', () => {
  const spawnStart = mainSource.indexOf('async function spawnPoolBackend')
  const spawnEnd = mainSource.indexOf('// Bounded, deduplicated pool teardown', spawnStart)
  const spawnBody = mainSource.slice(spawnStart, spawnEnd)

  assert.ok(spawnStart >= 0 && spawnEnd > spawnStart)
  assert.match(spawnBody, /HERMES_DESKTOP_POOL:\s*'1'/)
})
