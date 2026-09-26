import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

test('every profileBackendParentEnv call in main.ts uses the pinned profile binding', () => {
  const source = fs.readFileSync(path.join(import.meta.dirname, 'main.ts'), 'utf8')
  const calls = source.match(/profileBackendParentEnv\(\{[\s\S]*?\}/g) ?? []

  assert.ok(calls.length >= 2, `expected both local serve spawn sites, found ${calls.length}`)

  for (const call of calls) {
    assert.doesNotMatch(
      call,
      /profile\s*:\s*activeProfile/,
      'primary spawn must pass the in-scope `profile` binding; `activeProfile` is a different identifier and crashes boot when it is not in scope'
    )
    assert.match(call, /\bprofile\b/)
  }
})
