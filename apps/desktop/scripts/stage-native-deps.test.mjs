import assert from 'node:assert/strict'
import { chmodSync, mkdtempSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'

import { copySpawnHelper } from './stage-native-deps.mjs'

test('copySpawnHelper restores executable bits on the staged helper', () => {
  const dir = mkdtempSync(join(tmpdir(), 'hermes-stage-native-'))
  const src = join(dir, 'source-helper')
  const dest = join(dir, 'staged-helper')

  writeFileSync(src, '#!/bin/sh\nexit 0\n')
  chmodSync(src, 0o644)
  copySpawnHelper(src, dest)

  assert.equal(statSync(dest).mode & 0o777, 0o755)
})
