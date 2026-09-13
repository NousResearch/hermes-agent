import assert from 'node:assert/strict'
import { test } from 'vitest'

import { windowsFileVersion } from './windows-file-version.mjs'

test('stable tags leave VERSIONINFO to electron-builder', () => {
  assert.equal(windowsFileVersion('v0.20.5'), null)
  assert.equal(windowsFileVersion('v0.28.0'), null)
})

test('canary tags pack the timestamp into a legal quad', () => {
  assert.equal(windowsFileVersion('v0.28.0-canary.20260819171926'), '2026.819.1719.26')
  assert.equal(windowsFileVersion('v0.20.5-canary.20260826'), '2026.826.0.0')
})
