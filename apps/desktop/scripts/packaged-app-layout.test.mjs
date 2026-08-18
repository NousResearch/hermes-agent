import assert from 'node:assert/strict'
import test from 'node:test'

import { resolveLinuxUnpackedDirName } from './packaged-app-layout.mjs'

test('uses electron-builder default directory on Linux x64', () => {
  assert.equal(resolveLinuxUnpackedDirName('x64'), 'linux-unpacked')
})

test('includes architecture in electron-builder Linux ARM64 directory', () => {
  assert.equal(resolveLinuxUnpackedDirName('arm64'), 'linux-arm64-unpacked')
})
