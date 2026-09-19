import { expect, test } from 'vitest'

import { resolveLinuxUnpackedDirName } from './packaged-app-layout.mjs'

test('uses electron-builder default directory on Linux x64', () => {
  expect(resolveLinuxUnpackedDirName('x64')).toBe('linux-unpacked')
})

test('includes architecture in electron-builder Linux ARM64 directory', () => {
  expect(resolveLinuxUnpackedDirName('arm64')).toBe('linux-arm64-unpacked')
})
