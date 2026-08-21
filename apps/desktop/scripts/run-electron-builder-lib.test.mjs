import assert from 'node:assert/strict'
import { test } from 'vitest'

import { computeElectronBuilderArgs } from '../scripts/run-electron-builder-lib.mjs'

test('computeElectronBuilderArgs keeps local electronDist for host-target builds', () => {
  const args = computeElectronBuilderArgs({
    argv: ['--linux', 'AppImage'],
    dist: '/tmp/electron-dist',
    hasBinary: true,
    hostPlatform: 'linux'
  })

  assert.deepEqual(args, ['-c.electronDist=/tmp/electron-dist', '--linux', 'AppImage'])
})

test('computeElectronBuilderArgs drops local electronDist for linux-to-windows cross-builds', () => {
  const args = computeElectronBuilderArgs({
    argv: ['--win', 'nsis', '--dir'],
    dist: '/tmp/electron-dist',
    hasBinary: true,
    hostPlatform: 'linux'
  })

  assert.deepEqual(args, ['--win', 'nsis', '--dir'])
})

test('computeElectronBuilderArgs keeps explicit args even when no local electron binary exists', () => {
  const args = computeElectronBuilderArgs({
    argv: ['--win', 'nsis'],
    dist: '/tmp/electron-dist',
    hasBinary: false,
    hostPlatform: 'linux'
  })

  assert.deepEqual(args, ['--win', 'nsis'])
})
