// updater/updater.test.ts — resolution precedence + wire-shape contracts
// for the strategy layer. Pure DI: no Electron, no payload.

import { execFileSync } from 'node:child_process'
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import { buildStampPayload } from '../../scripts/write-build-stamp.mjs'

import { buildManualUpdateCommand } from './checkout'

import { resolveUpdaterMechanism } from './index'

describe('build stamp → update ownership', () => {
  const provenance = { commit: 'a'.repeat(40), branch: 'main', dirty: false, source: 'ci' }

  const runtime = {
    repoDir: 'app',
    toolsDir: 'tools',
    storePython: 'tools/python/python',
    sitePackages: 'deps',
    commands: { hermes: 'bin/hermes' }
  }

  it.each([
    ['win32', 'bundled', 'app-installer', 'app-installer'],
    ['win32', 'store', 'microsoft-store', 'microsoft-store'],
    ['win32', 'light', 'external', 'external'],
    ['darwin', 'bundled', 'electron-updater', 'electron-updater'],
    ['darwin', 'light', 'electron-updater', 'electron-updater'],
    ['darwin', 'store', 'microsoft-store', 'microsoft-store'],
    ['linux', 'bundled', 'external', 'external'],
    ['linux', 'light', 'external', 'external'],
    ['win32', '', 'self', 'windows-handoff'],
    ['darwin', '', 'self', 'posix-handoff'],
    ['linux', '', 'self', 'posix-handoff']
  ] as const)('%s %s dispatches its declared owner without a Store flag', (platform, variant, declared, strategy) => {
    const stamp: ReturnType<typeof buildStampPayload> = buildStampPayload(
      provenance,
      { HERMES_DESKTOP_VARIANT: variant, HERMES_PAYLOAD_TAG: 'v1.2.3' },
      platform,
      { runtime }
    )

    expect(stamp.updateMechanism).toBe(declared)
    expect(stamp.payload).toBe(variant === 'store' ? 'bundled' : variant || 'bootstrap')
    expect(stamp.tag).toBe('v1.2.3')
    expect(stamp.distribution).toBe('desktop-app')
    expect(stamp).not.toHaveProperty('store')
    expect(resolveUpdaterMechanism({ platform, updateMechanism: stamp.updateMechanism })).toBe(strategy)
  })

  it.each(['win32', 'darwin', 'linux'] as const)('unstamped %s development uses source updates', platform => {
    expect(resolveUpdaterMechanism({ platform, updateMechanism: undefined })).toBe(
      platform === 'win32' ? 'windows-handoff' : 'posix-handoff'
    )
  })
})

describe('buildManualUpdateCommand', () => {
  it('bare command on main and detached HEAD', () => {
    expect(buildManualUpdateCommand('main')).toBe('hermes update')
    expect(buildManualUpdateCommand('HEAD')).toBe('hermes update')
    expect(buildManualUpdateCommand(null)).toBe('hermes update')
  })

  it('branch-pinned for non-main checkouts', () => {
    expect(buildManualUpdateCommand('ethie/pm')).toBe('hermes update --branch ethie/pm')
  })

  it.skipIf(process.platform === 'win32')('preserves a quoted Git branch in the manual command', () => {
    const root = mkdtempSync(join(tmpdir(), 'manual-update-'))
    const branch = "team/o'brien"

    try {
      const launcher = join(root, 'hermes')
      writeFileSync(launcher, '#!/bin/sh\nprintf \'%s\\n\' "$@"\n')
      chmodSync(launcher, 0o755)

      const argv = execFileSync('/bin/sh', ['-c', buildManualUpdateCommand(branch)], {
        env: { PATH: `${root}:/usr/bin:/bin` },
        encoding: 'utf8'
      }).trim().split('\n')

      expect(argv).toEqual(['update', '--branch', branch])
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  })
})
