import { describe, expect, it } from 'vitest'

import { appVersionInfo, assertSourceUpdateChannel } from './app-version'
import type { InstallStamp } from './install-stamp'

describe('artifact version identity', (): void => {
  it.each([
    ['v1.2.3', 'stable'],
    ['v1.2.4-canary.20260911010101', 'canary'],
    [null, null]
  ] as const)('reports the client version and fixed channel for %s, not a remote runtime', (tag: string | null, channel: 'stable' | 'canary' | null): void => {
    const stamp: InstallStamp = {
      payload: 'bundled', tag, displayVersion: '1.2.3+gabcdef12',
      source: tag ? 'ci' : 'commit-build', updateMechanism: 'external'
    } as InstallStamp

    expect(appVersionInfo(stamp, '9.9.9-remote', '1.2.3')).toMatchObject({
      appVersion: stamp.displayVersion, channel, source: stamp.source
    })
    expect((): void => assertSourceUpdateChannel(stamp)).toThrow()
  })

  it('keeps source installs on their runtime version without a fixed package channel', (): void => {
    expect(appVersionInfo(null, '9.9.9-runtime', '1.2.3')).toMatchObject({ appVersion: '9.9.9-runtime' })
    expect((): void => assertSourceUpdateChannel(null)).not.toThrow()
  })
})
