import { describe, expect, it } from 'vitest'

import { appVersionInfo, assertSourceUpdateChannel } from './app-version'
import type { InstallStamp } from './install-stamp'

describe('artifact version identity', (): void => {
  it.each([
    ['v1.2.3', 'stable'],
    ['v1.2.4+canary.20260911T010101Z', 'canary'],
    [null, null]
  ] as const)(
    'reports the client version and fixed channel for %s, not a remote runtime',
    (tag: string | null, channel: 'stable' | 'canary' | null): void => {
      const stamp: InstallStamp = {
        payload: 'bundled',
        tag,
        displayVersion: '1.2.3+gabcdef12',
        source: tag ? 'ci' : 'commit-build',
        updateMechanism: 'external'
      } as InstallStamp

      expect(appVersionInfo(stamp, '9.9.9-remote', '1.2.3')).toMatchObject({
        appVersion: stamp.displayVersion,
        channel,
        source: stamp.source
      })
      expect((): void => assertSourceUpdateChannel(stamp)).toThrow()
    }
  )

  it('reports baked channel source version independently of native sequence version', (): void => {
    const stamp: InstallStamp = {
      schemaVersion: 1,
      source: 'channel-build',
      payload: 'bundled',
      distribution: 'desktop-app',
      updateMechanism: 'electron-updater',
      tag: null,
      commit: 'a'.repeat(40),
      commitDate: null,
      branch: null,
      builtAt: null,
      dirty: false,
      baseVersion: '1.2.3',
      displayVersion: '0.0.7',
      distance: null,
      channelBuild: {
        schema: 1,
        buildId: 'b'.repeat(32),
        channel: 'new-name',
        sequence: 7,
        repository: 'example/hermes-agent',
        commit: 'a'.repeat(40),
        sourceVersion: '1.2.3',
        version: '0.0.7',
        windowsVersion: '0.0.7.0',
        publicBase: 'https://example.com',
        bundleEnv: {},
        identity: {
          token: '1234567890abcdef',
          displayName: 'Preview',
          appId: 'chat.preview',
          appNamePascal: 'Preview',
          artifactNamePascal: 'Preview',
          cliName: 'preview',
          windowsExecutableName: 'Preview.exe',
          msixAppIdWithOrg: 'Nous.Preview'
        }
      }
    }

    expect(appVersionInfo(stamp, '9.9.9-remote', '0.0.7')).toMatchObject({
      channel: 'new-name',
      baseVersion: '1.2.3',
      appVersion: '1.2.3 (new-name #7, aaaaaaaa)',
      sequence: 7,
      buildId: 'b'.repeat(32)
    })
  })

  it('keeps source installs on their runtime version without a fixed package channel', (): void => {
    expect(appVersionInfo(null, '9.9.9-runtime', '1.2.3')).toMatchObject({ appVersion: '9.9.9-runtime' })
    expect((): void => assertSourceUpdateChannel(null)).not.toThrow()
  })
})
