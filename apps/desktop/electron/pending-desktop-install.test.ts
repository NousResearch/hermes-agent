import { describe, expect, it } from 'vitest'

import { pendingDesktopInstallPath, readPendingDesktopInstall } from './pending-desktop-install'

describe('readPendingDesktopInstall', () => {
  const home = '/h'

  it('reads a record the updater wrote', () => {
    const record = {
      app: '/Applications/Hermes.app',
      rebuilt_app: '/src/apps/desktop/release/mac-arm64/Hermes.app',
      asar_hash: 'deadbeef',
      recorded_at: '2026-09-27T00:00:00+0000'
    }

    expect(
      readPendingDesktopInstall(home, () => JSON.stringify(record), () => true)
    ).toEqual({ app: record.app, rebuilt_app: record.rebuilt_app, asar_hash: 'deadbeef' })
  })

  it('is null when no record exists', () => {
    expect(readPendingDesktopInstall(home, () => '', () => false)).toBeNull()
  })

  it.each([
    ['corrupt JSON', '{not json'],
    ['missing app path', JSON.stringify({ rebuilt_app: '/x' })],
    ['missing staged path', JSON.stringify({ app: '/x' })],
    ['an array', '[1,2]']
  ])('treats %s as absent', (_label, text) => {
    expect(readPendingDesktopInstall(home, () => text, () => true)).toBeNull()
  })
})

describe('pendingDesktopInstallPath', () => {
  it('lives at the hermes home root next to the CLI record', () => {
    expect(pendingDesktopInstallPath('/home/u/.hermes')).toBe(
      '/home/u/.hermes/pending_desktop_install.json'
    )
  })
})
