import { describe, expect, it } from 'vitest'

import type { DesktopUpdateStatus } from '@/global'

import { sourceUpdateChannel } from './updates'

const status = (fields: Partial<DesktopUpdateStatus>): DesktopUpdateStatus => ({ supported: true, ...fields })

describe('sourceUpdateChannel', () => {
  it('offers the selector only to source checkouts on stable or main', () => {
    expect(sourceUpdateChannel(status({ mechanism: 'posix-handoff', channel: 'stable' }))).toBe('stable')
    expect(sourceUpdateChannel(status({ mechanism: 'windows-handoff', branch: 'main' }))).toBe('main')

    // Packages bake their channel; a preview channel is not the user's to flip here.
    for (const mechanism of ['electron-updater', 'app-installer', 'microsoft-store', 'external'] as const) {
      expect(sourceUpdateChannel(status({ mechanism, channel: 'stable' }))).toBeNull()
    }

    expect(sourceUpdateChannel(status({ mechanism: 'posix-handoff', channel: 'pm-preview' }))).toBeNull()
    expect(sourceUpdateChannel(status({ mechanism: 'posix-handoff', supported: false }))).toBeNull()
    expect(sourceUpdateChannel(null)).toBeNull()
  })
})
