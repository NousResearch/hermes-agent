import { describe, expect, it } from 'vitest'

import type { DesktopUpdateStatus } from '@/global'

import { sourceUpdateChannel } from './updates'

const status = (fields: Partial<DesktopUpdateStatus>): DesktopUpdateStatus => ({
  supported: true,
  mechanism: 'posix-handoff',
  channelSelectable: true,
  ...fields
})

describe('sourceUpdateChannel', () => {
  it('offers the selector only where the choice is stable or main and can be saved', () => {
    expect(sourceUpdateChannel(status({ channel: 'stable' }))).toBe('stable')
    expect(sourceUpdateChannel(status({ mechanism: 'windows-handoff', branch: 'main' }))).toBe('main')
    // Packages bake their channel; a preview channel is not the user's to flip here.
    for (const mechanism of ['electron-updater', 'app-installer', 'microsoft-store', 'external'] as const) {
      expect(sourceUpdateChannel(status({ mechanism, channel: 'stable' }))).toBeNull()
    }

    expect(sourceUpdateChannel(status({ channel: 'pm-preview' }))).toBeNull()
    expect(sourceUpdateChannel(status({ supported: false, channel: 'stable' }))).toBeNull()
    expect(sourceUpdateChannel(null)).toBeNull()
    // A runtime older than --set-channel answers checks but cannot save a channel.
    expect(sourceUpdateChannel(status({ channel: 'stable', channelSelectable: undefined }))).toBeNull()
    // A custom branch is neither stable releases nor every commit on main.
    expect(sourceUpdateChannel(status({ branch: 'feature/gui' }))).toBeNull()
  })
})
