import type * as HermesSdk from '@hermes/plugin-sdk'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as DataModule from './data'
import { translateBots } from './i18n-test-helper'
import type { RosterRow } from './types'

const mocks = vi.hoisted(() => ({
  notify: vi.fn(),
  saveBotMeta: vi.fn(async () => undefined)
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const original = await importOriginal<typeof HermesSdk>()

  return {
    ...original,
    host: {
      ...original.host,
      notify: mocks.notify
    },
    usePluginI18n: () => translateBots
  }
})

vi.mock('./data', async importOriginal => {
  const original = await importOriginal<typeof DataModule>()

  return {
    ...original,
    saveBotMeta: mocks.saveBotMeta
  }
})

const bot: RosterRow = {
  name: 'research'
}

beforeEach(async () => {
  vi.clearAllMocks()
  const [{ $botMeta }, { $groupChats }] = await Promise.all([import('./data'), import('./group-chat')])

  $botMeta.set({
    research: {
      groups: ['Hosted', 'Classic']
    }
  })
  $groupChats.set({
    Hosted: {
      continuityMode: 'gateway',
      hosted: 'install:studio',
      hostedConnectionId: 'host-a',
      hostedEpoch: 1,
      log: [],
      members: [bot],
      roomId: 'hosted-room',
      watermarks: {}
    },
    Classic: {
      continuityMode: 'desktop',
      log: [],
      members: [bot],
      roomId: 'classic-room',
      watermarks: {}
    }
  })
})

afterEach(() => {
  cleanup()
})

describe('mixed Group Chat membership controls', () => {
  it('keeps unrelated classic controls available while preserving the hosted membership', async () => {
    const { GroupDialog } = await import('./create-dialog')

    render(<GroupDialog bot={bot} onClose={() => undefined} />)

    const hosted = screen.getByText('Hosted').closest('label')!.querySelector('[role="checkbox"]') as HTMLButtonElement

    const classic = screen
      .getByText('Classic')
      .closest('label')!
      .querySelector('[role="checkbox"]') as HTMLButtonElement

    expect(hosted.disabled).toBe(true)
    expect(classic.disabled).toBe(false)
    expect((screen.getByRole('button', { name: 'Leave other groups' }) as HTMLButtonElement).disabled).toBe(false)

    fireEvent.click(screen.getByRole('button', { name: 'Leave other groups' }))

    expect(mocks.saveBotMeta).toHaveBeenCalledWith(bot, {
      groups: ['Hosted'],
      group: 'Hosted'
    })
  })
})
