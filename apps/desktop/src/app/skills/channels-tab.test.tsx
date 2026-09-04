// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { queryClient } from '@/lib/query-client'

import { ChannelsTab } from './channels-tab'

const getChannelCapabilities = vi.fn()
const updateChannelCapabilities = vi.fn()

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getChannelCapabilities: (profile?: null | string) => getChannelCapabilities(profile),
  updateChannelCapabilities: (platform: string, update: unknown, profile?: null | string) =>
    updateChannelCapabilities(platform, update, profile)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

function channelCapability(overrides: Record<string, unknown> = {}) {
  return {
    effective_toolsets: ['web'],
    explicit: true,
    implicit_toolsets: [],
    label: 'Email',
    mcp: {
      available: ['alpha'],
      effective: ['alpha'],
      mode: 'allowlist' as const,
      selected: ['alpha']
    },
    platform: 'email',
    plugins_locked: false,
    toolsets: [
      {
        description: 'Search the web',
        enabled: true,
        label: 'Web',
        name: 'web',
        tools: ['web_search']
      },
      {
        description: 'Remember context',
        enabled: false,
        label: 'Memory',
        name: 'memory',
        tools: ['memory_search']
      }
    ],
    ...overrides
  }
}

async function renderChannels(profile: null | string = null) {
  await act(async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <ChannelsTab profile={profile} query="" />
      </QueryClientProvider>
    )
  })
}

beforeEach(() => {
  getChannelCapabilities.mockResolvedValue([channelCapability()])
  updateChannelCapabilities.mockResolvedValue({ ok: true, channel: channelCapability() })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  queryClient.clear()
})

describe('ChannelsTab', () => {
  it('uses the selected capability profile for reads and writes', async () => {
    await renderChannels('review')

    await waitFor(() => expect(getChannelCapabilities).toHaveBeenCalledWith('review'))
    fireEvent.click(await screen.findByRole('button', { name: 'Save abilities' }))

    await waitFor(() =>
      expect(updateChannelCapabilities).toHaveBeenCalledWith(
        'email',
        {
          toolsets: ['web'],
          mcp_mode: 'allowlist',
          mcp_servers: ['alpha']
        },
        'review'
      )
    )
  })

  it('saves the exact edited toolset and MCP boundaries', async () => {
    await renderChannels()

    fireEvent.click(await screen.findByRole('switch', { name: 'Toggle Memory for this channel' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save abilities' }))

    await waitFor(() =>
      expect(updateChannelCapabilities).toHaveBeenCalledWith(
        'email',
        {
          toolsets: ['memory', 'web'],
          mcp_mode: 'allowlist',
          mcp_servers: ['alpha']
        },
        null
      )
    )
  })

  it('locks channel controls while a boundary update is pending', async () => {
    getChannelCapabilities.mockResolvedValue([
      channelCapability(),
      channelCapability({ label: 'Telegram', platform: 'telegram' })
    ])
    let finishSave: (() => void) | undefined
    updateChannelCapabilities.mockImplementationOnce(
      () =>
        new Promise<void>(resolve => {
          finishSave = resolve
        })
    )

    await renderChannels()
    fireEvent.click(await screen.findByRole('button', { name: 'Save abilities' }))
    await waitFor(() => expect(updateChannelCapabilities).toHaveBeenCalledOnce())

    expect(screen.getByRole('button', { name: 'Email' })).toHaveProperty('disabled', true)
    expect(screen.getByRole('button', { name: 'Telegram' })).toHaveProperty('disabled', true)

    finishSave?.()
    await waitFor(() => expect(screen.getByRole('button', { name: 'Telegram' })).toHaveProperty('disabled', false))
  })
})
