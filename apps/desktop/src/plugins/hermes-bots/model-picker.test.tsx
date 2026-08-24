/**
 * #95279: the Bot Mode model picker's catalog read must always SETTLE, and it
 * must not churn the network on every surface remount.
 *
 * The read rides the BOT's own socket — a second, lazily dialed pool backend
 * that can wedge (cold pool spawn, dropped remote hop) without the primary
 * socket noticing. Two defects made the picker unusable while Bots was active:
 *
 *   1. The RPC had no deadline, so a wedged dial left it pending forever and
 *      the spinner never settled ("model picker never settles").
 *   2. Every fetch forced `refresh: true`, bypassing the react-query cache, so
 *      each Bots view remount (tab re-front, dialog reopen, pane visibility
 *      flip) knocked the picker back into loading and discarded the staged
 *      selection mid-edit.
 *
 * Pinned here through the real component: bounded settlement into the
 * free-text fallback, one cached unforced read across remounts, and no
 * dispatch at all for a row whose connection is gone.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ModelPicker } from './model-picker'
import type { RosterRow } from './types'

const { hostMock } = vi.hoisted(() => ({
  hostMock: {
    request: vi.fn(),
    requestProfile: vi.fn(),
    state: { connectionId: { get: () => 'local' }, gateway: { get: () => 'open' }, profile: { get: () => 'default' } }
  }
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { useQuery } = await import('@tanstack/react-query')

  return {
    Button: (props: React.ComponentProps<'button'>) => <button {...props} />,
    GlyphSpinner: () => <span data-testid="spinner" />,
    host: hostMock,
    Input: (props: React.ComponentProps<'input'>) => <input {...props} />,
    Select: ({ children }: { children: ReactNode }) => <div>{children}</div>,
    SelectContent: ({ children }: { children: ReactNode }) => <div>{children}</div>,
    SelectItem: ({ children }: { children: ReactNode }) => <div>{children}</div>,
    SelectTrigger: ({ children }: { children: ReactNode }) => <div>{children}</div>,
    SelectValue: () => null,
    useQuery
  }
})

vi.mock('./shared', () => ({ ID: 'hermes-bots' }))

const remoteBot = {
  connectionId: 'remote-a',
  name: 'default',
  remoteSource: true,
  route: { connectionId: 'remote-a', mode: 'remote', profile: 'default', targetProfile: 'backend-default' },
  sourceScoped: true,
  targetProfile: 'backend-default'
} as RosterRow

/** One client per render group, so "did the remount refetch?" is a real
 *  question about the cache rather than about a fresh store. */
let client: QueryClient

function mount(bot: null | RosterRow) {
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )

  return render(<ModelPicker bot={bot} onChange={vi.fn()} value={{ model: '', provider: '' }} />, { wrapper })
}

/** The fallback the picker paints when the catalog is unavailable. */
const isFreeText = (container: HTMLElement) =>
  Boolean(container.querySelector('input[placeholder*="omnirouter"]')) &&
  !container.querySelector('[data-testid="spinner"]')

beforeEach(() => {
  vi.clearAllMocks()
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
})

afterEach(() => {
  vi.useRealTimers()
})

describe('a bot gateway that never answers', () => {
  it('settles into the free-text fallback instead of spinning forever', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    hostMock.requestProfile.mockReturnValue(new Promise(() => undefined))

    const { container } = mount(remoteBot)

    expect(container.querySelector('[data-testid="spinner"]')).toBeTruthy()

    await vi.advanceTimersByTimeAsync(30_000)
    await waitFor(() => expect(isFreeText(container)).toBe(true))
  })
})

describe('the catalog read', () => {
  it('rides the bot’s own route, unforced, and survives a remount from cache', async () => {
    hostMock.requestProfile.mockResolvedValue({ providers: [{ models: ['m1'], name: 'Prov', slug: 'prov' }] })

    const first = mount(remoteBot)

    await waitFor(() => expect(hostMock.requestProfile).toHaveBeenCalledTimes(1))

    expect(hostMock.requestProfile).toHaveBeenCalledWith(remoteBot.route, 'model.options', {
      explicit_only: false,
      include_unconfigured: true
    })
    // A forced refresh here is what re-entered the loading state on every
    // remount and wiped the user's in-progress pick.
    expect(hostMock.requestProfile.mock.calls[0][2]).not.toHaveProperty('refresh')

    first.unmount()

    const second = mount(remoteBot)

    await waitFor(() => expect(second.container.querySelector('[data-testid="spinner"]')).toBeNull())
    expect(hostMock.requestProfile).toHaveBeenCalledTimes(1)
  })

  it('never dispatches for a row whose connection was removed', async () => {
    const orphan = { name: 'ghost', remoteSource: true } as RosterRow
    const { container } = mount(orphan)

    await waitFor(() => expect(isFreeText(container)).toBe(true))
    expect(hostMock.request).not.toHaveBeenCalled()
    expect(hostMock.requestProfile).not.toHaveBeenCalled()
  })

  it('reads the ambient gateway for a local bot', async () => {
    hostMock.request.mockResolvedValue({ providers: [] })

    mount({ name: 'default' } as RosterRow)

    await waitFor(() => expect(hostMock.request).toHaveBeenCalledTimes(1))
    expect(hostMock.request).toHaveBeenCalledWith('model.options', {
      explicit_only: false,
      include_unconfigured: true
    })
  })

  it('shares one cache entry between two bots on the same route', async () => {
    hostMock.requestProfile.mockResolvedValue({ providers: [{ models: ['m1'], slug: 'prov' }] })

    mount(remoteBot)
    await waitFor(() => expect(hostMock.requestProfile).toHaveBeenCalledTimes(1))

    mount({ ...remoteBot, name: 'default' } as RosterRow)
    await waitFor(() => expect(hostMock.requestProfile).toHaveBeenCalledTimes(1))
  })
})
