// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setSessionArchived } from '@/hermes'
import { en } from '@/i18n/en'
import { notifySessionsChanged } from '@/store/live-sync'
import { $messagingSessions, $sessions, setMessagingSessions, setSessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { SessionsSettings } from './sessions-settings'

vi.mock('@/i18n', () => ({ useI18n: () => ({ t: en }) }))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getHermesConfigRecord: vi.fn().mockResolvedValue({ config: {} }),
  setSessionArchived: vi.fn().mockResolvedValue(undefined)
}))

// The settings page loads its archived list through the paginating store helper,
// which reads the sessions API module directly — mock that seam, not the barrel.
const listEveryArchivedSession = vi.hoisted(() => vi.fn())

vi.mock('@/store/sidebar-archive', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  listEveryArchivedSession
}))

const selectDesktopPaths = vi.hoisted(() => vi.fn())

vi.mock('@/lib/desktop-fs', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  selectDesktopPaths
}))

const archivedMatrixSession = {
  archived: true,
  ended_at: null,
  id: 'matrix-1',
  input_tokens: 0,
  is_active: false,
  last_active: 1,
  message_count: 2,
  model: null,
  output_tokens: 0,
  preview: null,
  source: 'matrix',
  started_at: 1,
  title: 'archived room',
  tool_call_count: 0
} as SessionInfo

beforeEach(() => {
  setSessions([])
  setMessagingSessions([])
  listEveryArchivedSession.mockReset().mockResolvedValue([archivedMatrixSession])
})

afterEach(() => {
  cleanup()
})

describe('SessionsSettings unarchive', () => {
  it('restores a messaging-source session into $messagingSessions, not $sessions', async () => {
    render(<SessionsSettings />)
    const button = await screen.findByRole('button', { name: en.settings.sessions.unarchive })

    await act(async () => fireEvent.click(button))

    await waitFor(() => expect(setSessionArchived).toHaveBeenCalledWith('matrix-1', false, undefined))
    expect($messagingSessions.get().map(session => session.id)).toEqual(['matrix-1'])
    expect($messagingSessions.get()[0]?.archived).toBe(false)
    expect($sessions.get()).toEqual([])
  })
})

describe('Archive & retention refresh', () => {
  it.each([
    ['returning to the window', () => window.dispatchEvent(new Event('focus'))],
    ['a gateway session change', notifySessionsChanged]
  ] as const)('updates the count and rows after an external prune on %s', async (_label, refresh) => {
    render(<SessionsSettings subpage="archived" />)
    await screen.findByText('archived room')
    expect(screen.getByText('1')).toBeTruthy()

    listEveryArchivedSession.mockResolvedValue([])
    vi.useFakeTimers()

    try {
      await act(async () => {
        refresh()
        await vi.runOnlyPendingTimersAsync()
      })
    } finally {
      vi.useRealTimers()
    }

    await screen.findByText(en.settings.sessions.emptyArchivedTitle)
    expect(screen.queryByText('archived room')).toBeNull()
    expect(screen.queryByText('1')).toBeNull()
  })

  it('keeps a restored session out of an older archive read', async () => {
    render(<SessionsSettings subpage="archived" />)
    const restore = await screen.findByRole('button', { name: en.settings.sessions.unarchive })
    const inFlight = Promise.withResolvers<SessionInfo[]>()
    listEveryArchivedSession.mockReturnValueOnce(inFlight.promise).mockResolvedValue([])

    fireEvent.focus(window)
    await waitFor(() => expect(listEveryArchivedSession).toHaveBeenCalledTimes(2))
    await act(async () => fireEvent.click(restore))
    await screen.findByText(en.settings.sessions.emptyArchivedTitle)
    await act(async () => inFlight.resolve([archivedMatrixSession]))

    expect(screen.queryByText('archived room')).toBeNull()
  })
})

describe('SessionsSettings default directory in a browser tab', () => {
  afterEach(() => {
    delete document.documentElement.dataset.hermesDesktopHost
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('picks through the server folder browser instead of the native dialog', async () => {
    document.documentElement.dataset.hermesDesktopHost = 'browser'

    const settings = {
      getDefaultProjectDir: vi.fn().mockResolvedValue({ defaultLabel: 'Server workspace', dir: null, resolvedCwd: '' }),
      pickDefaultProjectDir: vi.fn().mockResolvedValue({ canceled: true, dir: null }),
      setDefaultProjectDir: vi.fn(async (dir: null | string) => ({ dir }))
    }

    Object.assign(window, { hermesDesktop: { settings } })
    selectDesktopPaths.mockReset().mockResolvedValue(['/srv/work'])

    render(<SessionsSettings subpage="default-directory" />)
    const button = await screen.findByRole('button', { name: en.common.choose })

    await act(async () => fireEvent.click(button))

    await waitFor(() => expect(settings.setDefaultProjectDir).toHaveBeenCalledWith('/srv/work'))
    expect(selectDesktopPaths).toHaveBeenCalledWith(expect.objectContaining({ directories: true }))
    expect(settings.pickDefaultProjectDir).not.toHaveBeenCalled()
  })
})
