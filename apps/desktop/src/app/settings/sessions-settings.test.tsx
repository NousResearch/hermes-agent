// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { deleteSession, getHermesConfigRecord, saveHermesConfig, setSessionArchived } from '@/hermes'
import { en } from '@/i18n/en'
import { confirm } from '@/store/confirm'
import { $messagingSessions, $sessions, setMessagingSessions, setSessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { SessionsSettings } from './sessions-settings'

const { notifyError } = vi.hoisted(() => ({ notifyError: vi.fn() }))

vi.mock('@/i18n', () => ({ useI18n: () => ({ t: en }) }))
vi.mock('@/store/confirm', () => ({ confirm: vi.fn() }))
vi.mock('@/store/notifications', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  notifyError
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  deleteSession: vi.fn().mockResolvedValue({ ok: true }),
  getHermesConfigRecord: vi.fn().mockResolvedValue({ config: {} }),
  saveHermesConfig: vi.fn().mockResolvedValue({ ok: true }),
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
  vi.clearAllMocks()
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: {} })
  setSessions([])
  setMessagingSessions([])
  listEveryArchivedSession.mockReset().mockResolvedValue([archivedMatrixSession])
  vi.mocked(confirm).mockResolvedValue(true)
})

afterEach(() => {
  cleanup()
  Reflect.deleteProperty(window, 'hermesDesktop')
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

  it('routes a connection-tagged row back to its exact owner', async () => {
    listEveryArchivedSession.mockResolvedValue([
      { ...archivedMatrixSession, connection_id: 'gateway-b', profile: 'default' }
    ])

    render(<SessionsSettings />)
    const button = await screen.findByRole('button', { name: en.settings.sessions.unarchive })

    await act(async () => fireEvent.click(button))

    await waitFor(() =>
      expect(setSessionArchived).toHaveBeenCalledWith('matrix-1', false, {
        connectionId: 'gateway-b',
        profile: 'default'
      })
    )
  })

  it('unarchives only the clicked owner when same-id rows coexist', async () => {
    const visibleOwnerB = {
      ...archivedMatrixSession,
      archived: false,
      connection_id: 'gateway-b',
      profile: 'default'
    }

    setMessagingSessions([visibleOwnerB])
    listEveryArchivedSession.mockResolvedValue([
      { ...archivedMatrixSession, connection_id: 'gateway-a', profile: 'default' },
      { ...archivedMatrixSession, connection_id: 'gateway-b', profile: 'default' }
    ])

    render(<SessionsSettings />)
    const buttons = await screen.findAllByRole('button', { name: en.settings.sessions.unarchive })

    await act(async () => fireEvent.click(buttons[0]!))

    await waitFor(() => expect(screen.getAllByRole('button', { name: en.settings.sessions.unarchive })).toHaveLength(1))
    expect(setSessionArchived).toHaveBeenCalledWith('matrix-1', false, {
      connectionId: 'gateway-a',
      profile: 'default'
    })
    expect($messagingSessions.get()).toEqual([
      expect.objectContaining({ connection_id: 'gateway-a', id: 'matrix-1' }),
      visibleOwnerB
    ])
  })

  it('keeps each concurrent owner row busy until its own mutation settles', async () => {
    const resolvers = new Map<string, () => void>()
    vi.mocked(setSessionArchived).mockImplementation(
      (_id, _archived, scope) =>
        new Promise<{ ok: boolean }>(resolve => {
          const connectionId = scope && typeof scope === 'object' ? scope.connectionId : null
          resolvers.set(connectionId ?? 'legacy', () => resolve({ ok: true }))
        })
    )
    listEveryArchivedSession.mockResolvedValue([
      { ...archivedMatrixSession, connection_id: 'gateway-a', profile: 'default' },
      { ...archivedMatrixSession, connection_id: 'gateway-b', profile: 'default' }
    ])

    render(<SessionsSettings />)
    const buttons = await screen.findAllByRole('button', { name: en.settings.sessions.unarchive })
    fireEvent.click(buttons[0]!)
    fireEvent.click(buttons[1]!)

    await waitFor(() => expect(buttons.every(button => button.hasAttribute('disabled'))).toBe(true))
    await act(async () => resolvers.get('gateway-a')?.())

    const remaining = await screen.findByRole('button', { name: en.settings.sessions.unarchive })
    expect(remaining.hasAttribute('disabled')).toBe(true)

    await act(async () => resolvers.get('gateway-b')?.())
    await waitFor(() => expect(screen.queryByRole('button', { name: en.settings.sessions.unarchive })).toBeNull())
  })

  it('deletes only the clicked owner when same-id rows coexist', async () => {
    listEveryArchivedSession.mockResolvedValue([
      { ...archivedMatrixSession, connection_id: 'gateway-a', profile: 'default' },
      { ...archivedMatrixSession, connection_id: 'gateway-b', profile: 'default' }
    ])

    render(<SessionsSettings />)
    const buttons = await screen.findAllByRole('button', { name: en.settings.sessions.deletePermanently })

    await act(async () => fireEvent.click(buttons[0]!))

    await waitFor(() =>
      expect(screen.getAllByRole('button', { name: en.settings.sessions.deletePermanently })).toHaveLength(1)
    )
    expect(deleteSession).toHaveBeenCalledWith('matrix-1', {
      connectionId: 'gateway-a',
      profile: 'default'
    })
  })

  it('pins the archived list and unarchive mutation to the immutable settings owner', async () => {
    const owner = {
      connectionId: 'gateway-b',
      profile: 'default',
      connectionOwner: { baseUrl: 'http://127.0.0.1:9001', mode: 'local' as const, token: 'synthetic-token' }
    }

    listEveryArchivedSession.mockResolvedValue([{ ...archivedMatrixSession, profile: 'research' }])

    render(<SessionsSettings settingsOwner={owner} />)
    const button = await screen.findByRole('button', { name: en.settings.sessions.unarchive })

    expect(listEveryArchivedSession).toHaveBeenCalledWith(owner)
    await act(async () => fireEvent.click(button))

    await waitFor(() =>
      expect(setSessionArchived).toHaveBeenCalledWith('matrix-1', false, {
        ...owner,
        connectionOwnerProfile: 'default',
        profile: 'research'
      })
    )
  })

  it('uses the tagged row owner when an aggregate Settings list includes another gateway', async () => {
    const settingsOwner = {
      connectionId: 'gateway-a',
      profile: 'default',
      connectionOwner: { baseUrl: 'http://127.0.0.1:9001', mode: 'local' as const, token: 'synthetic-token-a' }
    }

    listEveryArchivedSession.mockResolvedValue([
      {
        ...archivedMatrixSession,
        _desktop_route_binding: 'synthetic-binding-b',
        connection_id: 'gateway-b',
        profile: 'research'
      }
    ])

    render(<SessionsSettings settingsOwner={settingsOwner} />)
    const button = await screen.findByRole('button', { name: en.settings.sessions.unarchive })

    await act(async () => fireEvent.click(button))

    await waitFor(() =>
      expect(setSessionArchived).toHaveBeenCalledWith('matrix-1', false, {
        connectionId: 'gateway-b',
        profile: 'research',
        sessionRouteBinding: 'synthetic-binding-b',
        sessionRouteId: 'matrix-1'
      })
    )
  })

  it('keeps a binding-only legacy row on its original route in an aggregate Settings list', async () => {
    const settingsOwner = {
      connectionId: 'gateway-a',
      profile: 'default',
      connectionOwner: { baseUrl: 'http://127.0.0.1:9001', mode: 'local' as const, token: 'synthetic-token-a' }
    }

    listEveryArchivedSession.mockResolvedValue([
      {
        ...archivedMatrixSession,
        _desktop_route_binding: 'synthetic-legacy-binding',
        profile: 'research'
      }
    ])

    render(<SessionsSettings settingsOwner={settingsOwner} />)
    const button = await screen.findByRole('button', { name: en.settings.sessions.unarchive })

    await act(async () => fireEvent.click(button))

    await waitFor(() =>
      expect(setSessionArchived).toHaveBeenCalledWith('matrix-1', false, {
        connectionId: null,
        profile: 'research',
        sessionRouteBinding: 'synthetic-legacy-binding',
        sessionRouteId: 'matrix-1'
      })
    )
  })

  it('pins permanent delete to the immutable settings owner', async () => {
    const owner = {
      connectionId: 'gateway-b',
      profile: 'default',
      connectionOwner: { baseUrl: 'http://127.0.0.1:9001', mode: 'local' as const, token: 'synthetic-token' }
    }

    listEveryArchivedSession.mockResolvedValue([{ ...archivedMatrixSession, profile: 'research' }])

    render(<SessionsSettings settingsOwner={owner} />)
    const button = await screen.findByRole('button', { name: en.settings.sessions.deletePermanently })

    await act(async () => fireEvent.click(button))

    await waitFor(() =>
      expect(deleteSession).toHaveBeenCalledWith('matrix-1', {
        ...owner,
        connectionOwnerProfile: 'default',
        profile: 'research'
      })
    )
  })

  it('permanently deletes a binding-only legacy row through its original route', async () => {
    const settingsOwner = {
      connectionId: 'gateway-a',
      profile: 'default',
      connectionOwner: { baseUrl: 'http://127.0.0.1:9001', mode: 'local' as const, token: 'synthetic-token-a' }
    }

    listEveryArchivedSession.mockResolvedValue([
      {
        ...archivedMatrixSession,
        _desktop_route_binding: 'synthetic-legacy-binding',
        profile: 'research'
      }
    ])

    render(<SessionsSettings settingsOwner={settingsOwner} />)
    const button = await screen.findByRole('button', { name: en.settings.sessions.deletePermanently })

    await act(async () => fireEvent.click(button))

    await waitFor(() =>
      expect(deleteSession).toHaveBeenCalledWith('matrix-1', {
        connectionId: null,
        profile: 'research',
        sessionRouteBinding: 'synthetic-legacy-binding',
        sessionRouteId: 'matrix-1'
      })
    )
  })

  it('does not dispatch a confirmed delete after Settings changes owner', async () => {
    let resolveConfirm!: (value: boolean) => void

    vi.mocked(confirm).mockReturnValueOnce(new Promise(resolve => (resolveConfirm = resolve)))

    const ownerA = {
      connectionId: 'gateway-a',
      profile: 'default',
      connectionOwner: { baseUrl: 'http://127.0.0.1:9001', mode: 'local' as const, token: 'token-a' }
    }

    const ownerB = {
      connectionId: 'gateway-b',
      profile: 'default',
      connectionOwner: { baseUrl: 'http://127.0.0.1:9002', mode: 'local' as const, token: 'token-b' }
    }

    const view = render(<SessionsSettings settingsOwner={ownerA} />)

    fireEvent.click(await screen.findByRole('button', { name: en.settings.sessions.deletePermanently }))

    view.rerender(<SessionsSettings settingsOwner={ownerB} />)
    await act(async () => resolveConfirm(true))

    expect(deleteSession).not.toHaveBeenCalled()
  })

  it('does not publish a completed unarchive into a replacement owner', async () => {
    let resolveUnarchive!: (value: { ok: boolean }) => void

    vi.mocked(setSessionArchived).mockReturnValueOnce(new Promise(resolve => (resolveUnarchive = resolve)))

    const ownerA = {
      connectionId: 'gateway-a',
      profile: 'default',
      connectionOwner: { baseUrl: 'http://127.0.0.1:9001', mode: 'local' as const, token: 'token-a' }
    }

    const ownerB = {
      connectionId: 'gateway-b',
      profile: 'default',
      connectionOwner: { baseUrl: 'http://127.0.0.1:9002', mode: 'local' as const, token: 'token-b' }
    }

    const view = render(<SessionsSettings settingsOwner={ownerA} />)

    fireEvent.click(await screen.findByRole('button', { name: en.settings.sessions.unarchive }))

    view.rerender(<SessionsSettings settingsOwner={ownerB} />)
    await act(async () => resolveUnarchive({ ok: true }))

    expect($messagingSessions.get()).toEqual([])
    expect($sessions.get()).toEqual([])
  })
})

describe('SessionsSettings auto archive', () => {
  it('pins config reads and writes to the selected settings owner', async () => {
    const owner = {
      connectionId: 'gateway-b',
      profile: 'profile-b',
      connectionOwner: { baseUrl: 'http://127.0.0.1:9001', mode: 'local' as const, token: 'synthetic-token' }
    }

    vi.mocked(getHermesConfigRecord).mockResolvedValue({ sessions: { auto_archive: false } } as never)

    render(<SessionsSettings settingsOwner={owner} />)

    await waitFor(() => expect(getHermesConfigRecord).toHaveBeenCalledWith(owner))
    fireEvent.click(await screen.findByRole('switch', { name: en.settings.sessions.autoArchiveTitle }))

    await waitFor(() =>
      expect(saveHermesConfig).toHaveBeenCalledWith(
        { sessions: { auto_archive: true, auto_archive_days: 3 } },
        owner
      )
    )
  })

  it('does not publish a failed save after Settings changes owner', async () => {
    let rejectSave!: (error: Error) => void

    const ownerA = {
      connectionId: 'gateway-a',
      profile: 'default',
      connectionOwner: { baseUrl: 'http://127.0.0.1:9001', mode: 'local' as const, token: 'token-a' }
    }

    const ownerB = {
      connectionId: 'gateway-b',
      profile: 'default',
      connectionOwner: { baseUrl: 'http://127.0.0.1:9002', mode: 'local' as const, token: 'token-b' }
    }

    vi.mocked(getHermesConfigRecord).mockResolvedValue({ sessions: { auto_archive: false } } as never)
    vi.mocked(saveHermesConfig).mockReturnValueOnce(new Promise((_resolve, reject) => (rejectSave = reject)))

    const view = render(<SessionsSettings settingsOwner={ownerA} />)
    fireEvent.click(await screen.findByRole('switch', { name: en.settings.sessions.autoArchiveTitle }))
    await waitFor(() => expect(saveHermesConfig).toHaveBeenCalledWith(expect.anything(), ownerA))

    view.rerender(<SessionsSettings settingsOwner={ownerB} />)
    await act(async () => rejectSave(new Error('owner A failed')))

    expect(notifyError).not.toHaveBeenCalled()
  })
})

describe('SessionsSettings default directory in a browser tab', () => {
  afterEach(() => {
    delete globalThis.document.documentElement.dataset.hermesDesktopHost
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('picks through the server folder browser instead of the native dialog', async () => {
    globalThis.document.documentElement.dataset.hermesDesktopHost = 'browser'

    const settings = {
      getDefaultProjectDir: vi.fn().mockResolvedValue({ defaultLabel: 'Server workspace', dir: null, resolvedCwd: '' }),
      pickDefaultProjectDir: vi.fn().mockResolvedValue({ canceled: true, dir: null }),
      setDefaultProjectDir: vi.fn(async (dir: null | string) => ({ dir }))
    }

    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { settings } })
    selectDesktopPaths.mockReset().mockResolvedValue(['/srv/work'])

    render(<SessionsSettings subpage="default-directory" />)
    const button = await screen.findByRole('button', { name: en.common.choose })

    await act(async () => fireEvent.click(button))

    await waitFor(() => expect(settings.setDefaultProjectDir).toHaveBeenCalledWith('/srv/work'))
    expect(selectDesktopPaths).toHaveBeenCalledWith(expect.objectContaining({ directories: true }))
    expect(settings.pickDefaultProjectDir).not.toHaveBeenCalled()
  })
})
