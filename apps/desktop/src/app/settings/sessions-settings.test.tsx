// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getHermesConfigRecord, listAllProfileSessions, saveHermesConfig, setSessionArchived } from '@/hermes'
import { en } from '@/i18n/en'
import { $messagingSessions, $sessions, setMessagingSessions, setSessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { SessionsSettings } from './sessions-settings'

vi.mock('@/i18n', () => ({ useI18n: () => ({ t: en }) }))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getHermesConfigRecord: vi.fn().mockResolvedValue({ config: {} }),
  listAllProfileSessions: vi.fn(),
  saveHermesConfig: vi.fn().mockResolvedValue({ ok: true }),
  setSessionArchived: vi.fn().mockResolvedValue(undefined)
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
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: {} })
  setSessions([])
  setMessagingSessions([])
  vi.mocked(listAllProfileSessions).mockResolvedValue({ sessions: [archivedMatrixSession], total: 1 } as never)
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
    vi.mocked(listAllProfileSessions).mockResolvedValue({
      sessions: [{ ...archivedMatrixSession, connection_id: 'gateway-b', profile: 'default' }],
      total: 1
    } as never)

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
})
