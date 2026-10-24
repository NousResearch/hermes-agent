// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { deleteSession, setSessionArchived } from '@/hermes'
import { en } from '@/i18n/en'
import { $messagingSessions, $sessions, setMessagingSessions, setSessions } from '@/store/session'
import { $removedSessionIds } from '@/store/session-removal'
import { forgetSessionUnread } from '@/store/session-unread'
import { $archivedSessions } from '@/store/sidebar-archive'
import type { SessionInfo } from '@/types/hermes'

import { SessionsSettings } from './sessions-settings'

vi.mock('@/i18n', () => ({ useI18n: () => ({ t: en }) }))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getHermesConfigRecord: vi.fn().mockResolvedValue({ config: {} }),
  deleteSession: vi.fn().mockResolvedValue({ ok: true }),
  setSessionArchived: vi.fn().mockResolvedValue(undefined)
}))

vi.mock('@/store/confirm', () => ({ confirm: vi.fn().mockResolvedValue(true) }))
vi.mock('@/store/session-unread', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  forgetSessionUnread: vi.fn()
}))

// The settings page loads its archived list through the paginating store helper,
// which reads the sessions API module directly — mock that seam, not the barrel.
const listEveryArchivedSession = vi.hoisted(() => vi.fn())

vi.mock('@/store/sidebar-archive', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  listEveryArchivedSession
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
  $archivedSessions.set([])
  $removedSessionIds.set(new Set())
  vi.mocked(forgetSessionUnread).mockReset()
  listEveryArchivedSession.mockReset().mockResolvedValue([archivedMatrixSession])
})

afterEach(() => {
  cleanup()
  setMessagingSessions([])
  $archivedSessions.set([])
  $removedSessionIds.set(new Set())
  vi.restoreAllMocks()
})

describe('SessionsSettings unarchive', () => {
  it('evicts a deleted row from shared sidebar stores even if unread cleanup fails', async () => {
    setMessagingSessions([archivedMatrixSession])
    $archivedSessions.set([archivedMatrixSession])
    vi.mocked(forgetSessionUnread).mockImplementation(() => {
      throw new Error('unread storage unavailable')
    })
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    render(<SessionsSettings />)
    const button = await screen.findByRole('button', { name: en.settings.sessions.deletePermanently })

    await act(async () => fireEvent.click(button))

    expect(deleteSession).toHaveBeenCalledWith(archivedMatrixSession.id, undefined)
    expect($messagingSessions.get()).toEqual([])
    expect($archivedSessions.get()).toEqual([])
    expect($removedSessionIds.get().has(archivedMatrixSession.id)).toBe(true)
    expect(screen.queryByRole('button', { name: en.settings.sessions.deletePermanently })).toBeNull()
  })

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
