// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  confirm: vi.fn(),
  deleteSession: vi.fn(),
  listAllProfileSessions: vi.fn(),
  setSessionArchived: vi.fn()
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  deleteSession: (...args: unknown[]) => mocks.deleteSession(...args),
  listAllProfileSessions: (...args: unknown[]) => mocks.listAllProfileSessions(...args),
  setSessionArchived: (...args: unknown[]) => mocks.setSessionArchived(...args)
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      settings: {
        sessions: {
          archivedIntro: 'Archived intro',
          archivedTitle: 'Archived',
          autoArchiveDaysLabel: 'days',
          autoArchiveDaysUnit: 'd',
          autoArchiveDesc: 'desc',
          autoArchiveFailed: 'auto failed',
          autoArchiveTitle: 'Auto',
          change: 'Change',
          choose: 'Choose',
          clear: 'Clear',
          clearDirFailed: 'clear failed',
          defaultDirDesc: 'dir desc',
          defaultDirTitle: 'dir title',
          defaultsTo: (dir: string) => dir,
          deleteConfirm: (title: string) => `Delete ${title}?`,
          deleteFailed: 'delete failed',
          deletePermanently: 'Delete permanently',
          emptyArchivedDesc: 'empty desc',
          emptyArchivedTitle: 'empty',
          failedLoad: 'load failed',
          getDefaultProjectDir: 'get dir',
          messages: (n: number) => `${n} messages`,
          notSet: 'not set',
          pickDefaultProjectDir: 'pick',
          restored: 'restored',
          setDefaultProjectDir: 'set dir',
          unarchive: 'Unarchive',
          unarchiveFailed: 'unarchive failed',
          updateDirFailed: 'update failed'
        }
      }
    }
  })
}))

vi.mock('@/store/confirm', () => ({
  confirm: (...args: unknown[]) => mocks.confirm(...args)
}))

import type { SessionInfo } from '@/types/hermes'

import { SessionsSettings } from './sessions-settings'

function archivedRow(over: Partial<SessionInfo> = {}): SessionInfo {
  return {
    archived: true,
    cwd: null,
    ended_at: null,
    id: 'arch-1',
    input_tokens: 0,
    is_active: false,
    last_active: 1,
    message_count: 2,
    model: null,
    output_tokens: 0,
    parent_session_id: null,
    preview: null,
    profile: 'work',
    source: 'desktop',
    started_at: 1,
    title: 'archived chat',
    tool_call_count: 0,
    ...over
  } as SessionInfo
}

async function renderWith(rows: SessionInfo[]) {
  mocks.listAllProfileSessions.mockResolvedValue({ sessions: rows })
  render(<SessionsSettings />)
  await waitFor(() => expect(mocks.listAllProfileSessions).toHaveBeenCalled())
  await screen.findByText('archived chat')
}

describe('SessionsSettings owner scope', () => {
  beforeEach(() => {
    mocks.confirm.mockResolvedValue(true)
    mocks.deleteSession.mockResolvedValue({ ok: true })
    mocks.setSessionArchived.mockResolvedValue({ ok: true })
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('unarchives a connection-tagged row against its exact scope', async () => {
    await renderWith([archivedRow({ connection_id: 'source-a' })])

    await act(async () => {
      fireEvent.click(screen.getByText('Unarchive'))
    })

    expect(mocks.setSessionArchived).toHaveBeenCalledWith('arch-1', false, {
      connectionId: 'source-a',
      profile: 'work'
    })
  })

  it('permanently deletes a connection-tagged row against its exact scope', async () => {
    await renderWith([archivedRow({ connection_id: 'source-a' })])

    await act(async () => {
      fireEvent.click(screen.getByLabelText('Delete permanently'))
    })

    expect(mocks.deleteSession).toHaveBeenCalledWith('arch-1', {
      connectionId: 'source-a',
      profile: 'work'
    })
  })

  it('keeps the bare profile form for untagged rows', async () => {
    await renderWith([archivedRow()])

    await act(async () => {
      fireEvent.click(screen.getByText('Unarchive'))
    })

    expect(mocks.setSessionArchived).toHaveBeenCalledWith('arch-1', false, 'work')
  })
})
