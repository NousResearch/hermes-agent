import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { ArchivedTranscript } from './archived-transcript'

const { getSessionMessages } = vi.hoisted(() => ({ getSessionMessages: vi.fn() }))

vi.mock('@/hermes', () => ({ getSessionMessages }))

describe('ArchivedTranscript', () => {
  beforeEach(() => {
    getSessionMessages.mockReset()
  })

  it('loads compacted history on demand and pages older messages', async () => {
    getSessionMessages
      .mockResolvedValueOnce({
        session_id: 'session-1',
        messages: [
          { id: 2, content: 'older visible prompt', role: 'user' },
          { id: 3, content: 'archived answer', role: 'assistant' }
        ],
        pagination: {
          has_more: true,
          limit: 100,
          offset: 0,
          order: 'latest',
          returned: 2,
          scope: 'compacted'
        }
      })
      .mockResolvedValueOnce({
        session_id: 'session-1',
        messages: [{ id: 1, content: 'oldest archived prompt', role: 'user' }],
        pagination: {
          has_more: false,
          limit: 100,
          offset: 2,
          order: 'latest',
          returned: 1,
          scope: 'compacted'
        }
      })

    render(<ArchivedTranscript profile="default" sessionId="session-1" />)

    expect(screen.queryByText('older visible prompt')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: /Archived history/ }))

    await waitFor(() => expect(screen.getByText('older visible prompt')).toBeTruthy())
    expect(getSessionMessages).toHaveBeenCalledWith('session-1', 'default', {
      limit: 100,
      offset: 0,
      order: 'latest',
      scope: 'compacted'
    })

    fireEvent.click(screen.getByRole('button', { name: 'Load older archived messages' }))

    await waitFor(() => expect(screen.getByText('oldest archived prompt')).toBeTruthy())
    expect(getSessionMessages).toHaveBeenLastCalledWith('session-1', 'default', {
      limit: 100,
      offset: 2,
      order: 'latest',
      scope: 'compacted'
    })
  })
})
