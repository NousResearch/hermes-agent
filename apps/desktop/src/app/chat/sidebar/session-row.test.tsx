import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $replyReadySessionIds, $replyReadySessionProfiles } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { SidebarSessionRow } from './session-row'

const session = (overrides: Partial<SessionInfo> = {}): SessionInfo => ({
  ended_at: null,
  id: 'tip-1',
  input_tokens: 0,
  is_active: false,
  last_active: 10,
  message_count: 3,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: 1,
  title: 'Needs review',
  tool_call_count: 0,
  ...overrides
})

describe('SidebarSessionRow reply-ready acknowledgement', () => {
  afterEach(() => {
    $replyReadySessionIds.set([])
    $replyReadySessionProfiles.set({})
  })

  it('clears reply-ready markers before resuming the clicked row', () => {
    const onResume = vi.fn()

    $replyReadySessionIds.set(['root-1'])
    $replyReadySessionProfiles.set({ 'root-1': 'hermes-dev' })

    render(
      <SidebarSessionRow
        isPinned={false}
        isSelected
        isWorking={false}
        onArchive={vi.fn()}
        onDelete={vi.fn()}
        onPin={vi.fn()}
        onResume={onResume}
        session={session({ _lineage_root_id: 'root-1' })}
      />
    )

    fireEvent.click(screen.getAllByRole('button', { name: /needs review/i })[0])

    expect($replyReadySessionIds.get()).toEqual([])
    expect($replyReadySessionProfiles.get()).toEqual({})
    expect(onResume).toHaveBeenCalledTimes(1)
  })
})
