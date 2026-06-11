import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { SidebarSessionRow } from './session-row'

const baseSession: SessionInfo = {
  archived: false,
  cwd: null,
  ended_at: null,
  id: 'session-1',
  input_tokens: 0,
  is_active: false,
  last_active: 1_780_000_000,
  message_count: 3,
  model: null,
  output_tokens: 0,
  preview: null,
  source: 'cli',
  started_at: 1_780_000_000,
  title: 'Compressed work',
  tool_call_count: 0
}

function renderRow(session: SessionInfo) {
  render(
    <SidebarSessionRow
      isPinned={false}
      isSelected={false}
      isWorking={false}
      onArchive={vi.fn()}
      onDelete={vi.fn()}
      onPin={vi.fn()}
      onResume={vi.fn()}
      session={session}
    />
  )
}

afterEach(() => {
  cleanup()
})

describe('SidebarSessionRow', () => {
  it('shows compression count for compressed sessions', () => {
    renderRow({ ...baseSession, compress_count: 2 })

    expect(screen.getByTitle('Compressed 2 times').textContent).toBe('x2')
  })

  it('hides compression count for uncompressed sessions', () => {
    renderRow(baseSession)

    expect(screen.queryByTitle(/Compressed/)).toBeNull()
  })
})
