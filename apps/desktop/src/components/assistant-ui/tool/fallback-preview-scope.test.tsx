import { cleanup, render } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ComponentProps, ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import type { ChatMessage } from '@/lib/chat-messages'
import { $previewStatusBySession } from '@/store/preview-status'
import { $activeSessionId, $currentCwd, $messages } from '@/store/session'

vi.mock('@assistant-ui/react', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useAuiState: (select: (state: unknown) => unknown) =>
    select({ message: { id: 'msg-1', status: { type: 'complete' } }, thread: { isRunning: false } })
}))

const { ToolFallback } = await import('./fallback')

const PRIMARY_ID = 'primary-session'
const TILE_ID = 'tile-session'
const messages: ChatMessage[] = [{ id: 'msg-1', role: 'assistant', parts: [] }]

/** Minimal tile view: only the fields the tool row reads. */
function tileView(): SessionView {
  return {
    ...({} as SessionView),
    $cwd: atom('/tile/work'),
    $messages: atom(messages),
    $runtimeId: atom<null | string>(TILE_ID),
    kind: 'tile'
  }
}

function renderToolRow(wrap: (node: ReactNode) => ReactNode, overrides: Record<string, unknown> = {}) {
  const props = {
    args: { path: '/tile/work/report.html' },
    result: { path: '/tile/work/report.html' },
    toolCallId: 'call-1',
    toolName: 'write_file',
    ...overrides
  } as unknown as ComponentProps<typeof ToolFallback>

  render(<>{wrap(<ToolFallback {...props} />)}</>)
}

afterEach(() => {
  cleanup()
  $previewStatusBySession.set({})
  $activeSessionId.set(null)
  $currentCwd.set('')
  $messages.set([])
})

describe('tool row preview recording', () => {
  // The row used to record under the global (primary-only) $activeSessionId, so
  // a preview produced inside a session TILE surfaced in the main chat's
  // composer instead of the tile's own.
  it('records into the session whose transcript the row is in, not the primary', () => {
    $activeSessionId.set(PRIMARY_ID)
    $currentCwd.set('/primary/work')

    const view = tileView()

    renderToolRow(node => <SessionViewProvider value={view}>{node}</SessionViewProvider>)

    const recorded = $previewStatusBySession.get()

    expect(Object.keys(recorded)).toEqual([TILE_ID])
    expect(recorded[TILE_ID]?.[0]?.cwd).toBe('/tile/work')
  })

  it('still records into the primary session for the main chat', () => {
    $activeSessionId.set(PRIMARY_ID)
    $currentCwd.set('/primary/work')
    $messages.set(messages)

    renderToolRow(node => node)

    expect(Object.keys($previewStatusBySession.get())).toEqual([PRIMARY_ID])
  })

  it('does not promote reads, failed writes or packaged renderer URLs into artifacts', () => {
    $activeSessionId.set(PRIMARY_ID)
    $messages.set(messages)

    for (const overrides of [
      { toolName: 'read_file' },
      { isError: true, result: { error: 'Permission denied' } },
      { args: { path: '/work/missing.html' }, result: undefined },
      { result: { preview: 'file:///opt/Hermes/resources/app.asar/dist/index.html' } }
    ]) {
      renderToolRow(node => node, overrides)
      expect($previewStatusBySession.get()[PRIMARY_ID]).toBeUndefined()
      cleanup()
    }

    renderToolRow(node => node)
    expect($previewStatusBySession.get()[PRIMARY_ID]).toHaveLength(1)
  })

  it('does not register a previous conversation row under the newly selected chat', () => {
    $activeSessionId.set('next-conversation')
    $messages.set([{ id: 'next-message', role: 'assistant', parts: [] }])
    renderToolRow(node => node)
    expect($previewStatusBySession.get()['next-conversation']).toBeUndefined()
  })
})
