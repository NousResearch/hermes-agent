import { cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ComponentProps, ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ComposerStatusStack } from '@/app/chat/composer/status-stack'
import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { TranscriptIdentityProvider } from '@/components/assistant-ui/thread/transcript-identity'
import { $previewStatusBySession, recordPreviewArtifact } from '@/store/preview-status'
import { $activeSessionId, $currentCwd } from '@/store/session'
import { $subagentsBySession, upsertSubagent } from '@/store/subagents'

vi.mock('@assistant-ui/react', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useAuiState: (select: (state: unknown) => unknown) =>
    select({ message: { id: 'msg-1', status: { type: 'complete' } }, thread: { isRunning: false } })
}))

const { ToolFallback } = await import('./fallback')

const PRIMARY_ID = 'primary-session'
const TILE_ID = 'tile-session'

/** Minimal tile view: only the fields the tool row reads. */
function tileView(): SessionView {
  return {
    ...({} as SessionView),
    $cwd: atom('/tile/work'),
    $messages: atom([]),
    $runtimeId: atom<null | string>(TILE_ID),
    kind: 'tile'
  }
}

function renderToolRow(wrap: (node: ReactNode) => ReactNode, path = '/tile/work/report.html') {
  const props = {
    args: { path },
    result: { path },
    toolCallId: 'call-1',
    toolName: 'write_file'
  } as unknown as ComponentProps<typeof ToolFallback>

  render(<>{wrap(<ToolFallback {...props} />)}</>)
}

afterEach(() => {
  cleanup()
  $previewStatusBySession.set({})
  $subagentsBySession.set({})
  $activeSessionId.set(null)
  $currentCwd.set('')
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

    renderToolRow(node => node)

    expect(Object.keys($previewStatusBySession.get())).toEqual([PRIMARY_ID])
  })

  it('does not attribute a previous project artifact to the destination session during navigation', () => {
    const projectAArtifact = '/project-a/report.html'

    $activeSessionId.set(PRIMARY_ID)
    $currentCwd.set('/project-a')
    renderToolRow(node => node, projectAArtifact)

    expect($previewStatusBySession.get()[PRIMARY_ID]?.map(item => item.label)).toEqual(['report.html'])

    // A session transition can publish the destination identity before its
    // transcript replaces the previous project's rows. Remount the still-stale
    // row under that intermediate state: it must retain session A ownership.
    cleanup()
    $activeSessionId.set('destination-session')
    $currentCwd.set('/project-b')
    renderToolRow(
      node => (
        <TranscriptIdentityProvider value={{ cwd: '/project-a', runtimeId: PRIMARY_ID }}>
          {node}
        </TranscriptIdentityProvider>
      ),
      projectAArtifact
    )

    upsertSubagent('destination-session', {
      goal: 'Inspect destination project',
      status: 'running',
      subagent_id: 'destination-agent',
      task_index: 0
    })

    cleanup()
    render(
      <MemoryRouter>
        <ComposerStatusStack queue={null} sessionId="destination-session" />
      </MemoryRouter>
    )

    expect(screen.getByRole('button', { name: /1 Subagent/ })).toBeTruthy()
    expect(screen.queryByText('report.html')).toBeNull()

    recordPreviewArtifact('destination-session', '/project-b/destination.html', '/project-b')
    cleanup()
    render(
      <MemoryRouter>
        <ComposerStatusStack queue={null} sessionId="destination-session" />
      </MemoryRouter>
    )

    const subagent = screen.getByRole('button', { name: /1 Subagent/ })
    const destinationArtifact = screen.getByText('destination.html')

    expect(subagent.compareDocumentPosition(destinationArtifact) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()

    cleanup()
    render(
      <MemoryRouter>
        <ComposerStatusStack queue={null} sessionId={PRIMARY_ID} />
      </MemoryRouter>
    )

    expect(screen.getByText('report.html')).toBeTruthy()
    expect(screen.queryByText('destination.html')).toBeNull()
  })
})
