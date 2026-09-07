import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { $connection } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'

import { PreviewAttachment } from './preview-attachment'

const TILE_SESSION_ID = 'tile-session'

/** Minimal tile view — only the fields PreviewAttachment reads. */
function tileView(): SessionView {
  return {
    ...({} as SessionView),
    $cwd: atom('/tile/work'),
    $storedId: atom<null | string>(TILE_SESSION_ID),
    kind: 'tile'
  }
}

describe('PreviewAttachment download connection scoping', () => {
  const saveGatewayFile = vi.fn(async (_request: Record<string, unknown>) => ({
    path: '/local/downloads/report.md',
    saved: true
  }))

  let originalDesktop: typeof window.hermesDesktop

  beforeEach(() => {
    saveGatewayFile.mockClear()
    originalDesktop = window.hermesDesktop
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { saveGatewayFile } })
    // The window is currently showing a DIFFERENT connection than the one
    // that owns the tile's session — this is the case a background tile can
    // legitimately be in (#478d772f2c only threaded profile/sessionId, never
    // connectionId).
    $connection.set({ connectionId: 'ambient-conn', mode: 'remote', profile: 'ambient-profile' } as never)
    $sessionTiles.set([
      { ownerRoute: { connectionId: 'homelab-ssh', profile: 'writer' }, storedSessionId: TILE_SESSION_ID }
    ] as never)
  })

  afterEach(() => {
    cleanup()
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: originalDesktop })
    $connection.set(null)
    $sessionTiles.set([])
  })

  it('downloads through the session tile owning connection, not the ambient one', async () => {
    render(
      <SessionViewProvider value={tileView()}>
        <PreviewAttachment target="/tile/work/report.md" />
      </SessionViewProvider>
    )

    fireEvent.click(screen.getByRole('button', { name: 'Download' }))

    await waitFor(() => expect(saveGatewayFile).toHaveBeenCalledTimes(1))
    expect(saveGatewayFile).toHaveBeenCalledWith({
      connectionId: 'homelab-ssh',
      path: '/tile/work/report.md',
      profile: 'writer',
      sessionId: TILE_SESSION_ID,
      suggestedName: 'report.md'
    })
  })

  it('still downloads through the ambient connection for the primary (untiled) view', async () => {
    render(<PreviewAttachment target="/primary/work/report.md" />)

    fireEvent.click(screen.getByRole('button', { name: 'Download' }))

    await waitFor(() => expect(saveGatewayFile).toHaveBeenCalledTimes(1))
    expect(saveGatewayFile.mock.calls[0]?.[0]).toMatchObject({
      connectionId: 'ambient-conn',
      profile: 'ambient-profile'
    })
  })
})
