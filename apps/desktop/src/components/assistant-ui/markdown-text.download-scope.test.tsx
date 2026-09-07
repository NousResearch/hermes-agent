import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { $connection } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'

import { MarkdownImage } from './markdown-text'

const TILE_SESSION_ID = 'tile-session'
const IMAGE_PATH = '/tile/work/photo.png'

/** Minimal tile view — only the fields the media components read. */
function tileView(): SessionView {
  return {
    ...({} as SessionView),
    $cwd: atom('/tile/work'),
    $storedId: atom<null | string>(TILE_SESSION_ID),
    kind: 'tile'
  }
}

// Regression: an image (or audio/video) rendered inline in a chat message can
// live in a background session tile pinned to a DIFFERENT connection than
// whichever one this window currently shows. The "Open image"/"Open <kind>
// file" fallback shown after a failed inline load must retry the download
// against THAT session's owning connection, not the ambient `$connection`
// (478d772f2c threaded sessionId/profile through the Artifacts page only —
// this call site, useOpenMediaFile, had no origin at all).
describe('MarkdownImage failed-load download connection scoping', () => {
  const saveGatewayFile = vi.fn(async (_request: Record<string, unknown>) => ({
    path: '/local/downloads/photo.png',
    saved: true
  }))

  const api = vi.fn().mockRejectedValue(new Error('boom'))
  let originalDesktop: typeof window.hermesDesktop

  beforeEach(() => {
    saveGatewayFile.mockClear()
    api.mockClear()
    originalDesktop = window.hermesDesktop
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api, saveGatewayFile } })
    // The window is currently showing a DIFFERENT connection than the one
    // that owns the tile's session.
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

  it('retries the download through the tile owning connection, not the ambient one', async () => {
    render(
      <SessionViewProvider value={tileView()}>
        <MarkdownImage alt="pic" src={IMAGE_PATH} />
      </SessionViewProvider>
    )

    const button = await screen.findByRole('button', { name: 'Open image' })

    fireEvent.click(button)

    await waitFor(() => expect(saveGatewayFile).toHaveBeenCalledTimes(1))
    expect(saveGatewayFile).toHaveBeenCalledWith({
      connectionId: 'homelab-ssh',
      path: IMAGE_PATH,
      profile: 'writer',
      sessionId: TILE_SESSION_ID,
      suggestedName: 'photo.png'
    })
  })

  it('falls back to the ambient connection for the primary (untiled) view', async () => {
    render(<MarkdownImage alt="pic" src={IMAGE_PATH} />)

    const button = await screen.findByRole('button', { name: 'Open image' })

    fireEvent.click(button)

    await waitFor(() => expect(saveGatewayFile).toHaveBeenCalledTimes(1))
    expect(saveGatewayFile.mock.calls[0]?.[0]).toMatchObject({
      connectionId: 'ambient-conn',
      profile: 'ambient-profile'
    })
  })
})
