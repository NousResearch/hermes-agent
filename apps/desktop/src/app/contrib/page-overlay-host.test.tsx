import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $routeTiles, closeRouteTile, openRouteTile } from '@/store/route-tiles'

import { PageOverlayHost } from './page-overlay-host'

vi.mock('../overlays/panel', () => ({
  Panel: ({ children }: { children: ReactNode }) => <section data-testid="page-overlay">{children}</section>
}))

vi.mock('../messaging', () => ({
  MessagingView: ({ overlay }: { overlay?: boolean }) => <div data-testid="messaging-view">overlay={String(overlay)}</div>
}))

vi.mock('../artifacts', () => ({ ArtifactsView: () => <div data-testid="artifacts-view" /> }))
vi.mock('../capabilities', () => ({ CapabilitiesView: () => <div data-testid="capabilities-view" /> }))

afterEach(() => {
  cleanup()
  for (const tile of $routeTiles.get()) {
    closeRouteTile(tile.path)
  }
})

beforeEach(() => {
  window.localStorage.removeItem('hermes.desktop.routeTiles.v1')
  $routeTiles.set([])
  openRouteTile('/already-open')
})

describe('PageOverlayHost', () => {
  it('opens the overlay destination in a route tab without replacing an existing tab', async () => {
    const closeOverlay = vi.fn()

    render(<PageOverlayHost onClose={closeOverlay} path="/messaging?filter=all" view="messaging" />)

    expect((await screen.findByTestId('messaging-view')).textContent).toBe('overlay=true')
    fireEvent.click(await screen.findByRole('button', { name: /open in new tab/i }))

    expect($routeTiles.get().map(tile => tile.path)).toEqual(['/already-open', '/messaging'])
    expect(closeOverlay).toHaveBeenCalledOnce()
  })

  it('renders capabilities through the same page-overlay card', async () => {
    render(<PageOverlayHost onClose={vi.fn()} path="/capabilities" view="capabilities" />)

    expect(await screen.findByTestId('capabilities-view')).toBeTruthy()
  })
})
