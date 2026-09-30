import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { $previewTabs, closeRightRail, openPreview, previewTabId } from '@/store/preview'
import { $connection } from '@/store/session'

import { watchPreviewTiles } from './preview-tile'

vi.mock('./right-rail/real-profile-consent-dialog', () => ({
  RealProfileConsentDialog: () => null
}))

// #87411: a persisted preview tab pointing at a deleted file rendered
// "Preview unavailable" with nothing to click. The body's Close must be the
// tab's own Close — the same verb the strip's ✕ and ⌘W run.

const target = {
  kind: 'file' as const,
  label: 'report.csv',
  path: '/tmp/gone/report.csv',
  previewKind: 'text' as const,
  source: '/tmp/gone/report.csv',
  url: 'file:///tmp/gone/report.csv'
}

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }

beforeAll(() => {
  watchPreviewTiles()
})

beforeEach(() => {
  $connection.set({ mode: 'local' } as never)
  desktopWindow.hermesDesktop = {
    readFileText: vi.fn(async () => {
      throw new Error('Text preview failed: file does not exist.')
    })
  } as unknown as Window['hermesDesktop']
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
    window.setTimeout(() => callback(Date.now()), 0)
  )
  vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))
})

afterEach(() => {
  cleanup()
  closeRightRail()
  $connection.set(null)
  delete desktopWindow.hermesDesktop
  vi.unstubAllGlobals()
})

describe('a preview whose file is gone', () => {
  it('offers Close in the error state, and it closes the tab', async () => {
    openPreview(target)

    const tabId = previewTabId(target)
    const pane = registry.getArea('panes').find(contribution => contribution.id === `preview-tile:${tabId}`)

    expect(pane?.render).toBeTypeOf('function')

    await act(async () => {
      render(<>{pane!.render!()}</>)
    })

    expect(await screen.findByText('Preview unavailable')).toBeTruthy()
    expect($previewTabs.get().map(tab => tab.id)).toEqual([tabId])

    fireEvent.click(screen.getByRole('button', { name: 'Close' }))

    expect($previewTabs.get()).toEqual([])
  })
})
