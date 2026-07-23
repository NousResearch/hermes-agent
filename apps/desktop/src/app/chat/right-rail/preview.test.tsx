import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { RIGHT_RAIL_PREVIEW_TAB_ID, $rightRailActiveTabId } from '@/store/layout'
import { $filePreviewTabs, setPreviewTarget } from '@/store/preview'

import { ChatPreviewRail } from './preview'

vi.mock('./preview-pane', () => ({
  PreviewPane: () => <div data-testid="preview-pane" />
}))

const liveTarget = {
  kind: 'url' as const,
  label: 'Live preview',
  source: 'http://localhost:5173',
  url: 'http://localhost:5173'
}

const alphaTarget = {
  kind: 'file' as const,
  label: 'alpha.txt',
  path: '/workspace/alpha.txt',
  source: '/workspace/alpha.txt',
  url: 'file:///workspace/alpha.txt'
}

const betaTarget = {
  kind: 'file' as const,
  label: 'beta.txt',
  path: '/workspace/beta.txt',
  source: '/workspace/beta.txt',
  url: 'file:///workspace/beta.txt'
}

describe('ChatPreviewRail tabs', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      callback(performance.now())
      return 1
    })
    vi.stubGlobal('cancelAnimationFrame', vi.fn())
    setPreviewTarget(liveTarget)
    $filePreviewTabs.set([
      { id: `file:${alphaTarget.url}`, target: alphaTarget },
      { id: `file:${betaTarget.url}`, target: betaTarget }
    ])
    $rightRailActiveTabId.set(RIGHT_RAIL_PREVIEW_TAB_ID)
  })

  afterEach(() => {
    cleanup()
    setPreviewTarget(null)
    $filePreviewTabs.set([])
    $rightRailActiveTabId.set(RIGHT_RAIL_PREVIEW_TAB_ID)
    vi.unstubAllGlobals()
  })

  it('supports roving keyboard selection for preview tabs', () => {
    render(<ChatPreviewRail />)

    const tabs = screen.getAllByRole('tab')
    expect(tabs.map(tab => tab.textContent)).toEqual(['Preview', 'alpha.txt', 'beta.txt'])

    fireEvent.keyDown(tabs[0], { key: 'ArrowRight' })
    expect(tabs[1]?.getAttribute('aria-selected')).toBe('true')
    expect(document.activeElement).toBe(tabs[1])

    fireEvent.keyDown(tabs[1], { key: 'End' })
    expect(tabs[2].getAttribute('aria-selected')).toBe('true')
    expect(document.activeElement).toBe(tabs[2])

    fireEvent.keyDown(tabs[2], { key: 'ArrowRight' })
    expect(tabs[0].getAttribute('aria-selected')).toBe('true')
    expect(document.activeElement).toBe(tabs[0])

    fireEvent.keyDown(tabs[0], { key: 'ArrowLeft' })
    expect(tabs[2].getAttribute('aria-selected')).toBe('true')
    expect(document.activeElement).toBe(tabs[2])

    fireEvent.keyDown(tabs[2], { key: 'Home' })
    expect(tabs[0].getAttribute('aria-selected')).toBe('true')
    expect(document.activeElement).toBe(tabs[0])
  })
})
