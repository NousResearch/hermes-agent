import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $rightRailActiveTabId } from '@/store/layout'
import {
  $filePreviewTabs,
  $previewSurfaceLayouts,
  clearSessionPreviewRegistry,
  detachRightRailTab,
  minimizeRightRailTab,
  type PreviewTarget,
  setCurrentSessionPreviewTarget,
  snapRightRailTab
} from '@/store/preview'
import { $activeSessionId } from '@/store/session'

import { ChatPreviewRail } from './preview'

vi.mock('./preview-pane', () => ({
  PreviewPane: ({ target }: { target: PreviewTarget }) => <div data-testid="preview-pane">{target.label}</div>
}))

const ORIGINAL_INNER_HEIGHT = window.innerHeight
const ORIGINAL_INNER_WIDTH = window.innerWidth

function fileTarget(path: string): PreviewTarget {
  return {
    kind: 'file',
    label: path.split('/').at(-1)!,
    path,
    previewKind: 'text',
    source: path,
    url: `file://${path}`
  }
}

describe('ChatPreviewRail workspace surfaces', () => {
  beforeEach(() => {
    window.localStorage.clear()
    clearSessionPreviewRegistry()
    $activeSessionId.set('session-1')
  })

  afterEach(() => {
    cleanup()
    clearSessionPreviewRegistry()
    $activeSessionId.set(null)
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: ORIGINAL_INNER_HEIGHT, writable: true })
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: ORIGINAL_INNER_WIDTH, writable: true })
    window.localStorage.clear()
    vi.restoreAllMocks()
  })

  it('switches the mounted pane across all open tabs', () => {
    setCurrentSessionPreviewTarget(fileTarget('/work/one.txt'), 'manual')
    setCurrentSessionPreviewTarget(fileTarget('/work/two.txt'), 'manual')
    render(<ChatPreviewRail />)

    expect(screen.getByTestId('preview-pane').textContent).toBe('two.txt')
    fireEvent.click(screen.getByRole('tab', { name: 'one.txt' }))
    expect(screen.getByTestId('preview-pane').textContent).toBe('one.txt')
  })

  it('does not mount a minimized pane and restores it from the taskbar', () => {
    setCurrentSessionPreviewTarget(fileTarget('/work/one.txt'), 'manual')
    const tabId = $filePreviewTabs.get()[0]!.id
    detachRightRailTab(tabId)
    minimizeRightRailTab(tabId)
    render(<ChatPreviewRail />)

    expect(screen.queryByTestId('preview-pane')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Restore one.txt' }))

    expect($previewSurfaceLayouts.get()[tabId]?.placement).toBe('floating')
    expect($rightRailActiveTabId.get()).toBe(tabId)
    expect(screen.getByTestId('preview-pane').textContent).toBe('one.txt')
  })

  it('maximizes a floating surface and preserves a restore placement', () => {
    setCurrentSessionPreviewTarget(fileTarget('/work/one.txt'), 'manual')
    const tabId = $filePreviewTabs.get()[0]!.id
    detachRightRailTab(tabId)
    render(<ChatPreviewRail />)

    fireEvent.click(screen.getByRole('button', { name: 'Maximize one.txt' }))

    expect($previewSurfaceLayouts.get()[tabId]).toMatchObject({
      placement: 'maximized',
      restore: { placement: 'floating' }
    })
  })

  it('keeps snapped surfaces above the minimized taskbar', () => {
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 1280, writable: true })
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: 633, writable: true })
    setCurrentSessionPreviewTarget(fileTarget('/work/snapped.txt'), 'manual')
    setCurrentSessionPreviewTarget(fileTarget('/work/minimized.txt'), 'manual')
    const [snapped, minimized] = $filePreviewTabs.get()
    snapRightRailTab(snapped!.id, 'right-half')
    minimizeRightRailTab(minimized!.id)
    render(<ChatPreviewRail />)

    const surface = screen.getByTestId('floating-preview-surface')
    const surfaceBottom = Number.parseFloat(surface.style.top) + Number.parseFloat(surface.style.height)

    expect(surfaceBottom).toBeLessThanOrEqual(window.innerHeight - 56)
  })

  it('recomputes snapped geometry when the viewport changes', () => {
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 1200, writable: true })
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: 900, writable: true })
    setCurrentSessionPreviewTarget(fileTarget('/work/one.txt'), 'manual')
    const tabId = $filePreviewTabs.get()[0]!.id
    snapRightRailTab(tabId, 'left-half')
    render(<ChatPreviewRail />)

    const surface = screen.getByTestId('floating-preview-surface')
    expect(surface.style.width).toBe('600px')

    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 800, writable: true })
    fireEvent(window, new Event('resize'))

    expect(surface.style.width).toBe('400px')
  })

  it('opens the keyboard snap picker and chooses a corner slot', () => {
    setCurrentSessionPreviewTarget(fileTarget('/work/one.txt'), 'manual')
    const tabId = $filePreviewTabs.get()[0]!.id
    detachRightRailTab(tabId)
    render(<ChatPreviewRail />)

    fireEvent.click(screen.getByRole('button', { name: 'Snap layouts for one.txt' }))
    fireEvent.click(screen.getByRole('button', { name: 'Snap one.txt to top left quarter' }))

    expect($previewSurfaceLayouts.get()[tabId]?.placement).toBe('top-left-quarter')
    expect(screen.queryByRole('button', { name: 'Snap one.txt to top left quarter' })).toBeNull()
  })
})
