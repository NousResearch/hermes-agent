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
  setCurrentSessionPreviewTarget
} from '@/store/preview'
import { $activeSessionId } from '@/store/session'

import { ChatPreviewRail } from './preview'

vi.mock('./preview-pane', () => ({
  PreviewPane: ({ target }: { target: PreviewTarget }) => <div data-testid="preview-pane">{target.label}</div>
}))

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
