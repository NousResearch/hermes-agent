import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $rightRailActiveTabId, RIGHT_RAIL_PREVIEW_TAB_ID, selectRightRailTab } from '@/store/layout'
import { $paneStates } from '@/store/panes'
import { $filePreviewTabs, $previewTabs, previewTabId, type PreviewTarget } from '@/store/preview'

import { ChatPreviewRail } from './preview'

function target(label: string, url: string): PreviewTarget {
  return {
    kind: 'url',
    label,
    source: url,
    url
  }
}

function fileTarget(label: string, path: string): PreviewTarget {
  return {
    kind: 'file',
    label,
    path,
    previewKind: 'text',
    renderMode: 'source',
    source: path,
    url: `file://${path}`
  }
}

function renderRail() {
  return render(
    <I18nProvider configClient={null}>
      <ChatPreviewRail setTitlebarToolGroup={() => undefined} />
    </I18nProvider>
  )
}

describe('ChatPreviewRail tabs', () => {
  beforeEach(() => {
    const first = target('Local app', 'http://localhost:5174')
    const second = target('Docs', 'https://example.com/docs')
    const file = fileTarget('preview-pane.tsx', '/work/preview-pane.tsx')

    $previewTabs.set([
      { id: previewTabId(first), target: first },
      { id: previewTabId(second), target: second }
    ])
    $filePreviewTabs.set([{ id: `file:${file.url}`, target: file }])
    selectRightRailTab(previewTabId(first))
  })

  afterEach(() => {
    cleanup()
    $previewTabs.set([])
    $filePreviewTabs.set([])
    $paneStates.set({})
    selectRightRailTab(RIGHT_RAIL_PREVIEW_TAB_ID)
  })

  it('switches between live preview tabs inside the docked rail', () => {
    renderRail()

    expect(screen.getByRole('tab', { name: 'Local app' }).getAttribute('aria-selected')).toBe('true')

    fireEvent.click(screen.getByRole('tab', { name: 'Docs' }))

    expect($rightRailActiveTabId.get()).toBe('preview:https://example.com/docs')
    expect(screen.getByRole('tab', { name: 'Docs' }).getAttribute('aria-selected')).toBe('true')
    expect((screen.getByLabelText('Preview URL') as HTMLInputElement).value).toBe('https://example.com/docs')
  })

  it('switches from a file tab back to a live preview tab', () => {
    renderRail()

    fireEvent.click(screen.getByRole('tab', { name: 'preview-pane.tsx' }))
    expect(screen.getByRole('tab', { name: 'preview-pane.tsx' }).getAttribute('aria-selected')).toBe('true')

    fireEvent.click(screen.getByRole('tab', { name: 'Docs' }))

    expect($rightRailActiveTabId.get()).toBe('preview:https://example.com/docs')
    expect(screen.getByRole('tab', { name: 'Docs' }).getAttribute('aria-selected')).toBe('true')
    expect((screen.getByLabelText('Preview URL') as HTMLInputElement).value).toBe('https://example.com/docs')
  })
})
