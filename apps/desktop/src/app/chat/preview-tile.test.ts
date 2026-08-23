import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

vi.mock('./right-rail/preview', () => ({
  PreviewTilePane: () => null
}))

vi.mock('./right-rail/preview-console-store', () => ({
  forgetPreviewConsole: () => undefined
}))

import { contributesToWorkspace } from '@/components/pane-shell/workspace-scope'
import { registry } from '@/contrib/registry'
import { closeRightRail, openPreview } from '@/store/preview'

import { watchPreviewTiles } from './preview-tile'

beforeAll(() => {
  watchPreviewTiles()
})

afterEach(() => {
  closeRightRail()
})

function browserPane() {
  return registry.getArea('panes').find(entry => entry.id === 'preview-tile:url:browser')
}

describe('preview tiles in Bot Mode', () => {
  it('registers the in-app Browser as a global pane so Bot Mode can show it', () => {
    openPreview(
      { kind: 'url', label: 'example.com', source: 'https://example.com', url: 'https://example.com' },
      'explicit-link'
    )

    const pane = browserPane()

    expect(pane).toBeTruthy()
    expect(pane?.workspaceMode).toBeUndefined()
    expect(contributesToWorkspace(pane, 'sessions')).toBe(true)
    expect(contributesToWorkspace(pane, 'bots', 'bot:connection-a::default')).toBe(true)
  })
})
