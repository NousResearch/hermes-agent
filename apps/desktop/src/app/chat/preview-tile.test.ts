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

import { browserTabLabel, watchPreviewTiles } from './preview-tile'

beforeAll(() => {
  watchPreviewTiles()
})

afterEach(() => {
  closeRightRail()
})

// By prefix, not by a literal id: a Browser tab's id is minted per tab now
// that there can be more than one of them.
function browserPane() {
  return registry.getArea('panes').find(entry => entry.id.startsWith('preview-tile:url:'))
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

describe('browserTabLabel', () => {
  const target = { kind: 'url', label: 'Browser', source: 'about:blank', url: 'about:blank' } as const

  it('names the tab after the page', () => {
    expect(browserTabLabel(target, { title: 'Hacker News', url: 'https://news.ycombinator.com/' })).toBe('Hacker News')
  })

  // Chromium hands back the address as the title when the page never set one,
  // which is a worse tab label than the host it came from.
  it('falls back to the host when the page has no title of its own', () => {
    expect(browserTabLabel(target, { title: '', url: 'https://www.example.com/a/b' })).toBe('example.com')
    expect(browserTabLabel(target, { title: 'https://example.com/a', url: 'https://example.com/a' })).toBe(
      'example.com'
    )
  })

  it('falls back to the surface when there is no page and no host', () => {
    expect(browserTabLabel(target)).toBe('Browser')
    expect(browserTabLabel(target, { title: '', url: 'about:blank' })).toBe('Browser')
  })

  // A tab restored from storage has reported nothing yet, so its target is all
  // there is to name it by.
  it('names an unreported tab from its target', () => {
    expect(browserTabLabel({ ...target, url: 'https://github.com/nous' })).toBe('github.com')
  })
})

type DockData = { dock?: { pane?: string; pos?: string } } | undefined

function dockOf(paneId: string) {
  return (registry.getArea('panes').find(entry => entry.id === paneId)?.data as DockData)?.dock
}

const fileTarget = (path: string) =>
  ({ kind: 'file', label: path.split('/').at(-1) ?? path, path, source: path, url: path }) as const

describe('preview tiles stack, not split (#93610)', () => {
  it('docks the first preview right and stacks the second as a center tab in the same zone', () => {
    openPreview(fileTarget('/tmp/a.ts'), 'file-browser')

    const first = dockOf('preview-tile:file:/tmp/a.ts')

    expect(first?.pos).toBe('right')

    openPreview(fileTarget('/tmp/b.ts'), 'file-browser')

    const second = dockOf('preview-tile:file:/tmp/b.ts')

    expect(second?.pos).toBe('center')
    expect(second?.pane).toBe('preview-tile:file:/tmp/a.ts')

    // The first pane's registration is untouched — one preview zone, two tabs.
    expect(dockOf('preview-tile:file:/tmp/a.ts')?.pos).toBe('right')
  })

  it('stacks an artifact opened after a file into the same preview zone', () => {
    openPreview(fileTarget('/tmp/a.ts'), 'file-browser')
    openPreview({ kind: 'artifact', label: 'Chart', source: 'artifact-1', url: 'artifact-1' }, 'explicit-link')

    const artifact = dockOf('preview-tile:artifact:artifact-1')

    expect(artifact?.pos).toBe('center')
    expect(artifact?.pane).toBe('preview-tile:file:/tmp/a.ts')
  })

  it('lets a lone preview open its own right-docked zone again after all tabs closed', () => {
    openPreview(fileTarget('/tmp/a.ts'), 'file-browser')
    openPreview(fileTarget('/tmp/b.ts'), 'file-browser')
    closeRightRail()

    openPreview(fileTarget('/tmp/c.ts'), 'file-browser')

    expect(dockOf('preview-tile:file:/tmp/c.ts')?.pos).toBe('right')
  })
})
