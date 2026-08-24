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
