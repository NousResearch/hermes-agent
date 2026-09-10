import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { group } from '@/components/pane-shell/tree/model'
import { TreeGroup } from '@/components/pane-shell/tree/renderer/tree-group'
import { declareDefaultTree } from '@/components/pane-shell/tree/store'
import { $previewTabs, closeRightRail, openPreview, type PreviewTarget } from '@/store/preview'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { watchPreviewTiles } from './preview-tile'

vi.mock('./right-rail/preview', () => ({ PreviewTilePane: () => null }))
vi.mock('./right-rail/preview-console-store', () => ({ forgetPreviewConsole: () => undefined }))

const writeText = vi.fn(async () => undefined)

beforeAll(() => {
  stubMenuDomApis()
  stubResizeObserver()
  vi.stubGlobal('CSS', { ...globalThis.CSS, escape: (value: string) => value })
  vi.stubGlobal('hermesDesktop', { writeClipboard: writeText })
  watchPreviewTiles()
})

afterAll(() => vi.unstubAllGlobals())

afterEach(() => {
  cleanup()
  closeRightRail()
  writeText.mockClear()
})

function mountTabs(targets: PreviewTarget[]) {
  for (const target of targets) {
    openPreview(target)
  }

  const panes = $previewTabs.get().map(tab => `preview-tile:${tab.id}`)
  const node = group(panes, { active: panes.at(-1), id: 'preview-zone' })
  declareDefaultTree(node)
  const { container } = render(<TreeGroup node={node} />)

  const tab = container.querySelector<HTMLElement>(`[data-tree-tab="${panes[0]}"]`)!
  fireEvent.pointerDown(tab, { button: 2, pointerType: 'mouse' })
  fireEvent.contextMenu(tab, { button: 2 })
}

describe('preview file tab context menu', () => {
  it('copies the full path of the right-clicked inactive file, including spaces and Unicode', async () => {
    const path = '/workspace/Project Notes/überblick.md'
    mountTabs([
      { kind: 'file', label: 'überblick.md', path, source: path, url: path },
      { kind: 'file', label: 'active.ts', path: '/workspace/active.ts', source: '', url: '/workspace/active.ts' }
    ])

    fireEvent.click(await screen.findByRole('menuitem', { name: 'Copy path' }))

    await waitFor(() => expect(writeText).toHaveBeenCalledExactlyOnceWith(path))
  })

  it.each(['url', 'artifact'] as const)('does not offer a filesystem path for a %s tab', async kind => {
    mountTabs([{ kind, label: 'Preview', source: 'https://example.com', url: 'https://example.com' }])

    expect(await screen.findByRole('menuitem', { name: /^Close$/ })).toBeTruthy()
    expect(screen.queryByRole('menuitem', { name: 'Copy path' })).toBeNull()
    expect(writeText).not.toHaveBeenCalled()
  })
})
