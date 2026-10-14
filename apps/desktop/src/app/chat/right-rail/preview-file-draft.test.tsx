import { EditorView } from '@codemirror/view'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $previewTabs, closeRightRail, closeRightRailTab, discardDraftAndCloseTab, openPreview } from '@/store/preview'
import { $dirtyPreviewUrls, $pendingPreviewDiscards, prunePreviewDrafts } from '@/store/preview-edit'

import { LocalFilePreview } from './preview-file'

vi.mock('@/components/chat/shiki-highlighter', () => ({
  LazyShiki: ({ code }: { code: string }) => <pre>{code}</pre>
}))

const target = {
  kind: 'file' as const,
  label: 'note.txt',
  path: '/workspace/note.txt',
  previewKind: 'text' as const,
  source: '/workspace/note.txt',
  url: 'file:///workspace/note.txt'
}

let disk = 'on disk'
const rangeDescriptors = Object.getOwnPropertyDescriptors(Range.prototype)

beforeEach(() => {
  disk = 'on disk'
  vi.stubGlobal('hermesDesktop', {
    readFileText: async () => ({ binary: false, path: target.path, text: disk, truncated: false }),
    writeTextFile: async (_file: string, content: string) => {
      disk = content

      return { path: target.path }
    }
  })
  Object.defineProperties(Range.prototype, {
    getClientRects: { configurable: true, value: () => [] },
    getBoundingClientRect: { configurable: true, value: () => new DOMRect() }
  })
})

afterEach(() => {
  cleanup()
  closeRightRail()
  $pendingPreviewDiscards.set([])
  prunePreviewDrafts(new Set())
  vi.unstubAllGlobals()

  for (const key of ['getClientRects', 'getBoundingClientRect']) {
    if (rangeDescriptors[key]) {
      Object.defineProperty(Range.prototype, key, rangeDescriptors[key])
    } else {
      Reflect.deleteProperty(Range.prototype, key)
    }
  }
})

async function typeDraft(container: HTMLElement, text: string) {
  fireEvent.click(await screen.findByRole('button', { name: 'Edit' }))
  const editor = EditorView.findFromDOM(container.querySelector('.cm-content')!)!
  act(() => editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: text } }))
}

const editorText = (container: HTMLElement) =>
  EditorView.findFromDOM(container.querySelector('.cm-content')!)!.state.doc.toString()

it('keeps an unsaved draft across a disk-change reload and an unmount of the preview body', async () => {
  const first = render(<LocalFilePreview reloadKey={0} target={target} />)
  await typeDraft(first.container, 'my unsaved edit')

  // The file watcher fires (an agent wrote the file): the pane bumps reloadKey.
  disk = 'agent rewrote it'
  await act(async () => first.rerender(<LocalFilePreview reloadKey={1} target={target} />))
  expect(editorText(first.container)).toBe('my unsaved edit')

  // Its session leaves the screen (or the zone evicts the tab): body unmounts.
  first.unmount()
  const second = render(<LocalFilePreview reloadKey={1} target={target} />)
  expect(editorText(second.container)).toBe('my unsaved edit')

  // The restored draft still knows its baseline: saving over the agent's
  // write raises the conflict instead of clobbering it.
  fireEvent.click(screen.getByRole('button', { name: /Save/ }))
  await screen.findByRole('button', { name: /Overwrite/ })
  expect(disk).toBe('agent rewrote it')
})

it('holds a tab with an unsaved draft open until the discard is confirmed', async () => {
  openPreview(target)
  const tabId = $previewTabs.get()[0].id
  const view = render(<LocalFilePreview reloadKey={0} target={target} />)
  await typeDraft(view.container, 'my unsaved edit')
  view.unmount()

  closeRightRailTab(tabId)
  expect($previewTabs.get().map(tab => tab.id)).toEqual([tabId])
  expect($pendingPreviewDiscards.get()).toEqual([tabId])

  discardDraftAndCloseTab(tabId)
  expect($previewTabs.get()).toEqual([])
  expect($dirtyPreviewUrls.get()).toEqual({})
})
