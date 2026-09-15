import { describe, expect, it, vi } from 'vitest'

import { onComposerAttachPathsRequest, onComposerInsertRequest } from './focus'
import { routeClipboardToComposer } from './paste-to-focus'

// `text` is the DOM shortcut flavor of `text/plain`; routeClipboardToComposer
// reads both, so the mock serves them from one value.
const clipboardWith = (flavors: Record<string, string>) => {
  const plain = flavors['text/plain'] ?? ''

  return {
    getData: (type: string) => (type === 'text' ? plain : (flavors[type] ?? '')),
    items: []
  } as unknown as DataTransfer
}

// The focus bus defers dispatch to a macrotask — tests wait one before asserting.
const nextTick = () => new Promise<void>(resolve => window.setTimeout(resolve, 0))

describe('routeClipboardToComposer with an OS file copy', () => {
  it('routes pasted file:// URLs to the attach-paths bus and swallows the URL text', async () => {
    const attachPaths = vi.fn()

    const off = onComposerAttachPathsRequest(({ paths }) => attachPaths(paths))

    try {
      const handled = routeClipboardToComposer(clipboardWith({ 'text/uri-list': 'file:///home/me/report.pdf' }))

      expect(handled).toBe(true)

      await nextTick()

      expect(attachPaths).toHaveBeenCalledWith(['/home/me/report.pdf'])
    } finally {
      off()
    }
  })

  it('inserts only the prose beside the file — the file:// line is not re-inserted', async () => {
    const attachPaths = vi.fn()
    const inserts: string[] = []

    const offPaths = onComposerAttachPathsRequest(({ paths }) => attachPaths(paths))
    const offInserts = onComposerInsertRequest(({ text }) => inserts.push(text))

    try {
      const handled = routeClipboardToComposer(
        clipboardWith({
          'text/plain': 'please review\nfile:///home/me/notes.txt',
          'text/uri-list': 'file:///home/me/notes.txt'
        })
      )

      expect(handled).toBe(true)

      await nextTick()

      expect(attachPaths).toHaveBeenCalledWith(['/home/me/notes.txt'])
      expect(inserts).toEqual(['please review'])
    } finally {
      offPaths()
      offInserts()
    }
  })

  it('pulls focus for a pure file copy (no prose to insert)', async () => {
    const attachPaths = vi.fn()
    const inserts: string[] = []

    const offPaths = onComposerAttachPathsRequest(({ paths }) => attachPaths(paths))
    const offInserts = onComposerInsertRequest(({ text }) => inserts.push(text))

    try {
      const handled = routeClipboardToComposer(clipboardWith({ 'text/uri-list': 'file:///home/me/b.pdf' }))

      expect(handled).toBe(true)

      await nextTick()

      expect(attachPaths).toHaveBeenCalledWith(['/home/me/b.pdf'])
      expect(inserts).toEqual([])
    } finally {
      offPaths()
      offInserts()
    }
  })
})
