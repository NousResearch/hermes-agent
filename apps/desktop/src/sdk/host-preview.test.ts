import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as LocalPreview from '@/lib/local-preview'
import type * as PreviewStore from '@/store/preview'

const openPreview = vi.hoisted(() => vi.fn())
const normalizeOrLocalPreviewTarget = vi.hoisted(() => vi.fn())

vi.mock('@/store/preview', async importOriginal => ({
  ...(await importOriginal<typeof PreviewStore>()),
  openPreview
}))

vi.mock('@/lib/local-preview', async importOriginal => ({
  ...(await importOriginal<typeof LocalPreview>()),
  normalizeOrLocalPreviewTarget
}))

const { host } = await import('./index')

const file = (extra: object) => ({ kind: 'file', label: 'f', source: '/w/f', url: 'file:///w/f', ...extra })

describe('host.preview', () => {
  beforeEach(() => {
    openPreview.mockClear()
    normalizeOrLocalPreviewTarget.mockReset()
  })

  // A one-letter scheme is a Windows drive, not a URL.
  it.each(['/w/plan.md', 'C:\\w\\plan.md', 'file:///w/plan.md'])('opens %s in the preview rail', async path => {
    const target = file({ previewKind: 'text' })
    normalizeOrLocalPreviewTarget.mockResolvedValue(target)

    await expect(host.preview(path)).resolves.toBe(true)
    expect(openPreview).toHaveBeenCalledWith(target)
  })

  // `false` is the caller's cue to fall back to a download.
  it.each([
    ['nothing resolves', null],
    ['the target is not a file', { ...file({}), kind: 'url' }],
    ['the file is binary', file({ previewKind: 'binary' })],
    ['a backend read found binary content', file({ binary: true, previewKind: 'text' })]
  ])('resolves false without opening a tab when %s', async (_, target) => {
    normalizeOrLocalPreviewTarget.mockResolvedValue(target)

    await expect(host.preview('/w/f')).resolves.toBe(false)
    expect(openPreview).not.toHaveBeenCalled()
  })

  // Plugins pass backend-supplied strings; none may become a browser tab.
  it.each(['https://example.test/plan.md', 'data:text/html,x'])('refuses %s before resolving it', async url => {
    await expect(host.preview(url)).resolves.toBe(false)
    expect(normalizeOrLocalPreviewTarget).not.toHaveBeenCalled()
  })
})
