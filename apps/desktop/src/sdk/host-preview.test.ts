import { beforeEach, describe, expect, it, vi } from 'vitest'

import { localPreviewTarget } from '@/lib/local-preview'
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

describe('host.preview', () => {
  beforeEach(() => {
    openPreview.mockClear()
    normalizeOrLocalPreviewTarget.mockReset()
  })

  it('opens a resolvable target in the preview rail and reports success', async () => {
    const target = { kind: 'file', label: 'plan.md', previewKind: 'text', source: '/w/plan.md', url: 'file:///w/plan.md' }
    normalizeOrLocalPreviewTarget.mockResolvedValue(target)

    await expect(host.preview('/w/plan.md')).resolves.toBe(true)
    expect(openPreview).toHaveBeenCalledWith(target)
  })

  it('reports failure instead of opening a broken tab when the target will not resolve', async () => {
    normalizeOrLocalPreviewTarget.mockResolvedValue(null)

    // The caller needs `false` to fall back (Kanban saves a copy instead).
    await expect(host.preview('/w/missing.md')).resolves.toBe(false)
    expect(openPreview).not.toHaveBeenCalled()
  })

  it('refuses a binary target rather than rendering garbage', async () => {
    normalizeOrLocalPreviewTarget.mockResolvedValue({
      kind: 'file',
      label: 'bundle.zip',
      previewKind: 'binary',
      source: '/w/bundle.zip',
      url: 'file:///w/bundle.zip'
    })

    await expect(host.preview('/w/bundle.zip')).resolves.toBe(false)
    expect(openPreview).not.toHaveBeenCalled()
  })

  it('opens only files, never a URL handed over by a backend', async () => {
    // Backend-supplied strings (attachment paths) must not become a browser tab.
    await expect(host.preview('https://example.test/plan.md')).resolves.toBe(false)
    expect(normalizeOrLocalPreviewTarget).not.toHaveBeenCalled()
    expect(openPreview).not.toHaveBeenCalled()
  })
})

describe('localPreviewTarget classification for agent artifacts', () => {
  it('treats a markdown attachment as readable markdown text, not a download', () => {
    expect(localPreviewTarget('/h/.hermes/kanban/boards/b/attachments/t_1/plan.md')).toMatchObject({
      kind: 'file',
      label: 'plan.md',
      language: 'markdown',
      previewKind: 'text'
    })
  })
})
