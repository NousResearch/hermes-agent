import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/local-preview')
vi.mock('@/lib/preview-reach')
vi.mock('@/store/preview')
vi.mock('@/lib/gateway-events')

import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { reachablePreviewUrl } from '@/lib/preview-reach'
import { closePreviewMatching, openPreview } from '@/store/preview'

import { resolveAndClosePreview, resolveAndOpenPreview } from './use-preview-routing'

describe('use-preview-routing rejection handlers', () => {
  afterEach(() => { vi.clearAllMocks() })

  describe('resolveAndOpenPreview', () => {
    it('opens preview when normalizeOrLocalPreviewTarget resolves', async () => {
      vi.mocked(normalizeOrLocalPreviewTarget).mockResolvedValue({ kind: 'url', url: 'https://x.com', source: 'https://x.com', label: '' } as any)
      vi.mocked(reachablePreviewUrl).mockResolvedValue('https://x.com')
      await resolveAndOpenPreview('https://x.com', 'Page', undefined)
      expect(openPreview).toHaveBeenCalled()
    })

    it('does not throw and does not open preview when normalizeOrLocalPreviewTarget rejects', async () => {
      vi.mocked(normalizeOrLocalPreviewTarget).mockRejectedValue(new Error('network'))
      await expect(resolveAndOpenPreview('bad://url', '', undefined)).resolves.toBeUndefined()
      expect(openPreview).not.toHaveBeenCalled()
    })
  })

  describe('resolveAndClosePreview', () => {
    it('closes preview with resolved candidates when normalizeOrLocalPreviewTarget resolves', async () => {
      vi.mocked(normalizeOrLocalPreviewTarget).mockResolvedValue({ kind: 'url', url: 'https://x.com', source: 'https://x.com' } as any)
      vi.mocked(reachablePreviewUrl).mockResolvedValue('https://x.com')
      await resolveAndClosePreview('https://x.com', undefined)
      expect(closePreviewMatching).toHaveBeenCalledWith('https://x.com', 'https://x.com', 'https://x.com', 'https://x.com')
    })

    it('falls back to raw target when normalizeOrLocalPreviewTarget rejects', async () => {
      vi.mocked(normalizeOrLocalPreviewTarget).mockRejectedValue(new Error('network'))
      await resolveAndClosePreview('https://x.com', undefined)
      expect(closePreviewMatching).toHaveBeenCalledWith('https://x.com')
    })
  })
})
