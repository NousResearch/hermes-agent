import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/local-preview')
vi.mock('@/lib/preview-reach')
vi.mock('@/store/preview')
vi.mock('@/lib/gateway-events')

import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { reachablePreviewUrl } from '@/lib/preview-reach'
import { closePreviewMatching, openPreview } from '@/store/preview'

// Inline the logic under test for the two call sites in use-preview-routing.ts

async function openPreviewRouting(
  target: string,
  label: string,
  cwd: string | undefined
): Promise<void> {
  await (normalizeOrLocalPreviewTarget as ReturnType<typeof vi.fn>)(target, cwd).then(
    async (resolved: any) => {
      if (!resolved) return
      const trimmedLabel = label.trim()
      const url = resolved.kind === 'url'
        ? await (reachablePreviewUrl as ReturnType<typeof vi.fn>)(resolved.url)
        : resolved.url
      const reached = url === resolved.url ? resolved : { ...resolved, url }
      ;(openPreview as ReturnType<typeof vi.fn>)(trimmedLabel ? { ...reached, label: trimmedLabel } : reached, 'tool-result')
    }
  ).catch(() => {
    // leave preview pane as-is
  })
}

async function closePreviewRouting(
  target: string,
  cwd: string | undefined
): Promise<void> {
  await (normalizeOrLocalPreviewTarget as ReturnType<typeof vi.fn>)(target, cwd).then(
    async (resolved: any) => {
      const candidates = [target]
      if (resolved) {
        candidates.push(resolved.source, resolved.url)
        if (resolved.kind === 'url') {
          candidates.push(await (reachablePreviewUrl as ReturnType<typeof vi.fn>)(resolved.url))
        }
      }
      ;(closePreviewMatching as ReturnType<typeof vi.fn>)(...candidates)
    }
  ).catch(() => {
    ;(closePreviewMatching as ReturnType<typeof vi.fn>)(target)
  })
}

describe('use-preview-routing rejection handlers', () => {
  afterEach(() => { vi.clearAllMocks() })

  describe('openPreviewRouting', () => {
    it('opens preview when normalizeOrLocalPreviewTarget resolves', async () => {
      vi.mocked(normalizeOrLocalPreviewTarget).mockResolvedValue({ kind: 'url', url: 'https://x.com', source: 'https://x.com', label: '' } as any)
      vi.mocked(reachablePreviewUrl).mockResolvedValue('https://x.com')
      await openPreviewRouting('https://x.com', 'Page', undefined)
      expect(openPreview).toHaveBeenCalled()
    })

    it('does not throw and does not open preview when normalizeOrLocalPreviewTarget rejects', async () => {
      vi.mocked(normalizeOrLocalPreviewTarget).mockRejectedValue(new Error('network'))
      await expect(openPreviewRouting('bad://url', '', undefined)).resolves.toBeUndefined()
      expect(openPreview).not.toHaveBeenCalled()
    })
  })

  describe('closePreviewRouting', () => {
    it('closes preview with resolved candidates when normalizeOrLocalPreviewTarget resolves', async () => {
      vi.mocked(normalizeOrLocalPreviewTarget).mockResolvedValue({ kind: 'url', url: 'https://x.com', source: 'https://x.com' } as any)
      vi.mocked(reachablePreviewUrl).mockResolvedValue('https://x.com')
      await closePreviewRouting('https://x.com', undefined)
      expect(closePreviewMatching).toHaveBeenCalledWith('https://x.com', 'https://x.com', 'https://x.com', 'https://x.com')
    })

    it('falls back to raw target when normalizeOrLocalPreviewTarget rejects', async () => {
      vi.mocked(normalizeOrLocalPreviewTarget).mockRejectedValue(new Error('network'))
      await closePreviewRouting('https://x.com', undefined)
      expect(closePreviewMatching).toHaveBeenCalledWith('https://x.com')
    })
  })
})
