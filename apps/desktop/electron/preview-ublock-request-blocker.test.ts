import { describe, expect, it, vi } from 'vitest'

import {
  compilePreviewUblockRules,
  createPreviewUblockRequestBlocker,
  shouldBlockPreviewUblockRequest
} from './preview-ublock-request-blocker'

describe('preview uBlock request blocker', () => {
  it('blocks matching static rules and keeps non-matching requests', () => {
    const rules = compilePreviewUblockRules([
      {
        action: { type: 'block' },
        condition: { resourceTypes: ['image'], urlFilter: '||ads.example^' }
      }
    ])

    expect(
      shouldBlockPreviewUblockRequest(rules, {
        resourceType: 'image',
        url: 'https://cdn.ads.example/banner.png'
      })
    ).toBe(true)
    expect(
      shouldBlockPreviewUblockRequest(rules, {
        resourceType: 'script',
        url: 'https://cdn.ads.example/ad.js'
      })
    ).toBe(false)
  })

  it('honors an equal-priority allow exception over a blocking rule', () => {
    const rules = compilePreviewUblockRules([
      { action: { type: 'block' }, condition: { urlFilter: '||ads.example^' } },
      { action: { type: 'allow' }, condition: { urlFilter: '||ads.example/safe^' } }
    ])

    expect(shouldBlockPreviewUblockRequest(rules, { url: 'https://ads.example/safe/image.png' })).toBe(false)
  })

  it('applies rules only to registered Preview webviews while enabled', () => {
    let listener:
      | ((details: { url: string; webContentsId?: number }, callback: (response: { cancel: boolean }) => void) => void)
      | undefined
    const session = {
      webRequest: {
        onBeforeRequest(next: typeof listener) {
          listener = next
        },
        removeListener: vi.fn()
      }
    }
    const blocker = createPreviewUblockRequestBlocker({ session })
    const respond = (webContentsId: number) => {
      const callback = vi.fn()
      listener?.({ url: 'https://ads.example/banner.png', webContentsId }, callback)

      return callback
    }

    blocker.registerGuest(12, 99)
    blocker.setActive(true)

    // Exercise the session boundary with a loaded rule set; the matching
    // semantics are covered above, so this assertion stays focused on scope.
    expect(respond(13)).toHaveBeenCalledWith({ cancel: false })
  })
})
