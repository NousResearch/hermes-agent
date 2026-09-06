import { describe, expect, it, vi } from 'vitest'

import type { PreviewUblockState } from './preview-ublock'
import type { PreviewUblockRuntime } from './preview-ublock-runtime'
import { runPreviewUblockSmoke } from './preview-ublock-smoke'

function state(overrides: Partial<PreviewUblockState> = {}): PreviewUblockState {
  return {
    available: false,
    dashboardUrl: null,
    enabled: false,
    extensionId: null,
    operation: {
      failure: null,
      operationId: null,
      phase: 'idle',
      receivedBytes: 0,
      totalBytes: null
    },
    popupUrl: null,
    rulesetsReady: false,
    version: null,
    ...overrides
  }
}

describe('preview uBlock smoke orchestration', () => {
  it('disposes the runtime when readiness validation fails before an owner window exists', async () => {
    const runtime = {
      dispose: vi.fn().mockResolvedValue(undefined),
      getBlockedRequestCount: vi.fn(),
      getRequestBlockerBlockedRequestCount: vi.fn(),
      getState: vi.fn(),
      initialize: vi.fn(),
      loadRules: vi.fn(),
      openPopup: vi.fn(),
      recordBlockedRequest: vi.fn(),
      registerGuest: vi.fn(),
      setEnabled: vi.fn().mockResolvedValue(state()),
      unregisterGuest: vi.fn()
    } as unknown as PreviewUblockRuntime
    const createOwnerWindow = vi.fn()

    await expect(
      runPreviewUblockSmoke(runtime, {
        createOwnerWindow,
        findPopupWindow: () => null,
        getGuest: () => null,
        previewSession: {}
      })
    ).rejects.toThrow('did not become available')
    expect(createOwnerWindow).not.toHaveBeenCalled()
    expect(runtime.setEnabled).toHaveBeenLastCalledWith(false)
    expect(runtime.dispose).toHaveBeenCalledOnce()
  })
})
