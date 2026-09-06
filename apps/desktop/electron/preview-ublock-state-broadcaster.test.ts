import { describe, expect, it, vi } from 'vitest'

import { broadcastPreviewUblockStateToWindows } from './preview-ublock-state-broadcaster'

const state = {
  available: false,
  dashboardUrl: null,
  enabled: false,
  extensionId: null,
  operation: {
    failure: null,
    operationId: null,
    phase: 'idle' as const,
    receivedBytes: 0,
    totalBytes: null
  },
  popupUrl: null,
  rulesetsReady: false,
  version: null
}

describe('preview uBlock state broadcasting', () => {
  it('skips destroyed contents and continues after an individual send failure', () => {
    const healthySend = vi.fn()
    const log = vi.fn()

    broadcastPreviewUblockStateToWindows(
      [
        { isDestroyed: () => false, webContents: { isDestroyed: () => true, send: vi.fn() } },
        {
          isDestroyed: () => false,
          webContents: {
            isDestroyed: () => false,
            send: () => {
              throw new Error('closed')
            }
          }
        },
        { isDestroyed: () => false, webContents: { isDestroyed: () => false, send: healthySend } }
      ],
      state,
      log
    )

    expect(healthySend).toHaveBeenCalledWith('hermes:preview-ublock:state', state)
    expect(log).toHaveBeenCalledOnce()
  })
})
