import { afterEach, describe, expect, it, vi } from 'vitest'

import { $previewUblock, loadPreviewUblock, setPreviewUblockEnabled } from './preview-ublock'

const originalBridge = window.hermesDesktop

function state(enabled: boolean) {
  return {
    enabled,
    available: enabled,
    dashboardUrl: enabled ? 'chrome-extension://ublock/dashboard.html' : null,
    extensionId: enabled ? 'ublock' : null,
    operation: {
      failure: null,
      operationId: null,
      phase: enabled ? ('ready' as const) : ('idle' as const),
      receivedBytes: 0,
      totalBytes: null
    },
    popupUrl: enabled ? 'chrome-extension://ublock/popup.html' : null,
    rulesetsReady: enabled,
    version: enabled ? '2026.825.1619' : null
  }
}

afterEach(() => {
  window.hermesDesktop = originalBridge
  $previewUblock.set({
    enabled: false,
    available: false,
    dashboardUrl: null,
    extensionId: null,
    operation: { failure: null, operationId: null, phase: 'idle', receivedBytes: 0, totalBytes: null },
    popupUrl: null,
    rulesetsReady: false,
    version: null
  })
  vi.restoreAllMocks()
})

describe('preview uBlock renderer state', () => {
  it('loads and mirrors the authoritative enabled state', async () => {
    window.hermesDesktop = {
      ...originalBridge,
      previewUblock: {
        getState: vi.fn().mockResolvedValue(state(true)),
        onState: vi.fn().mockReturnValue(() => undefined),
        setEnabled: vi.fn()
      }
    }

    await loadPreviewUblock()

    expect($previewUblock.get()).toEqual(state(true))
  })

  it('updates the atom from the main-process setter result', async () => {
    const setEnabled = vi.fn().mockResolvedValue(state(false))
    window.hermesDesktop = {
      ...originalBridge,
      previewUblock: {
        getState: vi.fn(),
        onState: vi.fn().mockReturnValue(() => undefined),
        setEnabled
      }
    }

    await expect(setPreviewUblockEnabled(false)).resolves.toEqual(state(false))
    expect(setEnabled).toHaveBeenCalledWith(false)
    expect($previewUblock.get()).toEqual(state(false))
  })

  it('rejects without changing state when the main-process setter fails', async () => {
    const setEnabled = vi.fn().mockRejectedValue(new Error('IPC unavailable'))
    window.hermesDesktop = {
      ...originalBridge,
      previewUblock: {
        getState: vi.fn(),
        onState: vi.fn().mockReturnValue(() => undefined),
        setEnabled
      }
    }
    $previewUblock.set(state(true))

    await expect(setPreviewUblockEnabled(false)).rejects.toThrow('IPC unavailable')
    expect($previewUblock.get()).toEqual(state(true))
  })
})
