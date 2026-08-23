import { afterEach, describe, expect, it, vi } from 'vitest'

import { $gateway } from '@/store/gateway'
import {
  $sendDiagnostics,
  confirmSendDiagnostics,
  dismissSendDiagnostics,
  requestSendDiagnostics
} from '@/store/send-diagnostics'

function stubGateway(
  request: (method: string, params?: Record<string, unknown>, timeout?: number) => Promise<unknown>
) {
  const original = $gateway.get()

  $gateway.set({ request } as never)

  return () => $gateway.set(original)
}

function stubDesktopLogs(lines: null | string[]) {
  const original = window.hermesDesktop

  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: lines ? { getRecentLogs: async () => ({ lines, path: '/tmp/desktop.log' }) } : undefined
  })

  return () => Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: original })
}

describe('send-diagnostics store', () => {
  afterEach(() => {
    $sendDiagnostics.set(null)
    vi.restoreAllMocks()
  })

  it('opens in consent phase without any network I/O', () => {
    const request = vi.fn()
    const restore = stubGateway(request)

    try {
      requestSendDiagnostics('layer: provider')

      expect($sendDiagnostics.get()).toEqual({ errorContext: 'layer: provider', phase: 'consent' })
      expect(request).not.toHaveBeenCalled()
    } finally {
      restore()
    }
  })

  it('uploads on confirm, attaching error context and the local desktop log', async () => {
    const request = vi.fn().mockResolvedValue({
      ok: true,
      view_url: 'https://nas.example/view/x1',
      upload_id: 'x1',
      expires_at: '2026-09-05T00:00:00Z'
    })

    const restoreGateway = stubGateway(request)
    const restoreDesktop = stubDesktopLogs(['boot ok', 'ws connected'])

    try {
      requestSendDiagnostics('layer: streaming\ncode: stream_drop')
      await confirmSendDiagnostics()

      expect(request).toHaveBeenCalledTimes(1)
      const [method, params] = request.mock.calls[0]

      expect(method).toBe('diagnostics.share_nous')
      expect(params.error_context).toContain('stream_drop')
      expect(params.extra_files['desktop.log']).toContain('ws connected')

      const state = $sendDiagnostics.get()

      expect(state?.phase).toBe('done')
      expect(state?.result?.viewUrl).toBe('https://nas.example/view/x1')
    } finally {
      restoreDesktop()
      restoreGateway()
    }
  })

  it('omits extra_files when the desktop IPC is unavailable (browser dashboard)', async () => {
    const request = vi.fn().mockResolvedValue({ ok: true, view_url: 'https://nas.example/view/x2' })
    const restoreGateway = stubGateway(request)
    const restoreDesktop = stubDesktopLogs(null)

    try {
      requestSendDiagnostics()
      await confirmSendDiagnostics()

      const [, params] = request.mock.calls[0]

      expect(params.extra_files).toBeUndefined()
      expect(params.error_context).toBeUndefined()
      expect($sendDiagnostics.get()?.phase).toBe('done')
    } finally {
      restoreDesktop()
      restoreGateway()
    }
  })

  it('surfaces upload failures inline and keeps the dialog open', async () => {
    const request = vi.fn().mockResolvedValue({ ok: false, error: 'NAS unavailable' })
    const restoreGateway = stubGateway(request)
    const restoreDesktop = stubDesktopLogs(null)

    try {
      requestSendDiagnostics()
      await confirmSendDiagnostics()

      const state = $sendDiagnostics.get()

      expect(state?.phase).toBe('error')
      expect(state?.error).toContain('NAS unavailable')
    } finally {
      restoreDesktop()
      restoreGateway()
    }
  })

  it('confirm is a no-op outside the consent phase (no double upload)', async () => {
    const request = vi.fn().mockResolvedValue({ ok: true })
    const restoreGateway = stubGateway(request)
    const restoreDesktop = stubDesktopLogs(null)

    try {
      requestSendDiagnostics()
      await confirmSendDiagnostics()
      await confirmSendDiagnostics()

      expect(request).toHaveBeenCalledTimes(1)
    } finally {
      restoreDesktop()
      restoreGateway()
    }
  })

  it('dismiss clears the dialog state', () => {
    requestSendDiagnostics()
    dismissSendDiagnostics()

    expect($sendDiagnostics.get()).toBeNull()
  })

  it('dismissal mid-upload is immediate and a stale completion cannot resurrect the dialog', async () => {
    let resolveRequest: (value: unknown) => void = () => {}

    const request = vi.fn().mockImplementation(() => new Promise(resolve => (resolveRequest = resolve)))

    const restoreGateway = stubGateway(request as never)
    const restoreDesktop = stubDesktopLogs(null)

    try {
      requestSendDiagnostics()
      const pending = confirmSendDiagnostics()

      // Wait for the request to actually start, then dismiss mid-flight.
      await vi.waitFor(() => expect(request).toHaveBeenCalled())
      dismissSendDiagnostics()
      expect($sendDiagnostics.get()).toBeNull()

      // The upload completes AFTER dismissal — it must not write back.
      resolveRequest({ ok: true, view_url: 'https://nas.example/view/stale' })
      await pending

      expect($sendDiagnostics.get()).toBeNull()

      // A NEW dialog opened after the stale completion is untouched by it.
      requestSendDiagnostics('fresh')
      expect($sendDiagnostics.get()?.phase).toBe('consent')
    } finally {
      restoreDesktop()
      restoreGateway()
    }
  })
})
