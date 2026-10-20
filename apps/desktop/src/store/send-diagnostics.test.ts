import { afterEach, describe, expect, it, vi } from 'vitest'

type GatewayRequest = (method: string, params?: Record<string, unknown>, timeout?: number) => Promise<unknown>

const routing = vi.hoisted(() => ({
  activeProfile: 'default',
  calls: [] as { connectionId: null | string; profile: string }[],
  request: null as GatewayRequest | null
}))

// The upload is a profile-routed RPC: a multiplexed backend serves several
// profiles on one socket, so the route (connection, profile) must be explicit.
vi.mock('@/store/gateway', () => ({
  activeGatewayProfileKey: () => routing.activeProfile,
  requestGatewayForAgent: (
    connectionId: null | string,
    profile: string,
    method: string,
    params?: Record<string, unknown>,
    timeout?: number
  ) => {
    routing.calls.push({ connectionId, profile })

    if (!routing.request) {
      return Promise.reject(new Error(`Hermes gateway unavailable for profile "${profile}"`))
    }

    return routing.request(method, params, timeout)
  }
}))

const { $sendDiagnostics, confirmSendDiagnostics, dismissSendDiagnostics, requestSendDiagnostics } =
  await import('@/store/send-diagnostics')

function stubGateway(request: GatewayRequest) {
  routing.request = request

  return () => {
    routing.request = null
  }
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
    routing.activeProfile = 'default'
    routing.calls = []
    vi.restoreAllMocks()
  })

  it('opens in consent phase without any network I/O', () => {
    const request = vi.fn()
    const restore = stubGateway(request)

    try {
      requestSendDiagnostics('layer: provider')

      expect($sendDiagnostics.get()).toEqual({ errorContext: 'layer: provider', phase: 'consent', profile: 'default' })
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
  it('uploads for the profile focused when the dialog opened, even after a profile switch', async () => {
    const request = vi.fn().mockResolvedValue({ ok: true, view_url: 'https://nas.example/view/w1' })
    const restoreGateway = stubGateway(request)
    const restoreDesktop = stubDesktopLogs(null)

    try {
      routing.activeProfile = 'work'
      requestSendDiagnostics('crash')
      routing.activeProfile = 'default'
      await confirmSendDiagnostics()

      expect(routing.calls).toEqual([{ connectionId: null, profile: 'work' }])
      expect(request.mock.calls[0][1].session_id).toBeUndefined()
    } finally {
      restoreDesktop()
      restoreGateway()
    }
  })

  it('scopes a session error to the session owner and its runtime id', async () => {
    const request = vi.fn().mockResolvedValue({ ok: true, view_url: 'https://nas.example/view/s1' })
    const restoreGateway = stubGateway(request)
    const restoreDesktop = stubDesktopLogs(null)

    try {
      routing.activeProfile = 'default'
      requestSendDiagnostics('layer: provider', { connectionId: 'ssh-lab', profile: 'research', sessionId: 'rt-42' })
      await confirmSendDiagnostics()

      expect(routing.calls).toEqual([{ connectionId: 'ssh-lab', profile: 'research' }])

      const [method, params] = request.mock.calls[0]

      expect(method).toBe('diagnostics.share_nous')
      expect(params.session_id).toBe('rt-42')
    } finally {
      restoreDesktop()
      restoreGateway()
    }
  })
})
