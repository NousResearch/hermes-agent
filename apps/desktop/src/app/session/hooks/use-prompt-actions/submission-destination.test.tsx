import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $sessions } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import { useSlashCommand } from './slash'
import { captureSubmissionDestination } from './submission-destination'
import { useSubmitPrompt } from './submit'
import type { GatewayRequest } from './utils'

const wire = vi.hoisted(() => ({ request: vi.fn(), release: vi.fn() }))
vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  requestGatewayForAgent: wire.request,
  requestGatewayForProfile: wire.request,
  retainGatewayForSessionTurn: vi.fn(async () => wire.release)
}))

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

function setup() {
  const activeSessionIdRef = { current: 'runtime-a' as string | null }
  const selectedStoredSessionIdRef = { current: 'stored-a' as string | null }

  const requestGateway = vi.fn(
    async (_method: string, params?: Record<string, unknown>) =>
      ({ admission_id: params?.submission_id, status: 'started' }) as never
  )

  const busyRef = { current: false }
  const state = createClientSessionState()

  const deps = {
    activeSessionIdRef,
    selectedStoredSessionIdRef,
    busyRef,
    copy: en.desktop,
    createBackendSessionForSend: vi.fn(async () => null),
    getRoutedStoredSessionId: () => null,
    getRuntimeIdForStoredSession: () => 'runtime-a',
    getRouteToken: () => 'same-route',
    requestGateway: requestGateway as GatewayRequest,
    runtimeIdByStoredSessionIdRef: { current: new Map([['stored-a', 'runtime-a']]) },
    resumeStoredSession: vi.fn(),
    syncAttachmentsForSubmit: vi.fn(async (sessionId: string) => ({ sessionId, attachments: [] })),
    updateSessionState: vi.fn((_sid, updater) => updater(state))
  }

  return { deps, requestGateway }
}

beforeEach(() => {
  window.localStorage.clear()
  $sessions.set([])
  $sessionStates.set({})
  $connection.set(null)
  $activeGatewayProfile.set('default')
  wire.request.mockReset()
  wire.request.mockImplementation(async (_connection, _profile, _method, params) => ({
    admission_id: params?.submission_id,
    status: 'started'
  }))
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('submission intent destinations', () => {
  it('stages canonical images on the captured owner without a legacy image RPC', async () => {
    const { uploadComposerAttachment } = await import('.')
    const original = window.hermesDesktop
    const gate = deferred<string>()
    const api = vi.fn(async () => ({ path: '/owner/cache/images/exact.png', mime_type: 'image/png' }))
    window.hermesDesktop = { ...original, readFileDataUrl: () => gate.promise, api } as never
    $connection.set({ connectionId: 'local', wsUrl: 'ws://localhost/ws?native_dial=1' } as never)
    $activeGatewayProfile.set('alice')
    const request = vi.fn() as GatewayRequest

    try {
      const pending = uploadComposerAttachment({ id: 'img', kind: 'image', label: 'exact.png', path: '/native/exact.png' },
        { remote: false, sessionId: 'runtime-a', requestGateway: request })

      $activeGatewayProfile.set('bob')
      gate.resolve('data:image/png;base64,aGVsbG8=')
      const staged = await pending
      expect(staged).toMatchObject({ path: '/owner/cache/images/exact.png', mime: 'image/png', attachedSessionId: 'runtime-a' })
      expect(api).toHaveBeenCalledWith(expect.objectContaining({ method: 'POST', path: '/api/chat/image-upload', connectionId: 'local', profile: 'alice', body: { data_url: 'data:image/png;base64,aGVsbG8=', filename: 'exact.png' } }))
      expect(request).not.toHaveBeenCalled()
    } finally { window.hermesDesktop = original }
  })

  it('retains exact canonical image parameters and identity through ambiguous submit and remount', async () => {
    const { deps, requestGateway } = setup()
    $connection.set({ connectionId: 'local', wsUrl: 'ws://localhost/ws?native_dial=1' } as never)
    const image = { id: 'image-retry', kind: 'image' as const, label: 'exact.png', path: '/owner/cache/images/exact.png', mime: 'image/png' }
    deps.syncAttachmentsForSubmit.mockResolvedValue({ sessionId: 'runtime-a', attachments: [image] } as never)
    requestGateway.mockRejectedValueOnce(new Error('connection closed'))
    let hook = renderHook(() => useSubmitPrompt(deps))
    await act(async () => { expect(await hook.result.current('image retry', { attachments: [image] })).toBe(false) })
    const first = requestGateway.mock.calls.find(call => call[0] === 'prompt.submit')![1]
    expect(first?.attachments).toEqual([{ path: image.path, mime: image.mime }])
    hook.unmount()
    hook = renderHook(() => useSubmitPrompt(deps))
    await act(async () => { expect(await hook.result.current('image retry', { attachments: [image] })).toBe(true) })
    expect(requestGateway.mock.calls.filter(call => call[0] === 'prompt.submit')[1][1]).toEqual(first)
    expect(deps.syncAttachmentsForSubmit).toHaveBeenCalledOnce()
    expect(window.localStorage.getItem('hermes.desktop.preparedSubmissions.v1')).not.toContain('image retry')
  })


  it('allows presentation and registry metadata replacement on the same authority', async () => {
    const connection = {
      baseUrl: 'http://localhost:1',
      wsUrl: 'ws://localhost:1/ws',
      token: 'secret',
      connectionId: 'local',
      mode: 'local',
      profile: 'default'
    }

    $connection.set(connection as never)
    const request = vi.fn(async () => ({ ok: true }))
    const captured = captureSubmissionDestination(null, request as GatewayRequest)
    $connection.set({ ...connection, registryScoped: true, isVisible: false } as never)
    await expect(captured.requestGateway('prompt.submit', { text: 'hello' })).resolves.toEqual({ ok: true })
    expect(request).toHaveBeenCalledOnce()
  })

  it('rejects actual endpoint, credential, connection and profile changes', async () => {
    const connection = {
      baseUrl: 'http://localhost:1',
      wsUrl: 'ws://localhost:1/ws',
      token: 'secret',
      connectionId: 'local',
      mode: 'local',
      profile: 'default'
    }

    const request = vi.fn(async () => ({}))

    for (const change of [
      { baseUrl: 'http://other' },
      { wsUrl: 'ws://other/ws' },
      { token: 'other' },
      { connectionId: 'other' },
      { profile: 'other' },
      { mode: 'remote' }
    ]) {
      $connection.set(connection as never)
      const captured = captureSubmissionDestination(null, request as GatewayRequest)
      $connection.set({ ...connection, ...change } as never)
      await expect(captured.requestGateway('prompt.submit')).rejects.toThrow('Submission destination changed')
    }

    expect(request).not.toHaveBeenCalled()
  })
  it('reuses a prepared direct submission after ambiguous failure without restaging attachments', async () => {
    const { deps, requestGateway } = setup()
    requestGateway.mockRejectedValueOnce(new Error('connection closed'))
    let hook = renderHook(() => useSubmitPrompt(deps))
    await act(async () => {
      expect(await hook.result.current('retry me')).toBe(false)
    })
    const journal = window.localStorage.getItem('hermes.desktop.preparedSubmissions.v1')
    expect(journal).toContain('retry me')
    hook.unmount()
    window.localStorage.clear()
    window.localStorage.setItem('hermes.desktop.preparedSubmissions.v1', journal!)
    // A same-text send in another profile/session is a different intent; it
    // must neither reuse nor retire the original uncertain admission.
    $activeGatewayProfile.set('bob')
    deps.selectedStoredSessionIdRef.current = 'stored-b'
    deps.activeSessionIdRef.current = 'runtime-b'
    deps.runtimeIdByStoredSessionIdRef.current.set('stored-b', 'runtime-b')
    hook = renderHook(() => useSubmitPrompt(deps))
    await act(async () => {
      expect(await hook.result.current('retry me')).toBe(true)
    })
    expect(requestGateway.mock.calls[1][1]?.submission_id).not.toBe(requestGateway.mock.calls[0][1]?.submission_id)
    hook.unmount()
    $activeGatewayProfile.set('default')
    deps.selectedStoredSessionIdRef.current = 'stored-a'
    deps.activeSessionIdRef.current = 'runtime-a'
    hook = renderHook(() => useSubmitPrompt(deps))
    const { result } = hook
    await act(async () => {
      expect(await result.current('retry me')).toBe(true)
    })

    const calls = requestGateway.mock.calls.filter(
      call => call[0] === 'prompt.submit' && call[1]?.session_id === 'runtime-a'
    )

    expect(calls).toHaveLength(2)
    expect(calls[1][1]).toEqual(calls[0][1])
    expect(deps.syncAttachmentsForSubmit).toHaveBeenCalledTimes(2)
    await act(async () => {
      expect(await result.current('retry me')).toBe(true)
    })
    expect(requestGateway.mock.calls.at(-1)?.[1]?.submission_id).not.toBe(calls[0][1]?.submission_id)
  })

  it.each([false, true])('keeps new-session %s kickoff identity and expansion across journal reload', async slash => {
    const { deps, requestGateway } = setup()
    deps.activeSessionIdRef.current = null
    deps.selectedStoredSessionIdRef.current = null
    $connection.set({ connectionId: 'local', profile: 'default', mode: 'local' } as never)
    deps.createBackendSessionForSend.mockImplementation(async () => {
      deps.activeSessionIdRef.current = 'runtime-a'
      deps.selectedStoredSessionIdRef.current = 'stored-a'
      $sessions.set([{ id: 'stored-a', connection_id: 'local', profile: 'default' }] as never)

      return 'runtime-a' as never
    })
    const accepted = new Map<string, Record<string, unknown>>()
    let loseAck = true
    let expansions = 0
    requestGateway.mockImplementation(async (method, params) => {
      if (method === 'slash.exec') {
        expansions++

        return {
          type: 'skill',
          name: 'private-skill',
          message: `expanded-${expansions}`,
          display: '/private-skill'
        } as never
      }

      if (method !== 'prompt.submit') {
        return {} as never
      }

      const id = String(params?.submission_id)
      const previous = accepted.get(id)

      if (previous) {
        expect(params).toEqual(previous)
      } else {
        accepted.set(id, params!)
      }

      if (loseAck) {
        throw new Error('connection closed')
      }

      return { admission_id: id, status: 'terminal' } as never
    })
    wire.request.mockImplementation((_connection, _profile, method, params) => requestGateway(method, params))

    const mount = () =>
      renderHook(() => {
        const submitPromptText = useSubmitPrompt(deps)

        const runSlash = useSlashCommand({
          ...deps,
          submitPromptText,
          appendSessionTextMessage: vi.fn(),
          branchCurrentSession: async () => true,
          handleSkinCommand: () => '',
          handoffSession: async () => ({ ok: true }),
          openMemoryGraph: vi.fn(),
          refreshSessions: async () => undefined,
          startFreshSessionDraft: vi.fn()
        })

        return slash ? runSlash : submitPromptText
      })

    const input = slash ? '/private-skill' : 'first prompt'
    let hook = mount()
    await act(async () => {
      expect(await hook.result.current(input)).toBe(false)
    })
    const serialized = window.localStorage.getItem('hermes.desktop.preparedSubmissions.v1')
    expect(serialized).toBeTruthy()
    expect(Object.values(JSON.parse(serialized!))[0]).toMatchObject({ owner: { connectionId: 'local', profile: 'default' } })
    hook.unmount()
    window.localStorage.clear()
    window.localStorage.setItem('hermes.desktop.preparedSubmissions.v1', serialized!)
    loseAck = false
    hook = mount()
    await act(async () => {
      expect(await hook.result.current(input)).toBe(true)
    })
    expect(accepted.size).toBe(1)
    expect(deps.createBackendSessionForSend).toHaveBeenCalledTimes(1)
    expect(deps.syncAttachmentsForSubmit).toHaveBeenCalledTimes(1)
    expect(expansions).toBe(slash ? 1 : 0)
  })

  it.each(['missing', 'wrong', 'unknown', '4094', 'lost-legacy'])(
    'requires a matching receipt or one pre-admission legacy refusal: %s',
    async mode => {
      const { deps, requestGateway } = setup()
      const refusal = Object.assign(new Error('durable admission unsupported'), { code: 4094 })
      requestGateway.mockImplementation(async (_method, params) => {
        if (mode === '4094' || mode === 'lost-legacy') {
          if (params?.submission_id) {
            throw refusal
          }

          if (mode === 'lost-legacy') {
            throw new Error('connection closed')
          }

          return { ok: true } as never
        }

        if (mode === 'missing') {
          return { ok: true } as never
        }

        return { admission_id: mode === 'wrong' ? 'wrong-id' : params?.submission_id, status: 'unknown' } as never
      })
      let hook = renderHook(() => useSubmitPrompt(deps))
      await act(async () => {
        expect(await hook.result.current('receipt')).toBe(mode === '4094')
      })
      hook.unmount()
      hook = renderHook(() => useSubmitPrompt(deps))

      if (mode !== '4094') {
        await act(async () => {
          expect(await hook.result.current('receipt')).toBe(false)
        })
        const attempts = requestGateway.mock.calls.filter(call => call[0] === 'prompt.submit')

        if (mode === 'lost-legacy') { expect(attempts).toHaveLength(2) }
        const identified = attempts.filter(call => call[1]?.submission_id)
        expect(new Set(identified.map(call => call[1]?.submission_id)).size).toBe(1)
        expect(attempts.filter(call => !call[1]?.submission_id)).toHaveLength(mode === 'lost-legacy' ? 1 : 0)
      }
    }
  )

  it('assigns one direct ID before preprocessing and keeps it through busy retries', async () => {
    const { deps, requestGateway } = setup()
    let attempts = 0
    requestGateway.mockImplementation(async (_method, params) => {
      if (++attempts === 1) {
        throw new Error('session busy')
      }

      return { admission_id: params?.submission_id, status: 'started' } as never
    })
    const { result } = renderHook(() => useSubmitPrompt(deps))
    await act(async () => {
      expect(await result.current('hello')).toBe(true)
    })
    const ids = requestGateway.mock.calls.map(call => call[1]?.submission_id)
    expect(ids.length).toBeGreaterThan(1)
    expect(ids[0]).toEqual(expect.any(String))
    expect(new Set(ids).size).toBe(1)
  })

  it('pins the exact owner before attachment preprocessing changes connection and profile', async () => {
    const { deps, requestGateway } = setup()
    $sessions.set([{ id: 'stored-a', connection_id: 'owner-a', profile: 'alice' }] as never)
    const gate = deferred<{ sessionId: string; attachments: [] }>()
    deps.syncAttachmentsForSubmit.mockImplementation(() => gate.promise)
    const { result } = renderHook(() => useSubmitPrompt(deps))
    let pending!: Promise<boolean>
    act(() => {
      pending = result.current('private text')
    })
    $sessions.set([{ id: 'stored-a', connection_id: 'owner-b', profile: 'bob' }] as never)
    $activeGatewayProfile.set('bob')
    await act(async () => {
      gate.resolve({ sessionId: 'runtime-a', attachments: [] })
      expect(await pending).toBe(true)
    })
    expect(deps.syncAttachmentsForSubmit).toHaveBeenCalledWith(
      'runtime-a',
      [],
      expect.objectContaining({
        storedSessionId: 'stored-a',
        requestGateway: expect.any(Function)
      })
    )
    expect(requestGateway).not.toHaveBeenCalled()
    expect(wire.request).toHaveBeenCalledWith(
      'owner-a',
      'alice',
      'prompt.submit',
      expect.objectContaining({ session_id: 'runtime-a', text: 'private text', submission_id: expect.any(String) }),
      expect.any(Number),
      undefined
    )
  })

  it('captures an attachment upload owner before native byte reading yields', async () => {
    const { uploadComposerAttachment } = await import('.')
    const gate = deferred<string>()
    const original = window.hermesDesktop
    window.hermesDesktop = { ...original, readFileDataUrl: vi.fn(() => gate.promise) } as never
    $sessions.set([{ id: 'stored-a', connection_id: 'owner-a', profile: 'alice' }] as never)
    wire.request.mockResolvedValue({ attached: true, ref_text: '@file:staged.txt' })
    const ambient = vi.fn(async () => ({ attached: true, ref_text: '@file:wrong.txt' })) as GatewayRequest
    const captured = captureSubmissionDestination('stored-a', ambient)
    $sessions.set([{ id: 'stored-a', connection_id: 'owner-b', profile: 'bob' }] as never)

    try {
      const pending = uploadComposerAttachment(
        { id: 'file-a', kind: 'file', label: 'private.txt', path: '/private.txt' },
        {
          remote: true,
          sessionId: 'runtime-a',
          storedSessionId: 'stored-a',
          requestGateway: captured.requestGateway
        }
      )

      $sessions.set([{ id: 'stored-a', connection_id: 'owner-b', profile: 'bob' }] as never)
      $activeGatewayProfile.set('bob')
      gate.resolve('data:text/plain;base64,cHJpdmF0ZQ==')
      await pending
      expect(wire.request).toHaveBeenCalledWith(
        'owner-a',
        'alice',
        'file.attach',
        expect.objectContaining({
          session_id: 'runtime-a',
          data_url: 'data:text/plain;base64,cHJpdmF0ZQ=='
        })
      )
      expect(ambient).not.toHaveBeenCalled()
    } finally {
      window.hermesDesktop = original
    }
  })

  it('pins slash fallback and generated kickoff to the entry owner and stored session', async () => {
    const { deps } = setup()
    $sessions.set([{ id: 'stored-a', connection_id: 'owner-a', profile: 'alice' }] as never)
    const gate = deferred<unknown>()
    const requests: Array<[string, string, string]> = []
    wire.request.mockImplementation(async (connection, profile, method) => {
      requests.push([connection, profile, method])

      if (method === 'slash.exec') {
        return gate.promise
      }

      return { type: 'skill', name: 'private-skill', message: 'expanded private skill' }
    })
    deps.requestGateway = vi.fn(async (method: string) => {
      if (method === 'slash.exec') {
        return gate.promise
      }

      return { type: 'skill', name: 'private-skill', message: 'expanded private skill' }
    }) as GatewayRequest
    const submitPromptText = vi.fn(async () => true)

    const { result } = renderHook(() =>
      useSlashCommand({
        ...deps,
        submitPromptText,
        appendSessionTextMessage: vi.fn(),
        branchCurrentSession: async () => true,
        handleSkinCommand: () => '',
        handoffSession: async () => ({ ok: true }),
        openMemoryGraph: vi.fn(),
        refreshSessions: async () => undefined,
        startFreshSessionDraft: vi.fn()
      })
    )

    let pending!: Promise<boolean>
    await act(async () => {
      pending = result.current('/private-skill')
    })
    deps.activeSessionIdRef.current = 'runtime-b'
    deps.selectedStoredSessionIdRef.current = 'stored-b'
    $sessions.set([{ id: 'stored-a', connection_id: 'owner-b', profile: 'bob' }] as never)
    $activeGatewayProfile.set('bob')
    await act(async () => {
      gate.resolve({ type: 'skill', name: 'private-skill', message: 'expanded private skill' })
      await pending
    })
    expect(requests).toEqual([['owner-a', 'alice', 'slash.exec']])
    expect(submitPromptText).toHaveBeenCalledWith(
      'expanded private skill',
      expect.objectContaining({
        sessionId: 'runtime-a',
        storedSessionId: 'stored-a',
        submission_id: expect.any(String),
        destination: expect.any(Object)
      })
    )
  })
})
