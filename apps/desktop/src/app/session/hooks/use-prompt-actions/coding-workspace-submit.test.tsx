import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $codingWorkspaceDrafts, codingWorkspaceKey } from '@/store/coding-workspaces'
import type { ComposerAttachment } from '@/store/composer'
import { requestGatewayForAgent } from '@/store/gateway'
import { $newChatRoute } from '@/store/profile'
import { $currentCwd, setSessionOwnerHint, setSessions } from '@/store/session'

import { useSubmitPrompt } from './submit'

vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn()
}))

afterEach(() => {
  cleanup()
  $codingWorkspaceDrafts.set({})
  $newChatRoute.set(null)
  $currentCwd.set('')
  setSessions(() => [])
  vi.clearAllMocks()
})

it.each([null, 'remote'])(
  'remaps selected-root references for owner %s before attach and sends once on retry',
  async connectionId => {
    const owner = { connectionId, profile: 'coder', draftKey: '__new__' }

    const prepared = {
      requestId: 'request',
      sourcePath: '/repo',
      cwd: '/repo/.worktrees/task',
      projectId: 'p',
      repoRoot: '/repo',
      branch: 'task'
    }

    const attachments: ComposerAttachment[] = [
      { id: 'folder', occurrenceId: 'one', kind: 'folder', label: 'repo', path: '/repo', refText: '@folder:/repo' },
      {
        id: 'file',
        occurrenceId: 'two',
        kind: 'file',
        label: 'file name.ts',
        path: '/repo/file name.ts',
        refText: '@file:`/repo/file name.ts`'
      },
      {
        id: 'other',
        occurrenceId: 'three',
        kind: 'file',
        label: 'other',
        path: '/unrelated/file',
        refText: '@file:/unrelated/file'
      }
    ]

    const original = structuredClone(attachments)
    const active = { current: null as string | null }
    const selected = { current: null as string | null }
    const busy = { current: false }
    const runtimeIds = { current: new Map<string, string>() }
    let state = createClientSessionState()
    const calls: string[] = []
    let fail = true
    vi.mocked(requestGatewayForAgent).mockImplementation(async (connection, profile, method, params) => {
      expect([connection, profile, method]).toEqual([owner.connectionId, owner.profile, 'session.workspace.references'])
      expect(params).toMatchObject({ session_id: 'runtime', paths: attachments.map(a => a.path) })
      calls.push('references')

      if (fail) {throw new Error('file is missing from selected checkout')}

      return { paths: [prepared.cwd, `${prepared.cwd}/file name.ts`, null], text: params?.text } as never
    })

    const create = vi.fn(async () => {
      calls.push('create')
      active.current = 'runtime'
      selected.current = 'stored'
      runtimeIds.current.set('stored', 'runtime')

      if (owner.connectionId !== null) {setSessionOwnerHint('stored', { ...owner, connectionId: owner.connectionId })}
      $codingWorkspaceDrafts.set({
        [codingWorkspaceKey(owner)]: {
          owner,
          intent: { path: '/repo', mode: 'worktree' },
          status: 'bound',
          requestId: prepared.requestId,
          prepared,
          createdSession: { session_id: 'runtime', stored_session_id: 'stored' },
          sessionId: 'stored'
        }
      })

      return 'runtime'
    })

    const sync = vi.fn(async (sessionId: string, atts: ComposerAttachment[]) => {
      calls.push('attach')
      expect(atts[0]).toMatchObject({ path: prepared.cwd, refText: `@folder:${prepared.cwd}` })
      expect(atts[1]).toMatchObject({
        path: `${prepared.cwd}/file name.ts`,
        refText: `@file:\`${prepared.cwd}/file name.ts\``
      })
      expect(atts[2]).toBe(attachments[2])

      // A drop-time eager upload may finish while references are resolving.
      // Attachment sync can return that older staged copy; it cannot override
      // the authoritative checkout reference for this send.
      return {
        sessionId,
        attachments: atts.map((attachment, index) => (index === 1 ? attachments[index] : attachment))
      }
    })

    const request = vi.fn(async (method: string) => {
      calls.push(method)

      return {} as never
    })

    const remove = vi.fn()

    const { result } = renderHook(() =>
      useSubmitPrompt({
        activeSessionIdRef: active,
        busyRef: busy,
        copy: en.desktop,
        createBackendSessionForSend: create,
        getRoutedStoredSessionId: () => null,
        getRuntimeIdForStoredSession: () => active.current,
        getRouteToken: () => 'new',
        requestGateway: request,
        runtimeIdByStoredSessionIdRef: runtimeIds,
        resumeStoredSession: vi.fn(),
        selectedStoredSessionIdRef: selected,
        syncAttachmentsForSubmit: sync,
        updateSessionState: (_id, update) => (state = update(state)),
        scope: {
          readAttachments: () => attachments,
          removeAttachments: remove,
          setAwaitingResponse: vi.fn(),
          setBusy: vi.fn(),
          setMessages: vi.fn()
        }
      })
    )

    await act(async () => {
      expect(await result.current('original request')).toBe(false)
    })
    expect(calls).toEqual(['create', 'references'])
    expect(attachments).toEqual(original)
    expect(remove).not.toHaveBeenCalled()
    fail = false
    await act(async () => {
      const sent = await result.current('original request')
      expect(sent, JSON.stringify({ calls, messages: state.messages })).toBe(true)
    })
    expect(calls).toEqual(['create', 'references', 'references', 'attach', 'prompt.submit'])
    expect(create).toHaveBeenCalledOnce()
    expect(request).toHaveBeenCalledWith(
      'prompt.submit',
      expect.objectContaining({
        text: `@folder:${prepared.cwd}\n@file:\`${prepared.cwd}/file name.ts\`\n@file:/unrelated/file\n\noriginal request`
      }),
      expect.any(Number)
    )
    expect(attachments).toEqual(original)
    expect(remove).toHaveBeenCalledOnce()
  }
)

it.each([null, 'recovery-owner'].flatMap(connectionId =>
  [null, 'Session not found', 'reference escapes selected checkout'].map(retryError => ({ connectionId, retryError }))
))('recovers pending reference handoff on its exact owner once: $connectionId / $retryError', async ({ connectionId, retryError }) => {
  const owner = { connectionId, profile: 'coder', draftKey: '__new__' }
  const key = codingWorkspaceKey(owner)
  const storedId = `pending-stored-${connectionId}-${retryError}`
  const active = { current: 'pending-stale' as string | null }
  const selected = { current: storedId }
  const runtimeIds = { current: new Map([[storedId, 'pending-stale']]) }
  const raw = 'Read @file:repo/README.md'
  const mapped = 'Read @file:/checkout/README.md'

  const attachment: ComposerAttachment = {
    id: 'readme', kind: 'file', label: 'README.md', path: 'repo/README.md', refText: '@file:repo/README.md'
  }

  const original = structuredClone(attachment)

  if (connectionId !== null) {
    setSessionOwnerHint(storedId, { connectionId, profile: owner.profile })
  }

  $newChatRoute.set({ connectionId: 'wrong-owner', profile: 'other' })
  $codingWorkspaceDrafts.set({
    [key]: {
      owner, intent: { path: '/fixtures/repo', mode: 'worktree' }, status: 'bound', requestId: 'pending',
      referenceCwd: '/fixtures', sessionId: storedId,
      prepared: { requestId: 'pending', sourcePath: '/fixtures/repo', cwd: '/checkout', projectId: 'p', repoRoot: '/fixtures/repo', branch: 'task' },
      createdSession: { session_id: 'pending-stale', stored_session_id: storedId }
    }
  })
  let state = createClientSessionState()
  let error = retryError
  const calls: string[] = []
  vi.mocked(requestGatewayForAgent).mockImplementation(async (connection, profile, method, params) => {
    expect([connection, profile]).toEqual([connectionId, owner.profile])
    calls.push(`${method}:${params?.session_id}`)

    if (method === 'session.resume') {
      expect(params).toMatchObject({ session_id: storedId, profile: owner.profile, source: 'desktop' })

      return { session_id: 'pending-live' } as never
    }

    expect(method).toBe('session.workspace.references')
    expect(params).toMatchObject({ text: raw, paths: [original.path], reference_cwd: '/fixtures' })

    if (params?.session_id === 'pending-stale') {
      throw new Error('Session not found')
    }

    expect(params?.session_id).toBe('pending-live')
    expect(active.current).toBe('pending-live')
    expect(runtimeIds.current.get(storedId)).toBe('pending-live')

    if (error) {
      throw new Error(error)
    }

    return { text: mapped, paths: ['/checkout/README.md'] } as never
  })
  const create = vi.fn()

  const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    expect(method).toBe('prompt.submit')
    expect(params).toMatchObject({ session_id: 'pending-live', text: `@file:/checkout/README.md\n\n${mapped}` })

    return {} as never
  })

  const remove = vi.fn()

  const sync = vi.fn(async (sessionId: string, attachments: ComposerAttachment[]) => {
    expect(sessionId).toBe('pending-live')
    expect(attachments[0]).toMatchObject({ path: '/checkout/README.md', attachedSessionId: 'pending-live' })

    return { sessionId, attachments }
  })

  const { result } = renderHook(() => useSubmitPrompt({
    activeSessionIdRef: active, busyRef: { current: false }, copy: en.desktop,
    createBackendSessionForSend: create, getRoutedStoredSessionId: () => storedId,
    getRuntimeIdForStoredSession: id => runtimeIds.current.get(id) ?? null,
    getRouteToken: () => storedId, requestGateway: request, runtimeIdByStoredSessionIdRef: runtimeIds,
    resumeStoredSession: vi.fn(), selectedStoredSessionIdRef: selected, syncAttachmentsForSubmit: sync,
    updateSessionState: (id, update, stored) => {
      if (stored) {
        runtimeIds.current.set(stored, id)
      }

      return state = update(state)
    },
    scope: { readAttachments: () => [attachment], removeAttachments: remove,
      setAwaitingResponse: vi.fn(), setBusy: vi.fn(), setMessages: vi.fn() }
  }))

  await act(async () => {
    expect(await result.current(raw, { referenceCwd: '/checkout' })).toBe(!retryError)
  })
  expect(calls).toEqual([
    'session.workspace.references:pending-stale', `session.resume:${storedId}`, 'session.workspace.references:pending-live'
  ])

  if (retryError) {
    expect(request).not.toHaveBeenCalled()
    expect(sync).not.toHaveBeenCalled()
    expect(remove).not.toHaveBeenCalled()
    expect($codingWorkspaceDrafts.get()[key].referenceCwd).toBe('/fixtures')
    expect(state.messages[0].parts).toEqual([{ type: 'text', text: raw }])
    error = null
    await act(async () => {
      expect(await result.current(raw, { referenceCwd: '/checkout' })).toBe(true)
    })
    expect(calls.slice(3)).toEqual(['session.workspace.references:pending-live'])
  }

  expect(request).toHaveBeenCalledOnce()
  expect(create).not.toHaveBeenCalled()
  expect(attachment).toEqual(original)
  expect(remove).toHaveBeenCalledOnce()
})

it.each([undefined, null, '/fixtures'])(
  'retains authoritative pre-bind CWD for text-only refs with composer CWD %s across a failed first Send', async initialCwd => {
  const owner = { connectionId: 'inline-owner', profile: 'coder', draftKey: '__new__' }
  const key = codingWorkspaceKey(owner)

  const prepared = {
    requestId: 'inline',
    sourcePath: '/fixtures/retry-app',
    cwd: '/checkout',
    projectId: 'p',
    repoRoot: '/fixtures/retry-app',
    branch: 'task'
  }

  const text =
    'Read @file:`retry-app/README.md` and @folder:`retry-app/docs`; keep retry-app/README.md prose @file:/unrelated/file'

  const mapped =
    'Read @file:/checkout/README.md and @folder:/checkout/docs; keep retry-app/README.md prose @file:/unrelated/file'

  const active = { current: null as string | null }
  const selected = { current: null as string | null }
  const runtimeIds = { current: new Map<string, string>() }
  let state = createClientSessionState()
  let cwd = initialCwd
  $newChatRoute.set(owner)
  $codingWorkspaceDrafts.set({
    [key]: { owner, intent: { path: prepared.sourcePath, mode: 'worktree' }, status: 'ready', requestId: prepared.requestId }
  })
  $currentCwd.set('/wrong-main-composer')

  const create = vi.fn(async () => {
    // The base must be captured before create can provision or bind anything.
    expect($codingWorkspaceDrafts.get()[key].referenceCwd).toBe('/fixtures')
    cwd = prepared.cwd
    $currentCwd.set('/wrong-bound-main-composer')
    active.current = 'inline-runtime'
    selected.current = 'inline-stored'
    runtimeIds.current.set(selected.current, active.current)
    setSessionOwnerHint(selected.current, owner)
    $codingWorkspaceDrafts.set({
      [key]: {
        ...$codingWorkspaceDrafts.get()[key],
        owner,
        intent: { path: prepared.sourcePath, mode: 'worktree' },
        status: 'bound',
        requestId: prepared.requestId,
        prepared,
        sessionId: selected.current
      }
    })

    return active.current
  })

  const request = vi.fn(async (_method: string, _params?: Record<string, unknown>) => ({}) as never)
  const remove = vi.fn()
  const sync = vi.fn(async (sessionId: string, attachments: ComposerAttachment[]) => ({ sessionId, attachments }))

  const { result } = renderHook(() =>
    useSubmitPrompt({
      activeSessionIdRef: active,
      busyRef: { current: false },
      copy: en.desktop,
      createBackendSessionForSend: create,
      getRoutedStoredSessionId: () => null,
      getRuntimeIdForStoredSession: () => active.current,
      getRouteToken: () => 'new',
      requestGateway: request,
      runtimeIdByStoredSessionIdRef: runtimeIds,
      resumeStoredSession: vi.fn(),
      selectedStoredSessionIdRef: selected,
      syncAttachmentsForSubmit: sync,
      updateSessionState: (_id, update) => (state = update(state)),
      scope: {
        readAttachments: () => [],
        removeAttachments: remove,
        setAwaitingResponse: vi.fn(),
        setBusy: vi.fn(),
        setMessages: vi.fn()
      }
    })
  )

  let failReferences = true
  vi.mocked(requestGatewayForAgent).mockImplementation(async (connection, profile, method, params) => {
    expect([connection, profile]).toEqual([owner.connectionId, owner.profile])

    if (method === 'complete.path') {
      expect(create).not.toHaveBeenCalled()
      expect(params).toEqual({ profile: owner.profile, word: '' })

      return { sourceCwd: '/fixtures' } as never
    }

    expect(method).toBe('session.workspace.references')

    if (failReferences) {throw new Error('missing selected checkout ref')}

    return { paths: [], text: mapped } as never
  })
  await act(async () => {
    expect(await result.current(text, { referenceCwd: cwd })).toBe(false)
  })
  expect(requestGatewayForAgent).toHaveBeenCalledWith(
    owner.connectionId,
    owner.profile,
    'session.workspace.references',
    {
      session_id: 'inline-runtime',
      profile: owner.profile,
      paths: [],
      text,
      reference_cwd: '/fixtures'
    }
  )
  expect(request).not.toHaveBeenCalled()
  expect(sync).not.toHaveBeenCalled()
  expect(remove).not.toHaveBeenCalled()
  failReferences = false
  await act(async () => {
    expect(await result.current(text, { referenceCwd: cwd })).toBe(true)
  })
  const referenceCalls = vi.mocked(requestGatewayForAgent).mock.calls.filter(call => call[2] === 'session.workspace.references')
  expect(referenceCalls).toHaveLength(2)
  expect(referenceCalls[1][3]).toMatchObject({ text, reference_cwd: '/fixtures' })
  expect(vi.mocked(requestGatewayForAgent).mock.calls.filter(call => call[2] === 'complete.path'))
    .toHaveLength(initialCwd ? 0 : 1)
  expect(request).toHaveBeenCalledWith('prompt.submit', expect.objectContaining({ text: mapped }), expect.any(Number))
  expect(create).toHaveBeenCalledOnce()
  expect(remove).toHaveBeenCalledOnce()

  // The handoff is over. A later stale runtime must reach ordinary submit
  // recovery, not re-enter first-Send remapping with a cleared original base.
  setSessions(() => [{
    id: 'inline-stored', profile: owner.profile, connection_id: owner.connectionId,
    ended_at: null, input_tokens: 0, is_active: true, last_active: 0, message_count: 1,
    model: null, output_tokens: 0, preview: null, source: 'desktop', started_at: 0,
    title: 'Coding task', tool_call_count: 0
  }])
  vi.mocked(requestGatewayForAgent).mockImplementation(async () => {
    throw new Error('Unexpected first-Send reference RPC after handoff')
  })
  request.mockImplementation(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'session.resume') {
      expect(params).toMatchObject({ session_id: 'inline-stored', profile: owner.profile })

      return { session_id: 'later-live' } as never
    }

    expect(method).toBe('prompt.submit')

    if (params?.session_id === 'inline-runtime') {
      throw new Error('Session not found')
    }

    expect(params?.session_id).toBe('later-live')

    return {} as never
  })
  await act(async () => {
    const sent = await result.current('ordinary later message')
    expect(sent, JSON.stringify({ calls: request.mock.calls, messages: state.messages })).toBe(true)
  })
  expect(request.mock.calls.filter(call => call[0] === 'session.resume')).toHaveLength(1)
  // New relative refs already belong to the bound session's workspace. They
  // must not go through the old draft remapper without its original CWD.
  runtimeIds.current.set('inline-stored', 'later-live')
  await act(async () => {
    expect(await result.current('Read @file:README.md')).toBe(true)
  })
  expect(request).toHaveBeenLastCalledWith(
    'prompt.submit', expect.objectContaining({ session_id: 'later-live', text: 'Read @file:README.md' }), expect.any(Number)
  )
  expect(vi.mocked(requestGatewayForAgent).mock.calls.filter(call => call[2] === 'session.workspace.references'))
    .toHaveLength(2)
  expect(create).toHaveBeenCalledOnce()
})
