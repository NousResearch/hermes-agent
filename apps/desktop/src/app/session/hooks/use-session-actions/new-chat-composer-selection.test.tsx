// Regression: a stored-session preview paints its model on the composer with an
// EMPTY provider while `session.resume` is in flight. A new chat started before
// that resume returned kept the painted half of a pair, and a manual selection
// then shipped it on `session.create` (#125336).
import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $currentModel,
  $currentProvider,
  setActiveSessionId,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider,
  setMessages,
  setResumeFailedSessionId
} from '@/store/session'

import type { ClientSessionState } from '../../../types'

import { desktopSessionCreateParams } from './session-create-request'
import { applyStoredSessionPreviewRuntimeInfo } from './utils'

import { useSessionActions } from './index'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getAllSessionMessages: vi.fn(),
  getLatestSessionMessages: vi.fn(),
  getSession: vi.fn(),
  listAllProfileSessions: vi.fn(),
  setApiRequestProfile: vi.fn()
}))

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGatewayAgent: vi.fn().mockResolvedValue(undefined),
  ensureGatewayProfile: vi.fn().mockResolvedValue(undefined)
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn(),
  requestGatewayForProfile: vi.fn(),
  retainGatewayForAgent: vi.fn(async () => () => undefined)
}))

vi.mock('@/components/pane-shell/tree/store', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  noteActiveTreeGroup: vi.fn(),
  revealTreePane: vi.fn()
}))

type Handle = Pick<ReturnType<typeof useSessionActions>, 'resumeSession' | 'startFreshSessionDraft'>
type RequestGateway = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

interface ResumeSettle {
  reject: (reason: unknown) => void
  resolve: (value: unknown) => void
}

function Harness({ onReady, requestGateway }: { onReady: (handle: Handle) => void; requestGateway: RequestGateway }) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef: ref<string | null>(null),
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    getRoutedStoredSessionId: () => null,
    navigate: vi.fn() as never,
    requestGateway,
    resetViewSync: vi.fn(),
    routedSessionId: null,
    runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef: ref<string | null>(null),
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  useEffect(() => {
    onReady(actions)
  }, [actions, onReady])

  return null
}

async function mountHarness(requestGateway: RequestGateway): Promise<Handle> {
  let handle: Handle | undefined
  render(<Harness onReady={value => (handle = value)} requestGateway={requestGateway} />)
  await waitFor(() => expect(handle).toBeDefined())

  return handle as Handle
}

function paintPreview() {
  applyStoredSessionPreviewRuntimeInfo({ cwd: '', model: 'gpt-6.1-terra' }, 'stored-preview')
  expect($currentModel.get()).toBe('gpt-6.1-terra')
  expect($currentProvider.get()).toBe('')
}

describe('a new chat after a stored-session preview whose resume was abandoned', () => {
  afterEach(() => {
    cleanup()
    setActiveSessionId(null)
    setResumeFailedSessionId(null)
    setMessages([])
  })

  const startDraftAfterPreview = async () => {
    const handle = await mountHarness(vi.fn(async () => ({}) as never))

    paintPreview()
    act(() => handle.startFreshSessionDraft({ preserveRoute: true, workspaceTarget: null }))
  }

  it('restores a manual pick as a complete model and provider pair', async () => {
    setCurrentModel('claude-opus-5')
    setCurrentProvider('anthropic')
    setCurrentModelSource('manual')

    await startDraftAfterPreview()

    expect($currentModel.get()).toBe('claude-opus-5')
    expect($currentProvider.get()).toBe('anthropic')
    expect(await desktopSessionCreateParams('/x')).toMatchObject({ model: 'claude-opus-5', provider: 'anthropic' })
  })

  it('restores an unpinned default without pinning the new session', async () => {
    setCurrentModel('claude-sonnet-5')
    setCurrentProvider('anthropic')
    setCurrentModelSource('default')

    await startDraftAfterPreview()

    expect($currentModel.get()).toBe('claude-sonnet-5')
    expect($currentProvider.get()).toBe('anthropic')

    const params = await desktopSessionCreateParams('/x')

    expect(params).not.toHaveProperty('model')
    expect(params).not.toHaveProperty('provider')
  })

  it('leaves nothing stale when no selection was ever stored', async () => {
    localStorage.removeItem('hermes.desktop.composer.model')
    localStorage.removeItem('hermes.desktop.composer.provider')
    setCurrentModelSource('')

    await startDraftAfterPreview()

    expect($currentModel.get()).toBe('')
    expect($currentProvider.get()).toBe('')
  })

  it.each([
    ['answers', (settle: ResumeSettle) => settle.resolve({ info: { model: 'gpt-6.1-terra', provider: 'openai-codex' }, session_id: 'late' })],
    ['fails', (settle: ResumeSettle) => settle.reject(new Error('resume failed'))]
  ])('keeps the restored pair when the resume %s after the new chat', async (_label, finish) => {
    setCurrentModel('claude-opus-5')
    setCurrentProvider('anthropic')
    setCurrentModelSource('manual')

    let settle: ResumeSettle | null = null

    const requestGateway = vi.fn((method: string) =>
      method === 'session.resume'
        ? new Promise((resolve, reject) => {
            settle = { reject, resolve }
          })
        : Promise.resolve({})
    ) as never

    const handle = await mountHarness(requestGateway)

    paintPreview()
    const resume = handle.resumeSession('stored-preview').catch(() => undefined)
    await waitFor(() => expect(settle).not.toBeNull())

    act(() => handle.startFreshSessionDraft({ preserveRoute: true, workspaceTarget: null }))
    finish(settle!)
    await act(async () => {
      await resume
    })

    expect($currentModel.get()).toBe('claude-opus-5')
    expect($currentProvider.get()).toBe('anthropic')
  })
})
