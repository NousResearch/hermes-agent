import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { $composerAttachments, type ComposerAttachment } from '@/store/composer'
import { $notifications, clearNotifications } from '@/store/notifications'
import { dropSessionState } from '@/store/session-states'

import { actRender, Harness, type HarnessHandle, RUNTIME_SESSION_ID } from './index-test-utils'
import { clearSingleFlightSessionResumeState } from './single-flight-resume'

const SLASH_METRIC = 'shared_metrics.slash_command'

vi.mock('@/hermes', () => ({
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: 'session' })),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getHermesConfig: vi.fn(async () => ({ display: { timestamps: false } })),
  getHermesConfigDefaults: vi.fn(async () => ({})),
  getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn()
}))

const PASTE_PATH_A = '/home/user/.hermes/composer-pastes/pasted_content_2026-11-01_12-00-00-000_aaaaaa.txt'
const PASTE_PATH_B = '/home/user/.hermes/composer-pastes/pasted_content_2026-11-01_12-00-01-000_bbbbbb.txt'

function makePasteChip(path = PASTE_PATH_A): ComposerAttachment {
  return {
    id: `file:${path}`,
    kind: 'file',
    label: 'Pasted content (4.9 KB)',
    path,
    refText: `@file:\`${path}\``,
    titlePreview: 'truncated-1000-char-preview'
  }
}

describe('usePromptActions slash commands with pasted-content chips', () => {
  const readFileText = vi.fn()

  beforeEach(() => {
    readFileText.mockReset()
    clearNotifications()
    $composerAttachments.set([])
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileText }
    })
  })

  afterEach(() => {
    cleanup()
    clearNotifications()
    $composerAttachments.set([])
    dropSessionState(RUNTIME_SESSION_ID)
    clearSingleFlightSessionResumeState()
    vi.restoreAllMocks()
  })

  it('inlines a paste chip into /goal after a newline and sends the kickoff without any paste attachment', async () => {
    const fullPaste = `${'goal line one\n'.repeat(250)}final goal line`
    readFileText.mockResolvedValueOnce({
      binary: false,
      byteSize: fullPaste.length,
      path: PASTE_PATH_A,
      text: fullPaste,
      truncated: false
    })

    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === SLASH_METRIC) {
        return {} as never
      }

      calls.push({ method, params })

      if (method === 'slash.exec') {
        return {
          type: 'send',
          notice: '⊙ Goal set. Starting now.',
          message: '[Active Goal] Start working on the goal.'
        } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    const ok = await handle!.submitText('/goal', {
      attachments: [makePasteChip(PASTE_PATH_A)]
    })

    expect(ok).toBe(true)
    expect(readFileText).toHaveBeenCalledExactlyOnceWith(PASTE_PATH_A)
    expect(calls.map(c => c.method)).toEqual(['slash.exec', 'prompt.submit'])
    expect(calls[0]?.params).toEqual({
      command: `goal\n${fullPaste}`,
      session_id: RUNTIME_SESSION_ID
    })
    expect(calls[1]?.params).toEqual({
      session_id: RUNTIME_SESSION_ID,
      submission_id: expect.any(String),
      text: '[Active Goal] Start working on the goal.'
    })
    expect($notifications.get().some(note => note.kind === 'warning')).toBe(false)
  })

  it('inlines two paste chips in chip order with a newline between them', async () => {
    const firstPaste = 'first pasted section'
    const secondPaste = 'second pasted section'
    readFileText
      .mockResolvedValueOnce({ path: PASTE_PATH_A, text: firstPaste })
      .mockResolvedValueOnce({ path: PASTE_PATH_B, text: secondPaste })

    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === SLASH_METRIC) {
        return {} as never
      }

      calls.push({ method, params })

      if (method === 'slash.exec') {
        return { output: '⊙ Goal set.' } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    const ok = await handle!.submitText('/goal', {
      attachments: [makePasteChip(PASTE_PATH_A), makePasteChip(PASTE_PATH_B)]
    })

    expect(ok).toBe(true)
    expect(calls).toEqual([
      {
        method: 'slash.exec',
        params: {
          command: `goal\n${firstPaste}\n${secondPaste}`,
          session_id: RUNTIME_SESSION_ID
        }
      }
    ])
  })

  it.each([
    ['a paste chip', true],
    ['an already-expanded queued command', false]
  ])('keeps the paste body trailing whitespace verbatim in slash.exec (%s)', async (_label, viaChip) => {
    const pasteBody = 'goal body with trailing space   \n'
    readFileText.mockResolvedValueOnce({ path: PASTE_PATH_A, text: pasteBody })

    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method !== SLASH_METRIC) {
        calls.push({ method, params })
      }

      return (method === 'slash.exec' ? { output: '⊙ Goal set.' } : {}) as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    const ok = viaChip
      ? await handle!.submitText('/goal  ', { attachments: [makePasteChip(PASTE_PATH_A)] })
      : await handle!.submitText(`/goal\n${pasteBody}`, { attachments: [] })

    expect(ok).toBe(true)
    expect(calls).toEqual([
      { method: 'slash.exec', params: { command: `goal\n${pasteBody}`, session_id: RUNTIME_SESSION_ID } }
    ])
  })

  it.each([
    ['truncated: true', () => Promise.resolve({ path: PASTE_PATH_A, text: 'cut', truncated: true })],
    ['binary: true', () => Promise.resolve({ binary: true, path: PASTE_PATH_A, text: 'bin' })],
    [
      '{ ok: false }',
      () => Promise.resolve({ error: 'ENOENT', message: 'Missing', ok: false as const, path: PASTE_PATH_A })
    ],
    ['thrown error', () => Promise.reject(new Error('disk error'))]
  ])('refuses without calling the gateway when reading a paste chip fails (%s)', async (_label, mockRead) => {
    readFileText.mockImplementationOnce(mockRead)
    const requestGateway = vi.fn(async () => ({}) as never)

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    const ok = await handle!.submitText('/goal', {
      attachments: [makePasteChip(PASTE_PATH_A)]
    })

    expect(ok).toBe(false)
    expect(requestGateway).not.toHaveBeenCalled()
    expect($notifications.get()).toEqual(
      expect.arrayContaining([expect.objectContaining({ message: en.desktop.slashCommandIgnoredBody })])
    )
  })

  it('creates a new session instead of hijacking an existing chat when navigating from a fresh draft during a slow paste read', async () => {
    let resolveRead!: (value: { path: string; text: string }) => void
    readFileText.mockImplementationOnce(
      () =>
        new Promise<{ path: string; text: string }>(resolve => {
          resolveRead = resolve
        })
    )

    const activeSessionIdRef = { current: null as string | null }
    const selectedStoredSessionIdRef = { current: null as string | null }
    const routedRef = { current: null as string | null }
    const createBackendSessionForSend = vi.fn(async () => 'rt-fresh-created')
    const calls: { method: string; params?: Record<string, unknown> }[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === SLASH_METRIC) {
        return {} as never
      }

      calls.push({ method, params })

      if (method === 'slash.exec') {
        return {
          message: 'fresh goal kickoff',
          notice: '⊙ Goal set: fresh goal body',
          type: 'send'
        } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        createBackendSessionForSend={createBackendSessionForSend}
        getRoutedStoredSessionId={() => routedRef.current}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        storedSessionId={null}
      />
    )

    const pendingSubmit = handle!.submitText('/goal', {
      attachments: [makePasteChip(PASTE_PATH_A)]
    })

    // User switches from the fresh draft to existing session B while the paste file is being read.
    activeSessionIdRef.current = 'rt-existing-b'
    selectedStoredSessionIdRef.current = 'stored-existing-b'
    routedRef.current = 'stored-existing-b'

    resolveRead({ path: PASTE_PATH_A, text: 'fresh goal body' })

    expect(await pendingSubmit).toBe(true)
    expect(createBackendSessionForSend).toHaveBeenCalledTimes(1)
    expect(calls).toEqual([
      {
        method: 'slash.exec',
        params: {
          command: 'goal\nfresh goal body',
          session_id: 'rt-fresh-created'
        }
      },
      {
        method: 'prompt.submit',
        params: expect.objectContaining({
          session_id: 'rt-fresh-created',
          text: 'fresh goal kickoff'
        })
      }
    ])
  })
})
