import { act, renderHook, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ComposerScope } from '@/app/chat/composer/scope'
import { useSessionTileActions } from '@/app/chat/session-tile-actions'
import { createClientSessionState } from '@/lib/chat-runtime'
import {
  type ComposerAttachment,
  createComposerAttachmentOccurrenceId,
  createComposerAttachmentScope
} from '@/store/composer'
import { $connection, $sessions } from '@/store/session'
import { $sessionStates, type SessionTileDelegate, setSessionTileDelegate } from '@/store/session-states'

const requestGateway = vi.fn()

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      desktop: {
        promptFailed: 'prompt failed',
        stopFailed: 'stop failed'
      }
    }
  })
}))

vi.mock('@/hermes', () => ({
  HermesGateway: class {},
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

const RUNTIME_ID = 'runtime-tile'
const STORED_ID = 'stored-tile'
const HOST_PATH = 'C:\\Users\\alice\\Pictures\\photo.png'
const STAGED_PATH = '/root/.hermes/attachments/photo.png'
const THUMBNAIL = 'data:image/png;base64,dGh1bWJuYWls'
const FULL_SOURCE = 'data:image/png;base64,b3JpZ2luYWw='

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(res => {
    resolve = res
  })

  return { promise, resolve }
}

function makeAttachment(occurrenceId = createComposerAttachmentOccurrenceId()): ComposerAttachment {
  return {
    detail: HOST_PATH,
    id: `image:${HOST_PATH}`,
    kind: 'image',
    label: 'photo.png',
    occurrenceId,
    path: HOST_PATH
  }
}

function makeUrlAttachment(label: string): ComposerAttachment {
  return {
    id: 'url:https://example.com',
    kind: 'url',
    label,
    refText: '@url:https://example.com'
  }
}

function makeFileAttachment(): ComposerAttachment {
  return {
    detail: 'C:\\Users\\alice\\Documents\\report.txt',
    id: 'file:report.txt',
    kind: 'file',
    label: 'report.txt',
    path: 'C:\\Users\\alice\\Documents\\report.txt',
    refText: '@file:`C:\\Users\\alice\\Documents\\report.txt`'
  }
}

function installDelegate(): void {
  const updateSession: SessionTileDelegate['updateSession'] = (runtimeId, updater) => {
    const current = $sessionStates.get()[runtimeId] ?? createClientSessionState(STORED_ID)
    const next = updater(current)
    $sessionStates.set({ ...$sessionStates.get(), [runtimeId]: next })

    return next
  }

  setSessionTileDelegate({
    archiveSession: vi.fn(async () => undefined),
    branchSession: vi.fn(async () => undefined),
    deleteSession: vi.fn(async () => undefined),
    executeSlash: vi.fn(async () => undefined),
    interruptSession: vi.fn(async () => undefined),
    resumeTile: vi.fn(async () => RUNTIME_ID),
    submitToSession: vi.fn(async () => undefined),
    updateSession
  })
}

function createScope(): ComposerScope {
  return {
    $awaitingInput: atom(false),
    $messages: atom([]),
    attachments: createComposerAttachmentScope(),
    target: `tile:${STORED_ID}`
  }
}

describe('session tile attachment occurrence ownership', () => {
  beforeEach(() => {
    requestGateway.mockReset()
    $connection.set({ mode: 'local' } as never)
    $sessions.set([])
    $sessionStates.set({
      [RUNTIME_ID]: {
        ...createClientSessionState(STORED_ID),
        cwd: '/root'
      }
    })
    installDelegate()
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl: vi.fn(async () => FULL_SOURCE) }
    })
  })

  afterEach(() => {
    Reflect.deleteProperty(window, 'hermesDesktop')
    $connection.set(null)
    $sessions.set([])
    $sessionStates.set({})
  })

  it('preserves a thumbnail that resolves before tile submit staging', async () => {
    const scope = createScope()
    const original = makeAttachment()
    const attach = deferred<{ attached: boolean; path: string }>()

    requestGateway.mockImplementation(async (method: string) => {
      if (method === 'image.attach_bytes') {
        return attach.promise
      }

      if (method === 'prompt.submit') {
        throw new Error('submit failed after staging')
      }

      return {}
    })

    scope.attachments.add(original)

    const { result } = renderHook(() =>
      useSessionTileActions({ runtimeId: RUNTIME_ID, scope, storedSessionId: STORED_ID })
    )

    let submitted!: Promise<boolean>
    act(() => {
      submitted = result.current.submitText('describe this')
    })

    await waitFor(() => expect(requestGateway).toHaveBeenCalledWith('image.attach_bytes', expect.anything()))
    expect(scope.attachments.updateIfCurrent(original, { thumbnailUrl: THUMBNAIL })).toBe(true)

    attach.resolve({ attached: true, path: STAGED_PATH })

    await act(async () => {
      await expect(submitted).resolves.toBe(false)
    })

    expect(scope.attachments.$attachments.get()[0]).toMatchObject({
      attachedSessionId: RUNTIME_ID,
      occurrenceId: original.occurrenceId,
      path: STAGED_PATH,
      thumbnailUrl: THUMBNAIL
    })
  })

  it('rejects stale tile staging after remove and reattach of the same path', async () => {
    const scope = createScope()
    const original = makeAttachment()
    const replacement = makeAttachment()
    const attach = deferred<{ attached: boolean; path: string }>()

    requestGateway.mockImplementation(async (method: string) => {
      if (method === 'image.attach_bytes') {
        return attach.promise
      }

      if (method === 'prompt.submit') {
        throw new Error('submit failed after staging')
      }

      return {}
    })

    scope.attachments.add(original)

    const { result } = renderHook(() =>
      useSessionTileActions({ runtimeId: RUNTIME_ID, scope, storedSessionId: STORED_ID })
    )

    let submitted!: Promise<boolean>
    act(() => {
      submitted = result.current.submitText('describe this')
    })

    await waitFor(() => expect(requestGateway).toHaveBeenCalledWith('image.attach_bytes', expect.anything()))
    scope.attachments.remove(original.id)
    scope.attachments.add(replacement)

    attach.resolve({ attached: true, path: STAGED_PATH })

    await act(async () => {
      await expect(submitted).resolves.toBe(false)
    })

    expect(scope.attachments.$attachments.get()).toEqual([replacement])
  })

  it('preserves a same-path replacement added while a successful tile submit is in flight', async () => {
    const scope = createScope()
    const original = makeAttachment()
    const replacement = makeAttachment()
    const attach = deferred<{ attached: boolean; path: string }>()

    requestGateway.mockImplementation(async (method: string) => {
      if (method === 'image.attach_bytes') {
        return attach.promise
      }

      return {}
    })

    scope.attachments.add(original)

    const { result } = renderHook(() =>
      useSessionTileActions({ runtimeId: RUNTIME_ID, scope, storedSessionId: STORED_ID })
    )

    let submitted!: Promise<boolean>
    act(() => {
      submitted = result.current.submitText('describe this')
    })

    await waitFor(() => expect(requestGateway).toHaveBeenCalledWith('image.attach_bytes', expect.anything()))
    scope.attachments.remove(original.id)
    scope.attachments.add(replacement)
    attach.resolve({ attached: true, path: STAGED_PATH })

    await act(async () => {
      await expect(submitted).resolves.toBe(true)
    })

    expect(scope.attachments.$attachments.get()).toEqual([replacement])
  })

  it('preserves a newer same-id URL added while a successful tile submit is in flight', async () => {
    const scope = createScope()
    const original = makeUrlAttachment('old')
    const replacement = makeUrlAttachment('new')
    const submit = deferred<Record<string, never>>()

    requestGateway.mockImplementation(async (method: string) => {
      if (method === 'prompt.submit') {
        return submit.promise
      }

      return {}
    })

    scope.attachments.add(original)

    const { result } = renderHook(() =>
      useSessionTileActions({ runtimeId: RUNTIME_ID, scope, storedSessionId: STORED_ID })
    )

    let submitted!: Promise<boolean>
    act(() => {
      submitted = result.current.submitText('read this')
    })

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith('prompt.submit', expect.anything(), expect.anything())
    )
    scope.attachments.remove(original.id)
    scope.attachments.add(replacement)
    submit.resolve({})

    await act(async () => {
      await expect(submitted).resolves.toBe(true)
    })

    expect(scope.attachments.$attachments.get()).toEqual([replacement])
  })

  it('removes a successfully staged legacy file from the tile composer', async () => {
    const scope = createScope()
    const original = makeFileAttachment()

    requestGateway.mockImplementation(async (method: string) => {
      if (method === 'file.attach') {
        return {
          attached: true,
          ref_text: '@file:.hermes/desktop-attachments/report.txt',
          uploaded: true
        }
      }

      return {}
    })

    scope.attachments.add(original)

    const { result } = renderHook(() =>
      useSessionTileActions({ runtimeId: RUNTIME_ID, scope, storedSessionId: STORED_ID })
    )

    await act(async () => {
      await expect(result.current.submitText('read this')).resolves.toBe(true)
    })

    expect(scope.attachments.$attachments.get()).toEqual([])
  })
})
