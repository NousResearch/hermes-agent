import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { stageDroppedFilePath, useComposerActions } from '@/app/chat/hooks/use-composer-actions'
import { uploadComposerAttachment } from '@/app/session/hooks/use-prompt-actions'
import type { HermesStagedUpload } from '@/global'
import {
  $composerAttachments, clearSessionDraft, type ComposerAttachment, stashSessionDraft, takeSessionDraft
} from '@/store/composer'
import { $connection, $sessions } from '@/store/session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'

const source: HermesStagedUpload = {
  install_id: '11111111111111111111111111111111',
  path: '/srv/hermes/profiles/owner/uploads/web-123-notes.txt',
  profile_home: '/srv/hermes/profiles/owner',
  profile_incarnation: '22222222222222222222222222222222'
}

const sourceName = source.path.split('/').pop()!
const refText = '@file:/srv/hermes/profiles/owner/attachments/web-123-notes.txt'
const file = () => new File(['payload'], 'notes.txt', { type: 'text/plain' })

const rawAttachment = (path = source.path): ComposerAttachment => ({
  id: `file:${path}`, kind: 'file', label: path.split('/').pop() || path, path
})

const attached = () => ({ attached: true, ref_text: refText, uploaded: true })

async function installBrowser(withProvenance = true) {
  Reflect.deleteProperty(window, 'hermesDesktop')
  Object.assign(window, { __HERMES_SESSION_TOKEN__: 'test-token' })

  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), window.location.origin)

    if (url.pathname === '/api/chat/file-upload' && init?.method === 'POST') {
      return new Response(JSON.stringify({
        path: source.path,
        ...(withProvenance ? { staged_upload: source } : {})
      }))
    }

    if (url.pathname === '/api/fs/read-data-url') {
      return new Response(JSON.stringify({ dataUrl: 'data:text/plain;base64,cGF5bG9hZA==' }))
    }

    throw new Error(`Unexpected fetch: ${url}`)
  })

  vi.stubGlobal('fetch', fetchMock)
  expect(installBrowserDesktopBridge()).toBe(true)
  $connection.set(await window.hermesDesktop.getConnection('owner'))

  return fetchMock
}

afterEach(() => {
  cleanup()
  $composerAttachments.set([])
  clearSessionDraft('staged-attachment-contract')
  $sessions.set([])
  $connection.set(null)
  Reflect.deleteProperty(window, 'hermesDesktop')
  Reflect.deleteProperty(window, '__HERMES_SESSION_TOKEN__')
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('browser staged attachment transport', () => {
  it.each(['picker', 'drop', 'edit-message drop'] as const)(
    '%s sends provenance to the owning gateway without downloading or retransmitting bytes',
    async flow => {
      const fetchMock = await installBrowser()
      let attachment: ComposerAttachment

      if (flow === 'edit-message drop') {
        // This is the actual edit-composer chain; it bypasses the main chip store.
        const path = await stageDroppedFilePath({ file: file(), path: '' })
        attachment = rawAttachment(path)
      } else {
        const { result } = renderHook(() => useComposerActions({
          activeSessionId: null,
          currentCwd: '/workspace',
          requestGateway: vi.fn()
        }))

        if (flow === 'picker') {
          vi.spyOn(HTMLInputElement.prototype, 'click').mockImplementation(function (this: HTMLInputElement) {
            Object.defineProperty(this, 'files', { configurable: true, value: [file()] })
            this.dispatchEvent(new Event('change'))
          })
          await act(async () => { await result.current.pickContextPaths('file') })
        } else {
          await act(async () => { await result.current.attachDroppedItems([{ file: file(), path: '' }]) })
        }

        expect($composerAttachments.get()[0]?.stagedUpload).toEqual(source)
        // A restored draft must not rely on the browser bridge's transient map.
        stashSessionDraft('staged-attachment-contract', 'draft', $composerAttachments.get())
        attachment = takeSessionDraft('staged-attachment-contract').attachments[0]!
        window.hermesDesktop.getStagedFileForAttach = () => undefined
      }

      // The supplied requester can be an explicitly routed tile/edit owner.
      // Changing the foreground profile must not retarget the attachment.
      $connection.set({ ...$connection.get()!, profile: 'foreground-other' })
      const requestGateway = vi.fn(async () => attached() as never)

      const result = await uploadComposerAttachment(attachment, {
        backendCwd: '/workspace', remote: true, requestGateway,
        sessionId: 'owner-runtime', terminalBackend: 'docker'
      })

      expect(requestGateway).toHaveBeenCalledWith('file.attach', {
        name: attachment.label, session_id: 'owner-runtime', staged_upload: source
      })
      expect(result.refText).toBe(refText)
      expect(result.stagedUpload).toEqual(source)
      expect(fetchMock).toHaveBeenCalledTimes(1)
      expect(new URL(String(fetchMock.mock.calls[0][0])).pathname).toBe('/api/chat/file-upload')
    }
  )

  it('keeps source ownership for multiple occurrences of the same staged path', async () => {
    const fetchMock = await installBrowser()
    await stageDroppedFilePath({ file: file(), path: '' })
    const requestGateway = vi.fn(async () => attached() as never)

    for (const sessionId of ['first-composer', 'second-composer']) {
      const result = await uploadComposerAttachment(rawAttachment(), {
        remote: true, requestGateway, sessionId
      })

      expect(result.stagedUpload).toEqual(source)
    }

    expect(requestGateway).toHaveBeenCalledTimes(2)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('retains the legacy byte transport when an older upload endpoint supplies only a path', async () => {
    const fetchMock = await installBrowser(false)
    const path = await stageDroppedFilePath({ file: file(), path: '' })
    const requestGateway = vi.fn(async () => attached() as never)
    await uploadComposerAttachment(rawAttachment(path), {
      remote: true, requestGateway, sessionId: 'runtime'
    })
    expect(requestGateway).toHaveBeenCalledWith('file.attach', {
      name: sourceName, path, session_id: 'runtime', data_url: 'data:text/plain;base64,cGF5bG9hZA=='
    })
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('retries a failed attach from the same source without another upload or download', async () => {
    const fetchMock = await installBrowser()
    const path = await stageDroppedFilePath({ file: file(), path: '' })
    const attachment = { ...rawAttachment(path), stagedUpload: source }

    const requestGateway = vi.fn()
      .mockRejectedValueOnce(new Error('temporary write failure'))
      .mockResolvedValueOnce(attached())

    const opts = { remote: true, requestGateway, sessionId: 'runtime' }
    await expect(uploadComposerAttachment(attachment, opts)).rejects.toThrow('temporary write failure')
    expect(attachment.stagedUpload).toEqual(source)
    expect(attachment.attachedSessionId).toBeUndefined()
    await expect(uploadComposerAttachment(attachment, opts)).resolves.toMatchObject({ refText })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(requestGateway.mock.calls.every(([, params]) => params.staged_upload === source)).toBe(true)
  })

  it('keeps provenance through the existing session-not-found recovery', async () => {
    const fetchMock = await installBrowser()
    await stageDroppedFilePath({ file: file(), path: '' })
    $sessions.set([{ id: 'stored-upload-recovery', profile: 'owner' }] as never)
    let first = true

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') { return { session_id: 'new-runtime' } as never }

      if (first) { first = false; throw new Error('session not found') }

      return attached() as never
    })

    const recovered = vi.fn()

    const result = await uploadComposerAttachment(rawAttachment(), {
      remote: true, requestGateway, sessionId: 'old-runtime', storedSessionId: 'stored-upload-recovery',
      onSessionRecovered: recovered
    })

    expect(result.attachedSessionId).toBe('new-runtime')
    expect(recovered).toHaveBeenCalledWith('new-runtime')
    expect(requestGateway).toHaveBeenLastCalledWith('file.attach', {
      name: sourceName, session_id: 'new-runtime', staged_upload: source
    })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it.each(['another Hermes backend', 'Profile incarnation is stale'])(
    'does not replace an authoritative %s rejection with a byte-upload fallback',
    async message => {
      const fetchMock = await installBrowser()
      await stageDroppedFilePath({ file: file(), path: '' })
      const requestGateway = vi.fn().mockRejectedValue(new Error(message))
      await expect(uploadComposerAttachment(rawAttachment(), {
        remote: true, requestGateway, sessionId: 'runtime'
      })).rejects.toThrow(message)
      expect(requestGateway).toHaveBeenCalledTimes(1)
      expect(fetchMock).toHaveBeenCalledTimes(1)
    }
  )

  it.each([
    { remote: false, terminalBackend: 'local', upload: false },
    { remote: true, terminalBackend: 'local', upload: true },
    { remote: false, terminalBackend: 'docker', upload: true }
  ])('preserves native attachment transport: %j', async ({ remote, terminalBackend, upload }) => {
    const readFileDataUrl = vi.fn(async () => 'data:text/plain;base64,cGF5bG9hZA==')
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { readFileDataUrl } })
    const requestGateway = vi.fn(async () => attached() as never)
    const path = '/native/notes.txt'
    await uploadComposerAttachment(rawAttachment(path), {
      remote, terminalBackend, requestGateway, sessionId: 'runtime'
    })
    expect(readFileDataUrl).toHaveBeenCalledTimes(upload ? 1 : 0)
    expect(requestGateway).toHaveBeenCalledWith('file.attach', {
      name: 'notes.txt', path, session_id: 'runtime',
      ...(upload ? { data_url: 'data:text/plain;base64,cGF5bG9hZA==' } : {})
    })
  })
})
