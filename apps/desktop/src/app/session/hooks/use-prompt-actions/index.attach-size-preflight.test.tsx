// A remote attach base64-loads each file into the renderer, so a send's byte uploads are sized
// with stat() first: an oversized file or batch must fail the send cleanly — no read, no
// file.attach, no prompt.submit — instead of aborting the renderer (#148787).
import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $composerAttachments, type ComposerAttachment } from '@/store/composer'
import { $connection, $currentCwd } from '@/store/session'

import { actRender, Harness, type HarnessHandle } from './index-test-utils'
import { clearSingleFlightSessionResumeState } from './single-flight-resume'

vi.mock('@/hermes', () => ({
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: 'session' })),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn()
}))

describe('usePromptActions attachment upload size preflight (stat before any byte read)', () => {
  beforeEach(() => {
    clearSingleFlightSessionResumeState()
    window.localStorage.removeItem('hermes.desktop.composerQueue.v1')
  })

  afterEach(() => {
    cleanup()
    $composerAttachments.set([])
    $connection.set(null)
    $currentCwd.set('')
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.restoreAllMocks()
  })

  const MiB = 1024 * 1024
  const limits = { maxFileBytes: 64 * MiB, maxTotalBytes: 128 * MiB }

  function installBridge(sizes: Record<string, number>) {
    const readFileDataUrl = vi.fn(async () => 'data:text/plain;base64,aGVsbG8=')
    const readFileDataUrlForAttach = vi.fn(async () => 'data:text/plain;base64,aGVsbG8=')

    const statFilesForAttach = vi.fn(async (paths: string[]) => ({
      limits,
      files: paths.map(path => ({ path, size: sizes[path] }))
    }))

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl, readFileDataUrlForAttach, statFilesForAttach }
    })

    return { readFileDataUrl, readFileDataUrlForAttach, statFilesForAttach }
  }

  function video(name: string): ComposerAttachment {
    const path = `/Users/alice/Downloads/${name}`

    return { id: `file:${name}`, kind: 'file', label: name, path, refText: `@file:\`${path}\`` }
  }

  async function submitWith(attachments: ComposerAttachment[]) {
    const calls: string[] = []

    const requestGateway = vi.fn(async (method: string) => {
      calls.push(method)

      return (
        method === 'file.attach' ? { attached: true, path: '/remote/x', ref_text: '@file:x', uploaded: true } : {}
      ) as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness onReady={h => (handle = h)} refreshSessions={async () => undefined} requestGateway={requestGateway} />
    )

    const ok = await handle!.submitText('here are the videos', { attachments })

    return { calls, ok }
  }

  it('refuses an oversized remote file without reading it and without submitting the turn', async () => {
    $connection.set({ mode: 'remote' } as never)
    const bridge = installBridge({ '/Users/alice/Downloads/course.mp4': 238 * 1000 * 1000 })

    const { calls, ok } = await submitWith([video('course.mp4')])

    expect(ok).toBe(false)
    expect(bridge.statFilesForAttach).toHaveBeenCalledTimes(1)
    expect(bridge.readFileDataUrlForAttach).not.toHaveBeenCalled()
    expect(bridge.readFileDataUrl).not.toHaveBeenCalled()
    expect(calls).not.toContain('file.attach')
    expect(calls).not.toContain('prompt.submit')
  })

  it('refuses a batch whose files each fit but whose total is over the per-message limit', async () => {
    $connection.set({ mode: 'remote' } as never)

    const bridge = installBridge({
      '/Users/alice/Downloads/a.mp4': 50 * MiB,
      '/Users/alice/Downloads/b.mp4': 50 * MiB,
      '/Users/alice/Downloads/c.mp4': 50 * MiB
    })

    const { calls, ok } = await submitWith([video('a.mp4'), video('b.mp4'), video('c.mp4')])

    expect(ok).toBe(false)
    expect(bridge.readFileDataUrlForAttach).not.toHaveBeenCalled()
    expect(calls).not.toContain('file.attach')
    expect(calls).not.toContain('prompt.submit')
  })

  it('still uploads and submits when the batch is within both limits', async () => {
    $connection.set({ mode: 'remote' } as never)

    const bridge = installBridge({
      '/Users/alice/Downloads/a.txt': 10 * MiB,
      '/Users/alice/Downloads/b.txt': 10 * MiB
    })

    const { calls, ok } = await submitWith([video('a.txt'), video('b.txt')])

    expect(ok).toBe(true)
    expect(bridge.readFileDataUrlForAttach).toHaveBeenCalledTimes(2)
    expect(calls).toEqual(['file.attach', 'file.attach', 'prompt.submit'])
  })

  it('does not size or refuse a large file that rides its path to a local backend', async () => {
    $connection.set({ mode: 'local' } as never)
    $currentCwd.set('/Users/alice')
    const bridge = installBridge({ '/Users/alice/Downloads/huge.log': 900 * MiB })

    const { calls, ok } = await submitWith([video('huge.log')])

    expect(ok).toBe(true)
    expect(bridge.statFilesForAttach).not.toHaveBeenCalled()
    expect(calls).toEqual(['file.attach', 'prompt.submit'])
  })
})
