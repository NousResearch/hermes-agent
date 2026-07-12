import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { artifactActionAvailable, performArtifactAction } from './artifact-actions'

const openExternal = vi.fn(async () => {})
const readFileText = vi.fn(async () => ({ path: '/work/report.md', text: '# Entire report' }))
const revealPath = vi.fn(async () => true)
const writeClipboard = vi.fn(async () => true)
const api = vi.fn(async () => ({ path: '/remote/report.md', text: '# Remote report' }))

const normalizePreviewTarget = vi.fn(async () => ({
  kind: 'file' as const,
  label: 'report.md',
  path: '/work/report.md',
  previewKind: 'text' as const,
  source: '/work/report.md',
  url: 'file:///work/report.md'
}))

const target = {
  artifact: true,
  filesystemKey: 'local',
  kind: 'file' as const,
  label: 'report.md',
  path: '/work/report.md',
  previewKind: 'text' as const,
  source: '/work/report.md',
  url: 'file:///work/report.md'
}

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

describe('Markdown artifact actions', () => {
  beforeEach(() => {
    vi.stubGlobal('window', {
      hermesDesktop: { api, normalizePreviewTarget, openExternal, readFileText, revealPath, writeClipboard }
    })
    $connection.set({ mode: 'local' } as never)
  })

  afterEach(() => {
    $connection.set(null)
    vi.clearAllMocks()
    vi.unstubAllGlobals()
  })

  it('opens, reveals, and copies through existing hardened desktop interfaces', async () => {
    await performArtifactAction('open', target)
    await performArtifactAction('reveal', target)
    await performArtifactAction('copy-path', target)
    await performArtifactAction('copy-contents', target)

    expect(openExternal).toHaveBeenCalledWith('file:///work/report.md')
    expect(revealPath).toHaveBeenCalledWith('/work/report.md')
    expect(writeClipboard).toHaveBeenNthCalledWith(1, '/work/report.md')
    expect(readFileText).toHaveBeenCalledWith('/work/report.md', { complete: true })
    expect(writeClipboard).toHaveBeenNthCalledWith(2, '# Entire report')
  })

  it('resolves a relative artifact before copying its absolute path', async () => {
    await performArtifactAction('copy-path', {
      ...target,
      path: './report.md',
      source: './report.md',
      url: 'file:///work/report.md'
    })

    expect(readFileText).toHaveBeenCalledWith('./report.md')
    expect(writeClipboard).toHaveBeenCalledWith('/work/report.md')
  })

  it('disables local OS actions for gateway files but copies remote data through authenticated REST', async () => {
    $connection.set({ mode: 'remote' } as never)

    const remoteTarget = {
      ...target,
      filesystemKey: 'remote::',
      path: '/remote/report.md',
      url: 'file:///remote/report.md'
    }

    expect(artifactActionAvailable('open', remoteTarget)).toBe(false)
    expect(artifactActionAvailable('reveal', remoteTarget)).toBe(false)
    expect(artifactActionAvailable('copy-path', remoteTarget)).toBe(true)
    expect(artifactActionAvailable('copy-contents', remoteTarget)).toBe(true)
    await expect(performArtifactAction('open', remoteTarget)).rejects.toThrow('local file')
    await performArtifactAction('copy-contents', remoteTarget)

    expect(openExternal).not.toHaveBeenCalled()
    expect(api).toHaveBeenCalledWith({ path: '/api/fs/read-text?path=%2Fremote%2Freport.md&complete=true' })
    expect(writeClipboard).toHaveBeenCalledWith('# Remote report')
  })

  it('aborts stale OS and clipboard side effects when the filesystem changes while an action is pending', async () => {
    const pendingNormalize = deferred<Awaited<ReturnType<typeof normalizePreviewTarget>>>()
    normalizePreviewTarget.mockReturnValueOnce(pendingNormalize.promise)

    const open = performArtifactAction('open', target)
    $connection.set({ mode: 'remote' } as never)
    pendingNormalize.resolve({
      kind: 'file',
      label: 'report.md',
      path: '/work/report.md',
      previewKind: 'text',
      source: '/work/report.md',
      url: 'file:///work/report.md'
    })

    await expect(open).rejects.toThrow('different filesystem')
    expect(openExternal).not.toHaveBeenCalled()

    $connection.set({ mode: 'local' } as never)
    const pendingRead = deferred<Awaited<ReturnType<typeof readFileText>>>()
    readFileText.mockReturnValueOnce(pendingRead.promise)

    const copy = performArtifactAction('copy-contents', target)
    $connection.set({ mode: 'remote' } as never)
    pendingRead.resolve({ path: '/work/report.md', text: '# Stale report' })

    await expect(copy).rejects.toThrow('different filesystem')
    expect(writeClipboard).not.toHaveBeenCalled()
  })

  it('refuses actions after switching between local and remote filesystems or remote profiles', async () => {
    const remoteTarget = {
      ...target,
      filesystemKey: 'remote:remote-a:',
      path: '/srv/report.md',
      url: 'file:///srv/report.md'
    }

    $connection.set({ mode: 'remote', profile: 'remote-a' } as never)
    expect(artifactActionAvailable('copy-contents', remoteTarget)).toBe(true)

    $connection.set({ mode: 'local' } as never)
    expect(artifactActionAvailable('open', remoteTarget)).toBe(false)
    await expect(performArtifactAction('open', remoteTarget)).rejects.toThrow('different filesystem')

    $connection.set({ mode: 'remote', profile: 'remote-b' } as never)
    expect(artifactActionAvailable('copy-contents', remoteTarget)).toBe(false)
    await expect(performArtifactAction('copy-contents', remoteTarget)).rejects.toThrow('different filesystem')
    expect(api).not.toHaveBeenCalled()
    expect(openExternal).not.toHaveBeenCalled()
  })
})
