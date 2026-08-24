import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from '@/api/client'
import { $connection } from '@/store/session'

import {
  desktopDefaultCwd,
  desktopFileDiff,
  desktopFsCacheKey,
  desktopGitRoot,
  readDesktopDir,
  readDesktopFileDataUrl,
  readDesktopFileDataUrlLocalFirst,
  readDesktopFileText,
  selectDesktopPaths,
  setDesktopFsRemotePicker
} from './desktop-fs'

const readDir = vi.fn(async () => ({ entries: [{ name: 'local', path: '/local', isDirectory: true }] }))
const readFileText = vi.fn(async () => ({ path: '/local/file.txt', text: 'local', byteSize: 5 }))
const readFileDataUrl = vi.fn(async () => 'data:text/plain;base64,bG9jYWw=')
const gitRoot = vi.fn(async () => '/local')
const selectPaths = vi.fn(async () => ['/local'])

const api = vi.fn(async ({ path }: { path: string }) => {
  if (path.startsWith('/api/fs/list?')) {
    return { entries: [{ name: 'remote', path: '/remote', isDirectory: true }] }
  }

  if (path.startsWith('/api/fs/read-text?')) {
    return { path: '/remote/file.txt', text: 'remote', byteSize: 6 }
  }

  if (path.startsWith('/api/fs/read-data-url?')) {
    return { dataUrl: 'data:text/plain;base64,cmVtb3Rl' }
  }

  if (path.startsWith('/api/fs/git-root?')) {
    return { root: '/remote' }
  }

  if (path === '/api/fs/default-cwd') {
    return { cwd: '/backend/project', branch: 'main' }
  }

  if (path.startsWith('/api/git/file-diff?')) {
    return { diff: 'remote diff' }
  }

  throw new Error(`unexpected path ${path}`)
})

function stubBridge() {
  vi.stubGlobal('window', {
    hermesDesktop: {
      api,
      gitRoot,
      readDir,
      readFileDataUrl,
      readFileText,
      selectPaths
    }
  })
}

describe('desktop filesystem facade', () => {
  beforeEach(() => {
    stubBridge()
    $connection.set(null)
    setApiRequestConnection(null)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllMocks()
    $connection.set(null)
    setApiRequestConnection(null)
    setDesktopFsRemotePicker(null)
  })

  it('uses local Electron filesystem methods in local mode', async () => {
    $connection.set({ mode: 'local', profile: 'team-local' } as never)

    await expect(readDesktopDir('/work')).resolves.toEqual({
      entries: [{ name: 'local', path: '/local', isDirectory: true }]
    })
    await expect(readDesktopFileText('/work/file.txt')).resolves.toMatchObject({ text: 'local' })
    await expect(readDesktopFileDataUrl('/work/file.txt')).resolves.toBe('data:text/plain;base64,bG9jYWw=')
    await expect(desktopGitRoot('/work')).resolves.toBe('/local')
    await expect(selectDesktopPaths({ directories: true })).resolves.toEqual(['/local'])

    expect(readDir).toHaveBeenCalledWith('/work')
    expect(readFileText).toHaveBeenCalledWith('/work/file.txt')
    expect(readFileDataUrl).toHaveBeenCalledWith('/work/file.txt')
    expect(gitRoot).toHaveBeenCalledWith('/work')
    expect(selectPaths).toHaveBeenCalledWith({ directories: true, profile: 'team-local' })
    expect(api).not.toHaveBeenCalled()
  })

  it('routes filesystem reads through authenticated backend REST in remote mode', async () => {
    $connection.set({ mode: 'remote' } as never)

    await expect(readDesktopDir('/home/user/project')).resolves.toMatchObject({ entries: [{ name: 'remote' }] })
    await expect(readDesktopFileText('/home/user/project/a b.txt')).resolves.toMatchObject({ text: 'remote' })
    await expect(readDesktopFileDataUrl('/home/user/project/a b.txt')).resolves.toBe('data:text/plain;base64,cmVtb3Rl')
    await expect(desktopGitRoot('/home/user/project')).resolves.toBe('/remote')
    await expect(desktopDefaultCwd()).resolves.toEqual({ cwd: '/backend/project', branch: 'main' })

    expect(api).toHaveBeenCalledWith({ path: '/api/fs/list?path=%2Fhome%2Fuser%2Fproject' })
    expect(api).toHaveBeenCalledWith({ path: '/api/fs/read-text?path=%2Fhome%2Fuser%2Fproject%2Fa%20b.txt' })
    expect(api).toHaveBeenCalledWith({ path: '/api/fs/read-data-url?path=%2Fhome%2Fuser%2Fproject%2Fa%20b.txt' })
    expect(api).toHaveBeenCalledWith({ path: '/api/fs/git-root?path=%2Fhome%2Fuser%2Fproject' })
    expect(api).toHaveBeenCalledWith({ path: '/api/fs/default-cwd' })
    expect(readDir).not.toHaveBeenCalled()
    expect(readFileText).not.toHaveBeenCalled()
    expect(readFileDataUrl).not.toHaveBeenCalled()
    expect(gitRoot).not.toHaveBeenCalled()
  })

  it('does not retry the same unreadable path through the local facade', async () => {
    const error = new Error('not readable')

    $connection.set({ mode: 'local' } as never)
    readFileDataUrl.mockRejectedValueOnce(error)

    await expect(readDesktopFileDataUrlLocalFirst('/missing.png')).rejects.toBe(error)
    expect(readFileDataUrl).toHaveBeenCalledOnce()
    expect(api).not.toHaveBeenCalled()
  })

  it('falls back from local disk to the active gateway in remote mode', async () => {
    $connection.set({ mode: 'remote' } as never)
    readFileDataUrl.mockRejectedValueOnce(new Error('not on host'))

    await expect(readDesktopFileDataUrlLocalFirst('/remote/image.png')).resolves.toBe('data:text/plain;base64,cmVtb3Rl')
    expect(readFileDataUrl).toHaveBeenCalledOnce()
    expect(api).toHaveBeenCalledWith({ path: '/api/fs/read-data-url?path=%2Fremote%2Fimage.png' })
  })

  it('targets the active profile backend so a remote profile never reads local disk', async () => {
    $connection.set({ mode: 'remote', profile: 'remote-docker' } as never)

    await readDesktopDir('/srv/project')
    await desktopDefaultCwd()

    expect(api).toHaveBeenCalledWith({ path: '/api/fs/list?path=%2Fsrv%2Fproject', profile: 'remote-docker' })
    expect(api).toHaveBeenCalledWith({ path: '/api/fs/default-cwd', profile: 'remote-docker' })
  })

  it('pins SSH filesystem reads to the active registry connection', async () => {
    $connection.set({
      connectionId: 'work-ssh',
      mode: 'remote',
      profile: 'default',
      remoteKind: 'ssh'
    } as never)
    setApiRequestConnection('work-ssh')

    await readDesktopFileDataUrl('/srv/project/image.png')

    expect(api).toHaveBeenCalledWith({
      connectionId: 'work-ssh',
      path: '/api/fs/read-data-url?path=%2Fsrv%2Fproject%2Fimage.png',
      profile: 'default'
    })
  })

  it('keys SSH filesystem caches by stable host identity instead of the forwarded port', () => {
    $connection.set({
      mode: 'remote',
      remoteKind: 'ssh',
      remoteHost: 'operator@remote-box',
      baseUrl: 'http://127.0.0.1:41001'
    } as never)
    const first = desktopFsCacheKey()

    $connection.set({
      mode: 'remote',
      remoteKind: 'ssh',
      remoteHost: 'operator@remote-box',
      baseUrl: 'http://127.0.0.1:52002'
    } as never)

    expect(desktopFsCacheKey()).toBe(first)
    expect(first).toContain('operator@remote-box')
    expect(first).not.toContain('41001')
  })

  it('separates SSH filesystem caches by ownership and profile', () => {
    $connection.set({
      mode: 'remote',
      remoteKind: 'ssh',
      remoteHost: 'host-a',
      remoteIdentity: 'owner-a',
      profile: 'one'
    } as never)
    const first = desktopFsCacheKey()
    $connection.set({
      mode: 'remote',
      remoteKind: 'ssh',
      remoteHost: 'host-a',
      remoteIdentity: 'owner-b',
      profile: 'one'
    } as never)
    const otherOwner = desktopFsCacheKey()
    $connection.set({
      mode: 'remote',
      remoteKind: 'ssh',
      remoteHost: 'host-a',
      remoteIdentity: 'owner-a',
      profile: 'two'
    } as never)

    expect(otherOwner).not.toBe(first)
    expect(desktopFsCacheKey()).not.toBe(first)
  })

  it('routes file diffs through backend git in remote mode', async () => {
    $connection.set({ mode: 'remote' } as never)

    await expect(desktopFileDiff('/repo', 'src/a b.ts')).resolves.toBe('remote diff')
    expect(api).toHaveBeenCalledWith({ path: '/api/git/file-diff?path=%2Frepo&file=src%2Fa%20b.ts' })
  })

  it('uses the registered in-app directory picker in remote mode', async () => {
    const remoteSelect = vi.fn(async () => ['/remote/project'])
    $connection.set({ mode: 'remote' } as never)
    setDesktopFsRemotePicker({ selectPaths: remoteSelect })

    await expect(selectDesktopPaths({ defaultPath: '/remote', directories: true, multiple: false })).resolves.toEqual([
      '/remote/project'
    ])

    expect(remoteSelect).toHaveBeenCalledWith({ defaultPath: '/remote', directories: true, multiple: false })
    expect(selectPaths).not.toHaveBeenCalled()
  })

  it('uses the local Electron picker for remote file selection', async () => {
    const remoteSelect = vi.fn(async () => ['/remote/project'])
    $connection.set({ mode: 'remote', profile: 'team-remote' } as never)
    setDesktopFsRemotePicker({ selectPaths: remoteSelect })

    await expect(selectDesktopPaths({ directories: false, multiple: false })).resolves.toEqual(['/local'])

    expect(selectPaths).toHaveBeenCalledWith({ directories: false, multiple: false, profile: 'team-remote' })
    expect(remoteSelect).not.toHaveBeenCalled()
  })

  it('limits the remote picker to single-directory selection', async () => {
    const remoteSelect = vi.fn(async () => ['/remote/project'])
    $connection.set({ mode: 'remote' } as never)
    setDesktopFsRemotePicker({ selectPaths: remoteSelect })

    await expect(selectDesktopPaths({ directories: true })).resolves.toEqual(['/remote/project'])

    expect(remoteSelect).toHaveBeenCalledWith({ directories: true, multiple: false })
    expect(selectPaths).not.toHaveBeenCalled()
  })
})
