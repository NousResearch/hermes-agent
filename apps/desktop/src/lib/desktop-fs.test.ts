import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import {
  createDesktopFolder,
  desktopDefaultCwd,
  desktopFileDiff,
  desktopGitRoot,
  moveDesktopPath,
  readDesktopDir,
  readDesktopFileDataUrl,
  readDesktopFileText,
  renameDesktopPath,
  selectDesktopPaths,
  setDesktopFsRemotePicker,
  trashDesktopPath,
  writeDesktopFileText
} from './desktop-fs'

const readDir = vi.fn(async () => ({ entries: [{ name: 'local', path: '/local', isDirectory: true }] }))
const readFileText = vi.fn(async () => ({ path: '/local/file.txt', text: 'local', byteSize: 5 }))
const readFileDataUrl = vi.fn(async () => 'data:text/plain;base64,bG9jYWw=')
const gitRoot = vi.fn(async () => '/local')
const selectPaths = vi.fn(async () => ['/local'])
const createDirectory = vi.fn(async (parent: string, name: string) => ({ path: `${parent}/${name}` }))
const movePath = vi.fn(async (_source: string, destination: string) => ({ path: `${destination}/moved.txt` }))
const renamePath = vi.fn(async (target: string, name: string) => ({ path: `${target}/../${name}` }))
const trashPath = vi.fn(async () => true)
const writeTextFile = vi.fn(async (target: string) => ({ path: target }))

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

  if (
    [
      '/api/fs/mkdir',
      '/api/fs/create-file',
      '/api/fs/rename',
      '/api/fs/move',
      '/api/fs/delete',
      '/api/fs/write-text'
    ].includes(path)
  ) {
    return { ok: true, path: '/remote/result' }
  }

  throw new Error(`unexpected path ${path}`)
})

function stubBridge() {
  vi.stubGlobal('window', {
    hermesDesktop: {
      api,
      createDirectory,
      gitRoot,
      movePath,
      readDir,
      readFileDataUrl,
      readFileText,
      renamePath,
      selectPaths,
      trashPath,
      writeTextFile
    }
  })
}

describe('desktop filesystem facade', () => {
  beforeEach(() => {
    stubBridge()
    $connection.set(null)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllMocks()
    $connection.set(null)
    setDesktopFsRemotePicker(null)
  })

  it('uses local Electron filesystem methods in local mode', async () => {
    $connection.set({ mode: 'local' } as never)

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
    expect(selectPaths).toHaveBeenCalledWith({ directories: true })
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

  it('targets the active profile backend so a remote profile never reads local disk', async () => {
    $connection.set({ mode: 'remote', profile: 'remote-docker' } as never)

    await readDesktopDir('/srv/project')
    await desktopDefaultCwd()

    expect(api).toHaveBeenCalledWith({ path: '/api/fs/list?path=%2Fsrv%2Fproject', profile: 'remote-docker' })
    expect(api).toHaveBeenCalledWith({ path: '/api/fs/default-cwd', profile: 'remote-docker' })
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
    $connection.set({ mode: 'remote' } as never)
    setDesktopFsRemotePicker({ selectPaths: remoteSelect })

    await expect(selectDesktopPaths({ directories: false, multiple: false })).resolves.toEqual(['/local'])

    expect(selectPaths).toHaveBeenCalledWith({ directories: false, multiple: false })
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

  it('routes local mutations through Electron without a shared busy gate', async () => {
    $connection.set({ mode: 'local' } as never)

    await expect(createDesktopFolder('/work', 'new')).resolves.toEqual({ path: '/work/new' })
    await expect(renameDesktopPath('/work/a.txt', 'b.txt')).resolves.toContain('b.txt')
    await expect(moveDesktopPath('/work/b.txt', '/target', '/work')).resolves.toBe('/target/moved.txt')
    await expect(trashDesktopPath('/work/b.txt', '/work')).resolves.toBeUndefined()
    await expect(writeDesktopFileText('/work/edit.txt', 'saved while running')).resolves.toEqual({
      path: '/work/edit.txt'
    })
  })

  it('routes remote mutations through authenticated fs endpoints with browser-root guards', async () => {
    $connection.set({ mode: 'remote', profile: 'prod' } as never)

    await createDesktopFolder('/srv/project', 'new')
    await renameDesktopPath('/srv/project/a.txt', 'b.txt')
    await moveDesktopPath('/srv/project/b.txt', '/srv/archive', '/srv/project')
    await trashDesktopPath('/srv/project/b.txt', '/srv/project')

    expect(api).toHaveBeenCalledWith({
      body: { name: 'new', parent: '/srv/project' },
      method: 'POST',
      path: '/api/fs/mkdir',
      profile: 'prod'
    })
    expect(api).toHaveBeenCalledWith({
      body: { browserRoot: '/srv/project', destination: '/srv/archive', source: '/srv/project/b.txt' },
      method: 'POST',
      path: '/api/fs/move',
      profile: 'prod'
    })
    expect(api).toHaveBeenCalledWith({
      body: { browserRoot: '/srv/project', path: '/srv/project/b.txt' },
      method: 'POST',
      path: '/api/fs/delete',
      profile: 'prod'
    })
  })
})
