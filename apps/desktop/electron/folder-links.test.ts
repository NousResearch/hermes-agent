import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

import { assertFolderPath, openExistingDirectory } from './folder-links'
import { WindowConnectionRouteRegistry } from './window-connection-route'

const roots: string[] = []
afterEach(async () => {
  await Promise.all(roots.splice(0).map(root => fs.promises.rm(root, { recursive: true, force: true })))
})

async function fixture() {
  const root = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hermes-folder-links-'))
  roots.push(root)
  const routes = new WindowConnectionRouteRegistry()
  const owner = { connectionId: 'local', profile: 'default' }
  routes.set(7, { ...owner, registryScoped: true })

  const deps = {
    getRoute: (id: number) => routes.get(id),
    ensureBackend: vi.fn(async (_id: number) => ({ mode: 'local' })),
    openPath: vi.fn(async (_path: string) => ''),
    fs
  }

  return { root, owner, routes, deps }
}

describe('folder path syntax independent of the host OS', () => {
  it('validates Windows syntax, reserved names and ambiguous endings on every host', () => {
    for (const target of ['C:/Reports', String.raw`C:\Reports\Проект #1 50% (final)`, 'C:/CONsole/COM10/NULs']) {
      expect(() => assertFolderPath(target, 'win32'), target).not.toThrow()
    }

    for (const target of [
      'C:/Reports:stream',
      'C:/dir/NUL',
      'C:/dir/nul.txt',
      'C:/CON',
      'C:/PRN',
      'C:/AUX',
      'C:/COM1',
      'C:/COM9.log',
      'C:/LPT1',
      'C:/LPT9',
      'C:/COM¹',
      'C:/LPT²',
      'C:/dir./child',
      'C:/dir /child',
      'C:/bad?name',
      'C:/bad*name',
      'C:/bad|name'
    ]) {
      expect(() => assertFolderPath(target, 'win32'), target).toThrow('Unsupported Windows folder path.')
    }

    expect(() => assertFolderPath('/tmp/reports', 'win32')).toThrow('Folder path does not match this computer.')
  })

  it.each(['linux', 'darwin'] as const)('uses POSIX syntax on %s without applying Windows reserved names', platform => {
    for (const target of ['/tmp/Reports', '/tmp/NUL', '/tmp/Reports:stream', '/tmp/Проект #1 50% (final)']) {
      expect(() => assertFolderPath(target, platform), target).not.toThrow()
    }

    for (const target of ['C:/Reports', '/tmp\\Reports']) {
      expect(() => assertFolderPath(target, platform), target).toThrow('Folder path does not match this computer.')
    }
  })
})

describe('openExistingDirectory', () => {
  it('never creates missing paths or opens files, bundles, or symlinks to them', async () => {
    const { root, owner, deps } = await fixture()
    const file = path.join(root, 'program.exe')
    const bundle = path.join(root, 'program.app')
    const alias = path.join(root, 'alias')
    await fs.promises.writeFile(file, 'not executable')
    await fs.promises.mkdir(bundle)
    await fs.promises.symlink(bundle, alias, process.platform === 'win32' ? 'junction' : 'dir')
    const missing = path.join(root, 'missing')

    for (const target of [missing, file, bundle, alias]) {
      expect((await openExistingDirectory(7, { path: target, owner }, deps)).ok, target).toBe(false)
    }

    expect(fs.existsSync(missing)).toBe(false)
    expect(deps.openPath).not.toHaveBeenCalled()
    // A path can change between stat and realpath; validate the resolved type too.
    const realpath = vi.spyOn(fs.promises, 'realpath').mockResolvedValueOnce(file)

    try {
      expect((await openExistingDirectory(7, { path: root, owner }, deps)).ok).toBe(false)
      expect(deps.openPath).not.toHaveBeenCalled()
    } finally {
      realpath.mockRestore()
    }
  })

  it('rejects a missing path with trailing U+00A0 instead of opening its existing sibling', async () => {
    const { root, owner, deps } = await fixture()
    const existing = path.join(root, 'Reports')
    const missing = `${existing}\u00a0`
    await fs.promises.mkdir(existing)
    expect(fs.existsSync(missing)).toBe(false)

    const result = await openExistingDirectory(7, { path: missing, owner }, deps)

    expect(deps.openPath).not.toHaveBeenCalled()
    expect(result.ok).toBe(false)
    expect(fs.existsSync(missing)).toBe(false)
    expect(await fs.promises.readdir(root)).toEqual(['Reports'])
  })

  it('rejects unsupported syntax before any filesystem access', async () => {
    const { owner, deps } = await fixture()
    const stat = vi.spyOn(fs.promises, 'stat')
    const realpath = vi.spyOn(fs.promises, 'realpath')

    try {
      for (const target of [
        '',
        'relative',
        '~',
        '~/Reports',
        'file:///tmp',
        '//server/share',
        '\\\\server\\share',
        '\\\\?\\C:\\Reports',
        '\\\\.\\C:',
        'C:Reports',
        '/tmp/\0bad',
        ...(process.platform === 'win32'
          ? ['/tmp', '\\Reports', 'C:/Reports:stream', 'C:/dir/NUL', 'C:/dir./child']
          : ['C:/Reports', 'C:\\Reports', '/tmp\\Reports'])
      ]) {
        expect((await openExistingDirectory(7, { path: target, owner }, deps)).ok, target).toBe(false)
      }

      expect(stat).not.toHaveBeenCalled()
      expect(realpath).not.toHaveBeenCalled()
      expect(deps.openPath).not.toHaveBeenCalled()
    } finally {
      vi.restoreAllMocks()
    }
  })

  it('rejects remote, unknown and mismatched owners before filesystem access', async () => {
    const { root, owner, routes, deps } = await fixture()
    const stat = vi.spyOn(fs.promises, 'stat')
    const realpath = vi.spyOn(fs.promises, 'realpath')

    try {
      for (const mode of ['remote', undefined]) {
        deps.ensureBackend.mockResolvedValueOnce({ mode } as { mode: string })
        expect((await openExistingDirectory(7, { path: root, owner }, deps)).ok).toBe(false)
      }

      for (const invalid of [
        undefined,
        { ...owner, connectionId: 'other' },
        { ...owner, profile: 'other' },
        { profile: 'default' }
      ]) {
        expect((await openExistingDirectory(7, { path: root, owner: invalid }, deps)).ok).toBe(false)
      }

      routes.delete(7)
      expect((await openExistingDirectory(7, { path: root, owner }, deps)).ok).toBe(false)
      expect((await openExistingDirectory(8, { path: root, owner }, deps)).ok).toBe(false)
      expect(stat).not.toHaveBeenCalled()
      expect(realpath).not.toHaveBeenCalled()
      expect(deps.openPath).not.toHaveBeenCalled()
    } finally {
      vi.restoreAllMocks()
    }
  })

  it('cancels route or backend switches during async validation, including away-and-back', async () => {
    const { root, owner, routes, deps } = await fixture()
    deps.ensureBackend.mockImplementationOnce(async () => {
      routes.set(7, { ...owner, profile: 'other', registryScoped: true })
      routes.set(7, { ...owner, registryScoped: true })

      return { mode: 'local' }
    })
    expect((await openExistingDirectory(7, { path: root, owner }, deps)).ok).toBe(false)
    const realpath = fs.promises.realpath.bind(fs.promises)

    const spy = vi.spyOn(fs.promises, 'realpath').mockImplementationOnce(async target => {
      const result = await realpath(target)
      routes.delete(7)

      return result
    })

    try {
      expect((await openExistingDirectory(7, { path: root, owner }, deps)).ok).toBe(false)
    } finally {
      spy.mockRestore()
    }

    routes.set(7, { ...owner, registryScoped: true })
    deps.ensureBackend.mockResolvedValueOnce({ mode: 'local' }).mockResolvedValueOnce({ mode: 'remote' })
    expect((await openExistingDirectory(7, { path: root, owner }, deps)).ok).toBe(false)
    expect(deps.openPath).not.toHaveBeenCalled()
  })

  it('opens only an existing directory, using its canonical path and surfacing shell failure', async () => {
    const { root, owner, deps } = await fixture()
    const dir = path.join(root, 'Проект #1 50% (final)')
    await fs.promises.mkdir(dir)
    expect(await openExistingDirectory(7, { path: dir, owner }, deps)).toEqual({ ok: true })
    expect(deps.openPath).toHaveBeenCalledWith(await fs.promises.realpath(dir))
    deps.openPath.mockResolvedValueOnce('file manager unavailable')
    expect(await openExistingDirectory(7, { path: dir, owner }, deps)).toEqual({
      ok: false,
      error: 'file manager unavailable'
    })
  })
})
