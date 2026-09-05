import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { createDirectoryForIpc } from './fs-create-directory'

let root = ''

beforeEach(async () => {
  root = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hermes-create-directory-'))
})

afterEach(async () => {
  await fs.promises.rm(root, { force: true, recursive: true })
})

function create(parentPath: unknown, name: unknown, platform: NodeJS.Platform = process.platform) {
  return createDirectoryForIpc(parentPath, name, {
    directoryExists: value => fs.existsSync(value) && fs.statSync(value).isDirectory(),
    expandUserPath: value => value,
    platform,
    resolveRequestedPathForIpc: value => path.resolve(value)
  })
}

describe('createDirectoryForIpc', () => {
  it('creates one directory inside an existing parent', async () => {
    const result = await create(root, 'notes')

    expect(result).toEqual({ path: path.join(root, 'notes') })
    await expect(fs.promises.stat(result.path)).resolves.toMatchObject({})
    expect((await fs.promises.stat(result.path)).isDirectory()).toBe(true)
  })

  it.each(['', '.', '..', 'nested/folder', 'nested\\folder'])('rejects invalid folder name %j', async name => {
    await expect(create(root, name)).rejects.toThrow('Invalid folder name')
  })

  it.each(['CON', 'nul.txt', 'bad:name', 'trailing.'])('rejects Windows-invalid folder name %j', async name => {
    await expect(create(root, name, 'win32')).rejects.toThrow('Invalid folder name on Windows')
  })

  it.each(['COM¹', 'LPT³'])('rejects additional Windows-reserved folder name %j', async name => {
    await expect(create(root, name, 'win32')).rejects.toThrow('Invalid folder name on Windows')
  })

  it('rejects control characters and names beyond one filesystem segment', async () => {
    await expect(create(root, 'bad\u0001name')).rejects.toThrow('Invalid folder name')
    await expect(create(root, 'a'.repeat(256))).rejects.toThrow('Folder name is too long')
  })

  it.each(['win32', 'darwin'] as const)(
    'accepts a valid multibyte %s name longer than 255 UTF-8 bytes',
    async platform => {
      const name = '界'.repeat(100)

      const result = await createDirectoryForIpc(root, name, {
        directoryExists: () => true,
        expandUserPath: (value: string) => value,
        mkdir: async () => undefined,
        platform,
        resolveRequestedPathForIpc: (value: string) => path.resolve(value)
      })

      expect(result).toEqual({ path: path.join(root, name) })
    }
  )

  it('rejects a missing parent directory', async () => {
    await expect(create(path.join(root, 'missing'), 'notes')).rejects.toThrow('Parent directory does not exist')
  })

  it('does not overwrite an existing path', async () => {
    await fs.promises.mkdir(path.join(root, 'notes'))

    await expect(create(root, 'notes')).rejects.toThrow('"notes" already exists')
  })

  it('hardens the parent path before creating the directory', async () => {
    const calls: string[] = []

    await createDirectoryForIpc(root, 'notes', {
      directoryExists: () => true,
      expandUserPath: value => value,
      resolveRequestedPathForIpc: value => {
        calls.push(value)

        return path.resolve(value)
      }
    })

    expect(calls).toEqual([root])
  })

  it('bridges a WSL parent path before hardening it', async () => {
    const calls: string[] = []

    const result = await createDirectoryForIpc('/home/alex/repo', 'notes', {
      directoryExists: () => true,
      expandUserPath: (value: string) => value,
      resolveLocalPath: (value: string) => {
        calls.push(`bridge:${value}`)

        return root
      },
      resolveRequestedPathForIpc: (value: string) => {
        calls.push(`harden:${value}`)

        return path.resolve(value)
      }
    })

    expect(calls).toEqual(['bridge:/home/alex/repo', `harden:${root}`])
    expect(result).toEqual({ path: path.join(root, 'notes') })
  })
})
