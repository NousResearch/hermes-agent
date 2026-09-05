import fs from 'node:fs'
import path from 'node:path'

import { resolveLocalReadPath } from './wsl-path-bridge'

const WINDOWS_INVALID_CHARACTER_RE = /[<>:"|?*]/
const WINDOWS_RESERVED_NAME_RE = /^(con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])(?:\..*)?$/i

function hasControlCharacter(value: string): boolean {
  return [...value].some(character => {
    const code = character.charCodeAt(0)

    return code <= 0x1f || code === 0x7f
  })
}

export interface CreateDirectoryDeps {
  directoryExists: (value: string) => boolean
  expandUserPath: (value: string) => string
  mkdir?: (target: string) => Promise<unknown>
  platform?: NodeJS.Platform
  resolveLocalPath?: (value: string) => string
  resolveRequestedPathForIpc: (value: string, options: { purpose: string }) => string
}

export async function createDirectoryForIpc(
  parentPath: unknown,
  newName: unknown,
  {
    directoryExists,
    expandUserPath,
    mkdir = target => fs.promises.mkdir(target),
    platform = process.platform,
    resolveLocalPath = resolveLocalReadPath,
    resolveRequestedPathForIpc
  }: CreateDirectoryDeps
): Promise<{ path: string }> {
  const rawParent = String(parentPath || '').trim()
  const name = String(newName || '').trim()

  if (
    !rawParent ||
    !name ||
    name === '.' ||
    name === '..' ||
    name.includes('/') ||
    name.includes('\\') ||
    hasControlCharacter(name)
  ) {
    throw new Error('Invalid folder name')
  }

  const componentLength =
    platform === 'win32' || platform === 'darwin' ? name.length : Buffer.byteLength(name, 'utf8')

  if (componentLength > 255) {
    throw new Error('Folder name is too long')
  }

  if (
    platform === 'win32' &&
    (WINDOWS_INVALID_CHARACTER_RE.test(name) || WINDOWS_RESERVED_NAME_RE.test(name) || name.endsWith('.'))
  ) {
    throw new Error('Invalid folder name on Windows')
  }

  const expandedParent = expandUserPath(rawParent)
  const localParent = resolveLocalPath(expandedParent)
  const parent = resolveRequestedPathForIpc(localParent, { purpose: 'Create directory' })

  if (!directoryExists(parent)) {
    throw new Error('Parent directory does not exist')
  }

  const target = path.join(parent, name)

  if (fs.existsSync(target)) {
    throw new Error(`"${name}" already exists`)
  }

  await mkdir(target)

  return { path: target }
}
