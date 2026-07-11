import fs from 'node:fs'
import path from 'node:path'

import { resolveRequestedPathForIpc, sensitiveFileBlockReason } from './hardening'

function validName(raw: unknown): string {
  const name = String(raw || '').trim()

  if (!name || name === '.' || name === '..' || name.includes('\0') || name.includes('/') || name.includes('\\')) {
    throw new Error('Invalid name')
  }

  return name
}

function resolved(raw: unknown, purpose: string): string {
  return resolveRequestedPathForIpc(String(raw || '').trim(), { purpose })
}

function guardSensitive(target: string): void {
  if (sensitiveFileBlockReason(target)) {
    throw new Error('Sensitive paths cannot be mutated')
  }
}

function guardRoot(target: string, browserRoot?: unknown): void {
  if (path.parse(target).root === target) {
    throw new Error('Cannot mutate filesystem root')
  }

  if (browserRoot && target === resolved(browserRoot, 'Browser root')) {
    throw new Error('Cannot mutate browser root')
  }
}

async function statMutable(target: string): Promise<fs.Stats> {
  const stat = await fs.promises.lstat(target)

  if (stat.isSymbolicLink() || (!stat.isFile() && !stat.isDirectory())) {
    throw new Error('Symlinks and special files cannot be mutated')
  }

  return stat
}

async function collision(target: string): Promise<boolean> {
  try {
    await fs.promises.lstat(target)

    return true
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return false
    }
    throw error
  }
}

export async function createDirectoryForIpc(parentPath: unknown, rawName: unknown): Promise<{ path: string }> {
  const parent = resolved(parentPath, 'Create folder')
  const name = validName(rawName)
  const parentStat = await statMutable(parent)

  if (!parentStat.isDirectory()) {
    throw new Error('Parent is not a directory')
  }
  const target = resolved(path.join(parent, name), 'Create folder')
  guardSensitive(target)

  if (await collision(target)) {
    throw new Error(`"${name}" already exists`)
  }
  await fs.promises.mkdir(target)

  return { path: target }
}

export async function createFileForIpc(parentPath: unknown, rawName: unknown): Promise<{ path: string }> {
  const parent = resolved(parentPath, 'Create file')
  const name = validName(rawName)
  const parentStat = await statMutable(parent)

  if (!parentStat.isDirectory()) {
    throw new Error('Parent is not a directory')
  }
  const target = resolved(path.join(parent, name), 'Create file')
  guardSensitive(target)
  const handle = await fs.promises.open(target, 'wx')
  await handle.close()

  return { path: target }
}

export async function renamePathForIpc(targetPath: unknown, rawName: unknown): Promise<{ path: string }> {
  const source = resolved(targetPath, 'Rename path')
  const name = validName(rawName)
  guardRoot(source)
  guardSensitive(source)
  await statMutable(source)
  const target = resolved(path.join(path.dirname(source), name), 'Rename path')
  guardSensitive(target)

  if (target === source) {
    return { path: target }
  }

  if (await collision(target)) {
    throw new Error(`"${name}" already exists`)
  }
  await fs.promises.rename(source, target)

  return { path: target }
}

export async function movePathForIpc(
  sourcePath: unknown,
  destinationPath: unknown,
  browserRoot?: unknown
): Promise<{ path: string }> {
  const source = resolved(sourcePath, 'Move path')
  const destination = resolved(destinationPath, 'Move destination')
  guardRoot(source, browserRoot)
  guardSensitive(source)
  const sourceStat = await statMutable(source)
  const destinationStat = await statMutable(destination)

  if (!destinationStat.isDirectory()) {
    throw new Error('Destination is not a directory')
  }

  if (sourceStat.isDirectory() && (destination === source || destination.startsWith(`${source}${path.sep}`))) {
    throw new Error('Cannot move a directory into itself or a descendant')
  }

  const target = resolved(path.join(destination, path.basename(source)), 'Move path')
  guardSensitive(target)

  if (await collision(target)) {
    throw new Error(`"${path.basename(source)}" already exists`)
  }
  await fs.promises.rename(source, target)

  return { path: target }
}

export async function deletePathForIpc(targetPath: unknown, browserRoot?: unknown): Promise<string> {
  const target = resolved(targetPath, 'Delete path')
  guardRoot(target, browserRoot)
  guardSensitive(target)
  await statMutable(target)

  return target
}
