import { createHash } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import nodePath from 'node:path'

import { BrowserWindow, app, ipcMain, shell } from 'electron'

const MAX_BYTES = 10 * 1024 * 1024
const MAX_LIST_ENTRIES = 200
const MAX_LIST_SCAN = 10_000
const ROOT_NAMES = ['desktop', 'documents', 'downloads'] as const

type RootName = (typeof ROOT_NAMES)[number]

export type WorkstationMetadataReceipt = {
  dev: number
  ino: number
  mode: number
  mtimeMs: number
  path: string
  size: number
}

export type WorkstationSourceReceipt = WorkstationMetadataReceipt & {
  sha256: string
}

export type WorkstationEntry = WorkstationMetadataReceipt & {
  isDirectory: boolean
  name: string
}

export type WorkstationFoldersResult = {
  ok: boolean
  error?: string
  roots?: Partial<Record<RootName, string>>
  entries?: WorkstationEntry[]
  entry?: WorkstationEntry
  totalEntries?: number
  nextOffset?: number
  truncated?: boolean
  contentBase64?: string
  source?: WorkstationSourceReceipt
  originalPath?: string
}

export type WorkstationFoldersHandler = (
  action: unknown,
  payload?: Record<string, unknown>
) => Promise<WorkstationFoldersResult>

type Dependencies = {
  fs?: typeof fs
  home?: string
  roots?: Partial<Record<RootName, string>>
  path?: typeof nodePath
  trashItem?: (targetPath: string) => Promise<void>
}

class WorkstationError extends Error {
  constructor(readonly code: string) {
    super(code)
  }
}

function sha256(content: Buffer): string {
  return createHash('sha256').update(content).digest('hex')
}

function sameReceipt(left: WorkstationSourceReceipt, right: WorkstationSourceReceipt): boolean {
  return sameMetadata(left, right) && left.sha256 === right.sha256
}

function sameMetadata(left: WorkstationMetadataReceipt, right: WorkstationMetadataReceipt): boolean {
  return (
    left.path === right.path &&
    left.dev === right.dev &&
    left.ino === right.ino &&
    left.mode === right.mode &&
    left.mtimeMs === right.mtimeMs &&
    left.size === right.size
  )
}

function metadataReceipt(value: unknown, expectedPath: unknown): WorkstationMetadataReceipt {
  const source = value as WorkstationMetadataReceipt | undefined

  if (
    !source ||
    source.path !== expectedPath ||
    !Number.isFinite(source.dev) ||
    !Number.isFinite(source.ino) ||
    !Number.isFinite(source.mode) ||
    !Number.isFinite(source.mtimeMs) ||
    !Number.isFinite(source.size)
  ) {
    throw new WorkstationError('invalid-source-receipt')
  }

  return source
}

function decodeBase64(value: unknown): Buffer {
  if (typeof value !== 'string') {
    throw new WorkstationError('invalid-content')
  }

  if (value.length > Math.ceil(MAX_BYTES / 3) * 4) {
    throw new WorkstationError('file-too-large')
  }

  const padding = value.endsWith('==') ? 2 : value.endsWith('=') ? 1 : 0
  const body = padding ? value.slice(0, -padding) : value

  if (
    value.length % 4 !== 0 ||
    body.search(/[^A-Za-z0-9+/]/) !== -1 ||
    (padding === 2 && body.length % 4 !== 2) ||
    (padding === 1 && body.length % 4 !== 3)
  ) {
    throw new WorkstationError('invalid-content')
  }

  const content = Buffer.from(value, 'base64')

  if (content.byteLength > MAX_BYTES) {
    throw new WorkstationError('file-too-large')
  }

  return content
}

function nodeError(error: unknown): string {
  if (error instanceof WorkstationError) {
    return error.code
  }

  if (error && typeof error === 'object' && 'code' in error && typeof error.code === 'string') {
    return error.code
  }

  return 'filesystem-error'
}

export function createWorkstationFoldersHandler(dependencies: Dependencies = {}): WorkstationFoldersHandler {
  const fsImpl = dependencies.fs ?? fs
  const path = dependencies.path ?? nodePath
  const home = dependencies.home ?? os.homedir()
  const trashItem = dependencies.trashItem ?? ((targetPath: string) => shell.trashItem(targetPath))

  async function roots(): Promise<Partial<Record<RootName, string>>> {
    const resolved = await Promise.allSettled(
      ROOT_NAMES.map(async name => {
        const configured =
          dependencies.roots?.[name] ?? path.join(home, `${name[0].toUpperCase()}${name.slice(1)}`)

        return [name, await fsImpl.promises.realpath(configured)] as const
      })
    )

    return Object.fromEntries(
      resolved.flatMap(result => (result.status === 'fulfilled' ? [result.value] : []))
    ) as Partial<Record<RootName, string>>
  }

  const contained = (root: string, target: string): boolean => {
    const relative = path.relative(root, target)

    return (
      relative === '' ||
      (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative))
    )
  }

  async function approvedPath(value: unknown, existing: boolean): Promise<string> {
    if (typeof value !== 'string' || !path.isAbsolute(value)) {
      throw new WorkstationError('invalid-path')
    }

    const target = path.resolve(value)
    const approvedRoots = Object.values(await roots())
    const root = approvedRoots.find(candidate => contained(candidate, target))

    if (!root) {
      throw new WorkstationError('unapproved-root')
    }

    // Walk every existing component without following symbolic links. The
    // canonical component check rejects links/reparse points that resolve
    // outside an approved root; Node's platform-specific reparse handling is
    // otherwise left to its lstat/realpath implementation.
    const relative = path.relative(root, existing ? target : path.dirname(target))
    let current = root

    for (const segment of relative.split(path.sep).filter(Boolean)) {
      current = path.join(current, segment)
      const stat = await fsImpl.promises.lstat(current)

      if (stat.isSymbolicLink()) {
        throw new WorkstationError('symlink-not-allowed')
      }

      if (!contained(root, await fsImpl.promises.realpath(current))) {
        throw new WorkstationError('unapproved-root')
      }
    }

    if (existing) {
      const stat = await fsImpl.promises.lstat(target)

      if (stat.isSymbolicLink()) {
        throw new WorkstationError('symlink-not-allowed')
      }

      if (!contained(root, await fsImpl.promises.realpath(target))) {
        throw new WorkstationError('unapproved-root')
      }
    }

    return target
  }

  async function readRegularFile(target: string): Promise<{
    content: Buffer
    source: WorkstationSourceReceipt
    mode: number
  }> {
    const resolved = await approvedPath(target, true)
    const handle = await fsImpl.promises.open(resolved, fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW ?? 0))

    try {
      const before = await handle.stat()

      if (!before.isFile()) {
        throw new WorkstationError('not-a-file')
      }

      if (before.size > MAX_BYTES) {
        throw new WorkstationError('file-too-large')
      }

      const content = await handle.readFile()
      const after = await handle.stat()

      if (
        before.dev !== after.dev ||
        before.ino !== after.ino ||
        before.mode !== after.mode ||
        before.mtimeMs !== after.mtimeMs ||
        before.size !== after.size
      ) {
        throw new WorkstationError('source-changed-during-stage')
      }

      return {
        content,
        source: {
          dev: after.dev,
          ino: after.ino,
          mode: after.mode,
          mtimeMs: after.mtimeMs,
          path: resolved,
          sha256: sha256(content),
          size: after.size
        },
        mode: after.mode
      }
    } finally {
      await handle.close()
    }
  }

  async function entryFor(target: string): Promise<WorkstationEntry> {
    const resolved = await approvedPath(target, true)
    const value = await fsImpl.promises.lstat(resolved)

    if (!value.isFile() && !value.isDirectory()) {
      throw new WorkstationError('unsupported-file-type')
    }

    return {
      dev: value.dev,
      ino: value.ino,
      isDirectory: value.isDirectory(),
      mode: value.mode,
      mtimeMs: value.mtimeMs,
      name: path.basename(resolved),
      path: resolved,
      size: value.size
    }
  }

  async function readDirectoryNames(target: string): Promise<string[]> {
    const directory = await fsImpl.promises.opendir(target)
    const names: string[] = []

    try {
      for await (const entry of directory) {
        if (names.length >= MAX_LIST_SCAN) {
          throw new WorkstationError('directory-too-large')
        }

        names.push(entry.name)
      }
    } finally {
      await directory.close().catch(() => undefined)
    }

    return names.sort((left, right) => left.localeCompare(right))
  }

  async function removeFile(filePath: string): Promise<void> {
    try {
      await fsImpl.promises.rm(filePath, { force: true })
    } catch {
      // Cleanup must not hide the operation's original error.
    }
  }

  async function execute(action: unknown, payload: Record<string, unknown> = {}): Promise<WorkstationFoldersResult> {
    try {
      if (action === 'roots') {
        return { ok: true, roots: await roots() }
      }

      if (action === 'stat') {
        return { ok: true, entry: await entryFor(String(payload.path ?? '')) }
      }

      if (action === 'list') {
        const target = await approvedPath(payload.path, true)
        await approvedPath(target, true)
        const stat = await fsImpl.promises.lstat(target)

        if (!stat.isDirectory()) {
          throw new WorkstationError('not-a-directory')
        }

        const allNames = await readDirectoryNames(target)
        const query = typeof payload.query === 'string' ? payload.query.trim() : ''

        if (query.length > 256) {
          throw new WorkstationError('invalid-query')
        }

        const exact = payload.exact === true
        const matchingNames = query
          ? allNames.filter(name => (exact ? name === query : name.includes(query)))
          : allNames
        const offset = Number.isInteger(payload.offset) && Number(payload.offset) >= 0 ? Number(payload.offset) : 0
        const limit =
          Number.isInteger(payload.limit) && Number(payload.limit) > 0 && Number(payload.limit) <= MAX_LIST_ENTRIES
            ? Number(payload.limit)
            : MAX_LIST_ENTRIES
        const pageNames = matchingNames.slice(offset, offset + limit)
        const entries = (
          await Promise.all(
            pageNames.map(async name => {
              const entryPath = path.join(target, name)
              const entry = await fsImpl.promises.lstat(entryPath)

              if (entry.isSymbolicLink()) {
                return null
              }

              return entryFor(entryPath)
            })
          )
        ).filter((entry): entry is WorkstationEntry => entry !== null)
        const nextOffset = offset + pageNames.length

        return {
          ok: true,
          entries,
          nextOffset,
          totalEntries: matchingNames.length,
          truncated: nextOffset < matchingNames.length
        }
      }

      if (action === 'read') {
        const file = await readRegularFile(String(payload.path ?? ''))

        return {
          ok: true,
          contentBase64: file.content.toString('base64'),
          source: file.source
        }
      }

      if (action === 'mkdir') {
        const target = await approvedPath(payload.path, false)
        const revalidated = await approvedPath(target, false)
        await fsImpl.promises.mkdir(revalidated, { mode: 0o700 })

        return { ok: true, entry: await entryFor(revalidated) }
      }

      if (action === 'create') {
        const target = await approvedPath(payload.path, false)
        const content = decodeBase64(payload.contentBase64)
        const temporary = path.join(
          path.dirname(target),
          `.${path.basename(target)}.${process.pid}.${Date.now()}.tmp`
        )
        let temporaryCreated = false
        let committed = false
        let handle: Awaited<ReturnType<typeof fs.promises.open>> | undefined

        try {
          const revalidatedTemporary = await approvedPath(temporary, false)
          handle = await fsImpl.promises.open(
            revalidatedTemporary,
            fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_WRONLY | (fs.constants.O_NOFOLLOW ?? 0),
            0o600
          )
          temporaryCreated = true
          await handle.writeFile(content)
          await handle.sync()
          await handle.close()
          handle = undefined

          const revalidatedTarget = await approvedPath(target, false)
          await fsImpl.promises.link(revalidatedTemporary, revalidatedTarget)
          committed = true
          await fsImpl.promises.unlink(revalidatedTemporary)
          temporaryCreated = false

          const createdFile = await readRegularFile(revalidatedTarget)

          return { ok: true, source: createdFile.source }
        } catch (error) {
          await handle?.close().catch(() => undefined)
          if (temporaryCreated) {
            await removeFile(temporary)
          }
          if (committed) {
            await removeFile(target)
          }
          throw error
        }
      }

      if (action === 'write') {
        const source = payload.source as WorkstationSourceReceipt | undefined

        if (
          !source ||
          typeof source.path !== 'string' ||
          !Number.isFinite(source.dev) ||
          !Number.isFinite(source.ino) ||
          !Number.isFinite(source.mode) ||
          !Number.isFinite(source.mtimeMs) ||
          typeof source.sha256 !== 'string' ||
          !Number.isFinite(source.size)
        ) {
          throw new WorkstationError('invalid-source-receipt')
        }

        const current = await readRegularFile(source.path)

        if (!sameReceipt(current.source, source)) {
          throw new WorkstationError('source-conflict')
        }

        const content = decodeBase64(payload.contentBase64)
        const temporary = path.join(
          path.dirname(current.source.path),
          `.${path.basename(current.source.path)}.${process.pid}.${Date.now()}.tmp`
        )
        let temporaryCreated = false
        let handle: Awaited<ReturnType<typeof fs.promises.open>> | undefined

        try {
          const revalidatedTemporary = await approvedPath(temporary, false)
          handle = await fsImpl.promises.open(
            revalidatedTemporary,
            fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_WRONLY | (fs.constants.O_NOFOLLOW ?? 0),
            current.mode & 0o777
          )
          temporaryCreated = true
          await handle.writeFile(content)
          await handle.sync()
          await handle.close()
          handle = undefined

          const latest = await readRegularFile(current.source.path)

          if (!sameReceipt(latest.source, source)) {
            throw new WorkstationError('source-conflict')
          }

          const revalidatedSource = await approvedPath(current.source.path, true)
          await fsImpl.promises.rename(revalidatedTemporary, revalidatedSource)
          temporaryCreated = false
        } catch (error) {
          await handle?.close().catch(() => undefined)
          if (temporaryCreated) {
            await removeFile(temporary)
          }
          throw error
        }

        const saved = await readRegularFile(current.source.path)

        return { ok: true, source: saved.source }
      }

      if (action === 'chmod' || action === 'utimes') {
        const source = metadataReceipt(payload.source, payload.path)
        const current = await entryFor(source.path)

        if (!sameMetadata(current, source)) {
          throw new WorkstationError('source-conflict')
        }

        const revalidatedPath = await approvedPath(current.path, true)
        const latest = await entryFor(revalidatedPath)

        if (!sameMetadata(latest, source)) {
          throw new WorkstationError('source-conflict')
        }

        if (action === 'chmod') {
          if (!Number.isInteger(payload.mode)) {
            throw new WorkstationError('invalid-mode')
          }

          await fsImpl.promises.chmod(revalidatedPath, Number(payload.mode) & 0o777)
        } else {
          if (!Number.isFinite(payload.atimeMs) || !Number.isFinite(payload.mtimeMs)) {
            throw new WorkstationError('invalid-times')
          }

          await fsImpl.promises.utimes(
            revalidatedPath,
            Number(payload.atimeMs) / 1000,
            Number(payload.mtimeMs) / 1000
          )
        }

        return { ok: true, entry: await entryFor(revalidatedPath) }
      }

      if (action === 'rename') {
        const source = metadataReceipt(payload.source, payload.sourcePath)
        const current = await entryFor(source.path)

        if (!sameMetadata(current, source)) {
          throw new WorkstationError('source-conflict')
        }

        if (Object.values(await roots()).some(root => path.relative(root, current.path) === '')) {
          throw new WorkstationError('root-mutation-forbidden')
        }

        const destination = await approvedPath(payload.destinationPath, false)
        const revalidatedSource = await approvedPath(current.path, true)
        const latest = await entryFor(revalidatedSource)

        if (!sameMetadata(latest, source)) {
          throw new WorkstationError('source-conflict')
        }

        const revalidatedDestination = await approvedPath(destination, false)

        try {
          await fsImpl.promises.lstat(revalidatedDestination)
          throw new WorkstationError('destination-exists')
        } catch (error) {
          if (!(error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT')) {
            throw error
          }
        }

        await fsImpl.promises.rename(revalidatedSource, revalidatedDestination)

        return { ok: true, entry: await entryFor(revalidatedDestination) }
      }

      if (action === 'trash') {
        const target = await approvedPath(payload.path, true)
        const approvedRoots = await roots()

        if (Object.values(approvedRoots).some(root => path.relative(root, target) === '')) {
          throw new WorkstationError('root-mutation-forbidden')
        }

        const revalidatedTarget = await approvedPath(target, true)
        await trashItem(revalidatedTarget)

        return { ok: true, originalPath: revalidatedTarget }
      }

      throw new WorkstationError('unsupported-action')
    } catch (error) {
      return { ok: false, error: nodeError(error) }
    }
  }

  // ponytail: one process-wide queue keeps plugin operations from racing their
  // own validation; per-account locks would only matter if this grows beyond
  // one local workstation capability.
  let serialized = Promise.resolve()
  const handler: WorkstationFoldersHandler = (action, payload = {}) => {
    const result = serialized.then(() => execute(action, payload))
    serialized = result.then(
      () => undefined,
      () => undefined
    )

    return result
  }

  return handler
}

export function registerWorkstationFoldersIpc(): void {
  const handler = createWorkstationFoldersHandler({
    roots: {
      desktop: app.getPath('desktop'),
      documents: app.getPath('documents'),
      downloads: app.getPath('downloads')
    }
  })

  ipcMain.handle('hermes:workstation-folders', (event, action, payload) => {
    try {
      const window = BrowserWindow.fromWebContents(event.sender)

      if (
        !window ||
        window.isDestroyed() ||
        window.webContents !== event.sender ||
        event.senderFrame !== event.sender.mainFrame
      ) {
        return { ok: false, error: 'unauthorized-sender' }
      }
    } catch {
      return { ok: false, error: 'unauthorized-sender' }
    }

    return handler(action, payload)
  })
}
