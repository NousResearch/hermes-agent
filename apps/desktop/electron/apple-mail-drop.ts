import { execFile } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)
const MAX_MAIL_SOURCE_BYTES = 50 * 1024 * 1024
const APPLE_MAIL_SCRIPT_TIMEOUT_MS = 120_000

const READ_SELECTED_MAIL_SOURCE_SCRIPT = String.raw`
on run argv
  if (count of argv) is not 1 then error "Apple Mail message ID is missing"
  set wantedId to item 1 of argv

  tell application id "com.apple.mail"
    set picked to selection

    repeat with currentMessage in picked
      set actualId to message id of currentMessage

      if actualId is wantedId or actualId is ("<" & wantedId & ">") then
        return source of currentMessage
      end if
    end repeat
  end tell

  error "The dragged Apple Mail message is no longer selected"
end run
`

async function managedExportDir(userDataDir: string, create: boolean): Promise<string | null> {
  const root = await fs.promises.realpath(userDataDir)
  const dir = path.join(root, 'apple-mail-drops')
  let stat: fs.Stats

  try {
    stat = await fs.promises.lstat(dir)
  } catch (err) {
    if ((err as NodeJS.ErrnoException)?.code !== 'ENOENT') {
      throw err
    }

    if (!create) {
      return null
    }

    try {
      await fs.promises.mkdir(dir, { mode: 0o700 })
    } catch (mkdirErr) {
      if ((mkdirErr as NodeJS.ErrnoException)?.code !== 'EEXIST') {
        throw mkdirErr
      }
    }

    stat = await fs.promises.lstat(dir)
  }

  if (stat.isSymbolicLink()) {
    throw new Error('The managed Apple Mail export directory must not be a symlink')
  }

  if (!stat.isDirectory()) {
    throw new Error('The managed Apple Mail export path is not a directory')
  }

  const realDir = await fs.promises.realpath(dir)

  if (realDir !== dir) {
    throw new Error('The managed Apple Mail export directory resolved outside its expected path')
  }

  await fs.promises.chmod(dir, 0o700)

  return dir
}

export function decodeAppleMailMessageUri(messageUri: string): string | null {
  if (typeof messageUri !== 'string' || !/^message:/i.test(messageUri) || messageUri.length > 4096) {
    return null
  }

  try {
    const payload = messageUri.slice(messageUri.indexOf(':') + 1).replace(/^\/\//, '')
    const decoded = decodeURIComponent(payload).trim()
    const messageId = decoded.startsWith('<') && decoded.endsWith('>') ? decoded.slice(1, -1) : decoded

    if (!messageId || messageId.length > 998 || !/^[^\s<>@]+@[^\s<>@]+$/.test(messageId)) {
      return null
    }

    return messageId
  } catch {
    return null
  }
}

export function isTrustedDesktopRendererUrl(
  actualUrl: string,
  devServer: string | undefined,
  packagedIndexUrl: string
): boolean {
  try {
    const actual = new URL(actualUrl)
    const expected = new URL(devServer || packagedIndexUrl)

    if (devServer) {
      return actual.origin === expected.origin && actual.pathname === expected.pathname
    }

    return (
      actual.protocol === 'file:' &&
      actual.host === expected.host &&
      actual.username === expected.username &&
      actual.password === expected.password &&
      actual.pathname === expected.pathname
    )
  } catch {
    return false
  }
}

export class AppleMailDropCapabilityRegistry {
  private readonly entries = new Map<string, { expiresAt: number; messageId: string }>()

  constructor(private readonly ttlMs = 15_000) {}

  get size(): number {
    return this.entries.size
  }

  private prune(now: number): void {
    for (const [key, entry] of this.entries) {
      if (entry.expiresAt < now) {
        this.entries.delete(key)
      }
    }
  }

  register(senderId: number, messageUri: string, token: string, now = Date.now()): boolean {
    this.prune(now)
    const messageId = decodeAppleMailMessageUri(messageUri)

    if (!Number.isInteger(senderId) || senderId <= 0 || !messageId) {
      return false
    }

    if (!token || token.length > 256 || !/^[A-Za-z0-9._-]+$/.test(token)) {
      return false
    }

    this.entries.set(`${senderId}\u0000${token}`, { expiresAt: now + this.ttlMs, messageId })

    return true
  }

  consume(senderId: number, messageUri: string, token: string, now = Date.now()): boolean {
    const key = `${senderId}\u0000${token}`
    const entry = this.entries.get(key)
    const messageId = decodeAppleMailMessageUri(messageUri)
    this.entries.delete(key)

    return Boolean(entry && messageId && entry.expiresAt >= now && entry.messageId === messageId)
  }

  clearSender(senderId: number): void {
    const prefix = `${senderId}\u0000`

    for (const key of this.entries.keys()) {
      if (key.startsWith(prefix)) {
        this.entries.delete(key)
      }
    }
  }
}

export function appleMailOsaScriptArgs(messageId: string): string[] {
  return ['-l', 'AppleScript', '-e', READ_SELECTED_MAIL_SOURCE_SCRIPT, '--', messageId]
}

export async function readSelectedAppleMailSource(messageId: string): Promise<string> {
  if (process.platform !== 'darwin') {
    throw new Error('Apple Mail drops are supported only on macOS')
  }

  try {
    const { stdout } = await execFileAsync('/usr/bin/osascript', appleMailOsaScriptArgs(messageId), {
      encoding: 'utf8',
      maxBuffer: MAX_MAIL_SOURCE_BYTES,
      timeout: APPLE_MAIL_SCRIPT_TIMEOUT_MS
    })

    return stdout
  } catch (err) {
    const detail = err instanceof Error ? `${err.message} ${'stderr' in err ? String(err.stderr || '') : ''}` : String(err)

    if (/not authorized to send apple events|-1743/i.test(detail)) {
      throw new Error(
        'Hermes needs Apple Mail automation permission. Enable Hermes under System Settings → Privacy & Security → Automation.'
      )
    }

    if (/timed out|etimedout/i.test(detail)) {
      throw new Error('Apple Mail did not respond in time. Keep Mail open and try the drop again.')
    }

    if (/no longer selected/i.test(detail)) {
      throw new Error('The dragged Apple Mail message is no longer selected. Drag it again from Mail.')
    }

    throw new Error('Apple Mail could not export the dragged message.')
  }
}

interface ExportSelectedAppleMailMessageOptions {
  messageUri: string
  readSelectedSource?: (messageId: string) => Promise<string>
  userDataDir: string
}

export async function exportSelectedAppleMailMessage({
  messageUri,
  readSelectedSource = readSelectedAppleMailSource,
  userDataDir
}: ExportSelectedAppleMailMessageOptions): Promise<string> {
  const messageId = decodeAppleMailMessageUri(messageUri)

  if (!messageId) {
    throw new Error('Invalid Apple Mail message reference')
  }

  const source = await readSelectedSource(messageId)

  if (!source) {
    throw new Error('Apple Mail returned an empty message source')
  }

  if (Buffer.byteLength(source, 'utf8') > MAX_MAIL_SOURCE_BYTES) {
    throw new Error('Apple Mail message exceeds the 50 MB drop limit')
  }

  const dir = await managedExportDir(userDataDir, true)

  if (!dir) {
    throw new Error('Could not create the managed Apple Mail export directory')
  }

  const stamp = new Date().toISOString().replace(/[:.]/g, '-').replace('T', '_').replace('Z', '')
  const idHash = crypto.createHash('sha256').update(messageId).digest('hex').slice(0, 12)
  const random = crypto.randomBytes(6).toString('hex')
  const filePath = path.join(dir, `apple_mail_${stamp}_${idHash}_${random}.eml`)
  const flags = fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_NOFOLLOW
  const handle = await fs.promises.open(filePath, flags, 0o600)
  let committed = false

  try {
    await handle.writeFile(source, { encoding: 'utf8' })
    await handle.close()
    committed = true
  } finally {
    if (!committed) {
      await handle.close().catch(() => undefined)
      await fs.promises.unlink(filePath).catch(() => undefined)
    }
  }

  return filePath
}

async function validateManagedFilePath(dir: string, filePath: string): Promise<string> {
  const lexicalTarget = path.resolve(filePath)
  const realParent = await fs.promises.realpath(path.dirname(lexicalTarget))
  const target = path.join(dir, path.basename(lexicalTarget))

  if (realParent !== dir || path.extname(target).toLowerCase() !== '.eml') {
    throw new Error('Refusing to remove a path outside the managed Apple Mail export directory')
  }

  return target
}

export async function removeAppleMailExport(userDataDir: string, filePath: string): Promise<boolean> {
  const dir = await managedExportDir(userDataDir, false)

  if (!dir) {
    return false
  }

  const target = await validateManagedFilePath(dir, filePath)
  let stat: fs.Stats

  try {
    stat = await fs.promises.lstat(target)
  } catch (err) {
    if ((err as NodeJS.ErrnoException)?.code === 'ENOENT') {
      return false
    }

    throw err
  }

  if (stat.isSymbolicLink()) {
    throw new Error('Refusing to remove a symlink from the managed Apple Mail export directory')
  }

  if (!stat.isFile()) {
    throw new Error('Refusing to remove a non-file Apple Mail export path')
  }

  await fs.promises.unlink(target)

  return true
}

export async function cleanupAllAppleMailExports(userDataDir: string): Promise<number> {
  const dir = await managedExportDir(userDataDir, false)

  if (!dir) {
    return 0
  }

  const entries = await fs.promises.readdir(dir, { withFileTypes: true })
  let removed = 0

  for (const entry of entries) {
    if (!entry.isFile() || path.extname(entry.name).toLowerCase() !== '.eml') {
      continue
    }

    try {
      if (await removeAppleMailExport(userDataDir, path.join(dir, entry.name))) {
        removed += 1
      }
    } catch (err) {
      if ((err as NodeJS.ErrnoException)?.code !== 'ENOENT') {
        throw err
      }
    }
  }

  return removed
}
