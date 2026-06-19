const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { fileURLToPath } = require('node:url')

const DEFAULT_FETCH_TIMEOUT_MS = 15_000
const DATA_URL_READ_MAX_BYTES = 16 * 1024 * 1024
const TEXT_PREVIEW_SOURCE_MAX_BYTES = 64 * 1024 * 1024

const SAFE_ENV_SUFFIXES = new Set(['dist', 'example', 'sample', 'template'])
const SENSITIVE_EXTENSIONS = new Set(['.kdbx', '.p12', '.pem', '.pfx'])

function resolveTimeoutMs(timeoutMs, fallbackMs = DEFAULT_FETCH_TIMEOUT_MS) {
  const fallback =
    Number.isFinite(fallbackMs) && Number(fallbackMs) > 0 ? Math.round(Number(fallbackMs)) : DEFAULT_FETCH_TIMEOUT_MS
  const parsed = Number(timeoutMs)

  if (Number.isFinite(parsed) && parsed > 0) {
    return Math.round(parsed)
  }

  return fallback
}

function encryptDesktopSecret(value, safeStorageApi) {
  const raw = String(value || '')

  if (!raw) {
    return null
  }

  let encryptionAvailable = false

  try {
    encryptionAvailable = Boolean(safeStorageApi?.isEncryptionAvailable?.())
  } catch {
    encryptionAvailable = false
  }

  if (!encryptionAvailable) {
    throw new Error(
      'Secure token storage is unavailable, so Hermes Desktop cannot save remote gateway tokens. ' +
        'Set HERMES_DESKTOP_REMOTE_URL and HERMES_DESKTOP_REMOTE_TOKEN in your environment, or enable OS keychain access and try again.'
    )
  }

  try {
    return {
      encoding: 'safeStorage',
      value: safeStorageApi.encryptString(raw).toString('base64')
    }
  } catch (error) {
    const detail = error instanceof Error && error.message ? ` (${error.message})` : ''
    throw new Error(
      `Failed to encrypt the remote gateway token for secure storage${detail}. ` +
        'Set HERMES_DESKTOP_REMOTE_URL and HERMES_DESKTOP_REMOTE_TOKEN in your environment as a fallback.'
    )
  }
}

function sensitiveFileBlockReason(filePath) {
  const normalized = String(filePath || '')
    .replace(/\\/g, '/')
    .toLowerCase()
  const basename = path.basename(normalized)
  const ext = path.extname(basename)

  if (!basename) {
    return null
  }

  if (normalized.includes('/.ssh/')) {
    return 'SSH key/config files are blocked.'
  }

  if (normalized.includes('/.gnupg/')) {
    return 'GPG key material is blocked.'
  }

  if (normalized.endsWith('/.aws/credentials')) {
    return 'AWS credential files are blocked.'
  }

  if (basename === '.env') {
    return '.env files are blocked because they commonly contain secrets.'
  }

  if (basename.startsWith('.env.')) {
    const suffix = basename.slice('.env.'.length)
    if (!SAFE_ENV_SUFFIXES.has(suffix)) {
      return `${basename} is blocked because it appears to contain environment secrets.`
    }
  }

  if (/^id_(rsa|dsa|ecdsa|ed25519)(?:\..+)?$/.test(basename) && !basename.endsWith('.pub')) {
    return 'SSH private key files are blocked.'
  }

  if (SENSITIVE_EXTENSIONS.has(ext)) {
    return `${ext} key/certificate files are blocked.`
  }

  if (basename === '.npmrc' || basename === '.netrc' || basename === '.pypirc') {
    return `${basename} is blocked because it may include auth credentials.`
  }

  return null
}

function ipcPathError(code, message) {
  const error = new Error(message)
  error.code = code
  return error
}

function rejectUnsafePathSyntax(filePath, purpose = 'File read') {
  if (typeof filePath !== 'string') {
    throw ipcPathError('invalid-path', `${purpose} failed: file path is required.`)
  }

  const raw = filePath.trim()

  if (!raw) {
    throw ipcPathError('invalid-path', `${purpose} failed: file path is required.`)
  }

  if (raw.includes('\0')) {
    throw ipcPathError('invalid-path', `${purpose} failed: file path is invalid.`)
  }

  const normalized = raw.replace(/\\/g, '/').toLowerCase()
  if (
    normalized.startsWith('//?/') ||
    normalized.startsWith('//./') ||
    normalized.startsWith('globalroot/device/') ||
    normalized.includes('/globalroot/device/')
  ) {
    throw ipcPathError('device-path', `${purpose} blocked: Windows device paths are not allowed.`)
  }

  return raw
}

function resolveRequestedPathForIpc(filePath, options = {}) {
  const purpose = String(options.purpose || 'File read')
  let raw = rejectUnsafePathSyntax(filePath, purpose)

  // Gateway-reported cwds (config `terminal.cwd`, remote sessions) routinely
  // arrive as `~/...`. Node's fs has no shell — without expansion the path
  // resolves under process.cwd() and every read "ENOENT"s forever.
  if (raw === '~' || raw.startsWith('~/') || raw.startsWith('~\\')) {
    raw = path.join(os.homedir(), raw.slice(1))
  }

  if (/^file:/i.test(raw)) {
    let resolvedPath
    try {
      const parsed = new URL(raw)
      if (parsed.protocol !== 'file:') {
        throw new Error('not a file URL')
      }
      resolvedPath = fileURLToPath(parsed)
    } catch {
      throw ipcPathError('invalid-path', `${purpose} failed: file URL is invalid.`)
    }

    rejectUnsafePathSyntax(resolvedPath, purpose)
    return path.resolve(resolvedPath)
  }

  const baseInput = typeof options.baseDir === 'string' && options.baseDir.trim() ? options.baseDir : process.cwd()
  const safeBaseInput = rejectUnsafePathSyntax(baseInput, purpose)
  const resolvedBase = path.resolve(safeBaseInput)
  rejectUnsafePathSyntax(resolvedBase, purpose)
  const resolvedPath = path.resolve(resolvedBase, raw)
  rejectUnsafePathSyntax(resolvedPath, purpose)

  return resolvedPath
}

async function statForIpc(fsImpl, resolvedPath, purpose, typeLabel) {
  try {
    return await fsImpl.promises.stat(resolvedPath)
  } catch (error) {
    const code = error && typeof error === 'object' ? error.code : ''
    if (code === 'ENOENT' || code === 'ENOTDIR') {
      throw ipcPathError(code || 'ENOENT', `${purpose} failed: ${typeLabel} does not exist.`)
    }
    throw ipcPathError(
      code || 'read-error',
      `${purpose} failed: ${error instanceof Error ? error.message : String(error)}`
    )
  }
}

async function realpathForIpc(fsImpl, resolvedPath, purpose) {
  if (typeof fsImpl.promises.realpath !== 'function') {
    return resolvedPath
  }

  try {
    const realPath = await fsImpl.promises.realpath(resolvedPath)
    rejectUnsafePathSyntax(realPath, purpose)
    return realPath
  } catch (error) {
    const code = error && typeof error === 'object' ? error.code : ''
    throw ipcPathError(
      code || 'read-error',
      `${purpose} failed: ${error instanceof Error ? error.message : String(error)}`
    )
  }
}

function rejectSensitiveFilePath(filePath, purpose) {
  const blockReason = sensitiveFileBlockReason(filePath)
  if (blockReason) {
    throw ipcPathError('sensitive-file', `${purpose} blocked for sensitive file: ${blockReason}`)
  }
}

async function resolveDirectoryForIpc(dirPath, options = {}) {
  const purpose = String(options.purpose || 'Directory read')
  const fsImpl = options.fs || fs
  const resolvedPath = resolveRequestedPathForIpc(dirPath, { baseDir: options.baseDir, purpose })
  const stat = await statForIpc(fsImpl, resolvedPath, purpose, 'directory')

  if (!stat.isDirectory()) {
    throw ipcPathError('ENOTDIR', `${purpose} failed: path is not a directory.`)
  }

  const realPath = await realpathForIpc(fsImpl, resolvedPath, purpose)

  return { realPath, resolvedPath, stat }
}

async function resolveReadableFileForIpc(filePath, options = {}) {
  const purpose = String(options.purpose || 'File read')
  const fsImpl = options.fs || fs
  const resolvedPath = resolveRequestedPathForIpc(filePath, { baseDir: options.baseDir, purpose })

  if (options.blockSensitive !== false) {
    rejectSensitiveFilePath(resolvedPath, purpose)
  }

  const stat = await statForIpc(fsImpl, resolvedPath, purpose, 'file')

  if (stat.isDirectory()) {
    throw ipcPathError('EISDIR', `${purpose} failed: path points to a directory.`)
  }

  if (!stat.isFile()) {
    throw ipcPathError('EINVAL', `${purpose} failed: only regular files can be read.`)
  }

  const realPath = await realpathForIpc(fsImpl, resolvedPath, purpose)
  if (options.blockSensitive !== false) {
    rejectSensitiveFilePath(realPath, purpose)
  }

  const maxBytes = Number.isFinite(options.maxBytes) && Number(options.maxBytes) > 0 ? Number(options.maxBytes) : null
  if (maxBytes && stat.size > maxBytes) {
    throw ipcPathError('EFBIG', `${purpose} failed: file is too large (${stat.size} bytes; limit ${maxBytes} bytes).`)
  }

  try {
    await fsImpl.promises.access(resolvedPath, fs.constants.R_OK)
  } catch {
    throw ipcPathError('EACCES', `${purpose} failed: file is not readable.`)
  }

  return { realPath, resolvedPath, stat }
}

// Files we are willing to hand to the OS "open in default app" handler
// (shell.openPath) when a chat-/LLM-supplied path is clicked. This is a strict
// ALLOWLIST of inert document/media types — never executables, scripts,
// shortcuts, or launchers. A path emitted in chat is an attacker-influenced
// string (SECURITY.md: the LLM is assumed potentially adversarial), and the
// agent can write the very file it links, so opening one by OS file
// association is a code-execution primitive unless the type is provably inert.
// `.html`/`.svg` are excluded on purpose: they execute with a file:// origin in
// the OS browser, so the renderer routes them to the sandboxed preview pane.
const OPENABLE_DEFAULT_APP_EXTENSIONS = new Set([
  '.aac',
  '.avi',
  '.avif',
  '.bmp',
  '.conf',
  '.csv',
  '.flac',
  '.gif',
  '.heic',
  '.ico',
  '.ini',
  '.jpeg',
  '.jpg',
  '.json',
  '.log',
  '.m4a',
  '.m4v',
  '.markdown',
  '.md',
  '.mkv',
  '.mov',
  '.mp3',
  '.mp4',
  '.ogg',
  '.pdf',
  '.png',
  '.rtf',
  '.text',
  '.tif',
  '.tiff',
  '.toml',
  '.tsv',
  '.txt',
  '.wav',
  '.webm',
  '.webp',
  '.xml',
  '.yaml',
  '.yml'
])

// Reject network / UNC paths (`\\host\share`, `//host/share`). rejectUnsafe-
// PathSyntax already blocks Windows device paths; this closes the SMB/NTLM-leak
// and remote-exec surface for the open/reveal handlers.
function rejectRemotePathSyntax(filePath, purpose = 'Open file') {
  const normalized = String(filePath || '').replace(/\\/g, '/')

  if (normalized.startsWith('//')) {
    throw ipcPathError('remote-path', `${purpose} blocked: network/UNC paths are not allowed.`)
  }

  return filePath
}

// Reject paths containing control characters. rejectUnsafePathSyntax already
// blocks NUL; this also blocks newlines/tabs/etc. so a chat-supplied filename
// like `safe.pdf\n\n\n<scary text>.pdf` can't spoof the open-confirm dialog
// (which renders newlines as line breaks and would scroll the real path away).
function rejectControlCharPath(filePath, purpose = 'Open file') {
  // eslint-disable-next-line no-control-regex
  if (/[\u0000-\u001f]/.test(String(filePath || ''))) {
    throw ipcPathError('invalid-path', `${purpose} blocked: path contains control characters.`)
  }

  return filePath
}

// Confine `target` to an already-resolved `base` (the session workspace). Used
// only on the open-in-default-app path so a chat-supplied absolute path can't
// reach arbitrary host files. Callers compare the pre-symlink path against the
// resolved base AND the realpath against the realpath'd base, so a symlinked
// workspace root (e.g. macOS /var → /private/var) doesn't false-reject.
function assertPathWithinBase(target, base, purpose) {
  const rel = path.relative(base, target)

  if (rel === '..' || rel.startsWith(`..${path.sep}`) || path.isAbsolute(rel)) {
    throw ipcPathError('outside-workspace', `${purpose} blocked: file is outside the current workspace.`)
  }
}

function assertOpenableInDefaultApp(filePath, purpose = 'Open file') {
  const ext = path.extname(String(filePath || '')).toLowerCase()

  if (!ext || !OPENABLE_DEFAULT_APP_EXTENSIONS.has(ext)) {
    throw ipcPathError(
      'not-openable',
      `${purpose} blocked: "${ext || 'this file type'}" cannot be opened in the OS default app.`
    )
  }
}

// Resolve + harden an agent-/chat-supplied path for an OS-level "open" or
// "reveal" action. Rejects null-byte / device / UNC paths, blocks sensitive
// files (.ssh, .env, private keys, ...), requires a real existing regular file,
// resolves symlinks (realpath) and re-checks on the real path, and — when
// `requireWithinBase` is set — confines the file to `baseDir`. Does NOT enforce
// the open-in-default-app extension allowlist; callers that launch via
// shell.openPath must additionally call assertOpenableInDefaultApp().
async function resolveOpenablePathForIpc(filePath, options = {}) {
  const purpose = String(options.purpose || 'Open file')
  const fsImpl = options.fs || fs
  const resolvedPath = resolveRequestedPathForIpc(filePath, { baseDir: options.baseDir, purpose })

  rejectControlCharPath(resolvedPath, purpose)
  rejectRemotePathSyntax(resolvedPath, purpose)
  rejectSensitiveFilePath(resolvedPath, purpose)

  const stat = await statForIpc(fsImpl, resolvedPath, purpose, 'file')

  if (stat.isDirectory()) {
    throw ipcPathError('EISDIR', `${purpose} failed: path points to a directory.`)
  }

  if (!stat.isFile()) {
    throw ipcPathError('EINVAL', `${purpose} failed: only regular files can be opened.`)
  }

  const realPath = await realpathForIpc(fsImpl, resolvedPath, purpose)

  rejectControlCharPath(realPath, purpose)
  rejectRemotePathSyntax(realPath, purpose)
  rejectSensitiveFilePath(realPath, purpose)

  if (options.requireWithinBase) {
    const baseInput = typeof options.baseDir === 'string' && options.baseDir.trim() ? options.baseDir : process.cwd()
    const resolvedBase = path.resolve(rejectUnsafePathSyntax(baseInput, purpose))
    let realBase = resolvedBase

    try {
      realBase = await realpathForIpc(fsImpl, resolvedBase, purpose)
    } catch {
      realBase = resolvedBase
    }

    assertPathWithinBase(resolvedPath, resolvedBase, purpose)
    assertPathWithinBase(realPath, realBase, purpose)
  }

  return { realPath, resolvedPath, stat }
}

module.exports = {
  DATA_URL_READ_MAX_BYTES,
  DEFAULT_FETCH_TIMEOUT_MS,
  OPENABLE_DEFAULT_APP_EXTENSIONS,
  TEXT_PREVIEW_SOURCE_MAX_BYTES,
  assertOpenableInDefaultApp,
  encryptDesktopSecret,
  rejectRemotePathSyntax,
  rejectSensitiveFilePath,
  rejectUnsafePathSyntax,
  resolveDirectoryForIpc,
  resolveOpenablePathForIpc,
  resolveReadableFileForIpc,
  resolveRequestedPathForIpc,
  resolveTimeoutMs,
  sensitiveFileBlockReason
}
