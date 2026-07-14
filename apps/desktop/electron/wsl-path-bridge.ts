import { execFileSync } from 'node:child_process'
import fs from 'node:fs'

// Bridges WSL/POSIX paths into forms the *Windows host* can open, for the case
// where the desktop UI runs on Windows and the gateway runs inside WSL (remote
// mode). Only the Windows-side direction lives here: the native folder dialog's
// defaultPath and the fs read path. The reverse (whatever path the backend
// receives → POSIX) is handled once, gateway-side, in
// hermes_constants.translate_cwd_for_wsl_backend, so it stays picker-agnostic.

const IS_WINDOWS = process.platform === 'win32'
const WIN_DRIVE_RE = /^([A-Za-z]):[\\/]/
// `/mnt/c` and `/mnt/c/...` (drvfs default automount root).
const WSL_MOUNT_RE = /^\/mnt\/([a-z])(?:\/(.*))?$/i

let cachedDistro: null | string = null
let cachedUncBase: null | string = null

/**
 * Pick the default distro from `wsl.exe -l -q` output.
 *
 * `wsl.exe` emits UTF-16LE without a BOM unless `WSL_UTF8=1` (WSL >= 0.64), so
 * older builds leave NUL bytes between characters even when we ask for utf8 —
 * strip them defensively before splitting. The default distro is the first
 * (`*`-marked, decoration removed by `-q`) entry. See microsoft/WSL#4607.
 */
export function parseDefaultDistro(raw: string): null | string {
  return (
    String(raw || '')
      .replace(/\0/g, '')
      .split(/\r?\n/)
      .map(line => line.replace(/^\*?\s*/, '').trim())
      .find(Boolean) || null
  )
}

/** Default WSL distro name (cached). Falls back to `Ubuntu`. */
export function resolveDefaultWslDistro(): string {
  if (cachedDistro) {
    return cachedDistro
  }

  if (!IS_WINDOWS) {
    cachedDistro = 'Ubuntu'

    return cachedDistro
  }

  try {
    const out = execFileSync('wsl.exe', ['-l', '-q'], {
      encoding: 'utf8',
      env: { ...process.env, WSL_UTF8: '1' },
      timeout: 2000,
      windowsHide: true
    })
    cachedDistro = parseDefaultDistro(out) || 'Ubuntu'
  } catch {
    cachedDistro = 'Ubuntu'
  }

  return cachedDistro
}

// `\\wsl.localhost\<distro>` (Win11 / Win10 >= 21364) with a `\\wsl$\<distro>`
// fallback for older builds. Probed once; defaults to wsl.localhost.
function wslUncBase(distro: string): string {
  if (cachedUncBase) {
    return cachedUncBase
  }

  const modern = `\\\\wsl.localhost\\${distro}`
  const legacy = `\\\\wsl$\\${distro}`

  try {
    if (!fs.existsSync(modern) && fs.existsSync(legacy)) {
      cachedUncBase = legacy

      return cachedUncBase
    }
  } catch {
    // Network-path probe failed — prefer the modern form.
  }

  cachedUncBase = modern

  return cachedUncBase
}

/**
 * A WSL/POSIX path → a path the Windows host can open: `/mnt/c/...` → `C:\...`
 * (drvfs mount), any other absolute POSIX path → `\\wsl.localhost\<distro>\...`.
 * Non-absolute or already-Windows paths pass through.
 */
export function wslPosixToWindowsAccessible(posixPath: string, distro: string = resolveDefaultWslDistro()): string {
  const value = String(posixPath || '').trim()
  const normalized = value.replace(/\\/g, '/')

  if (!normalized.startsWith('/')) {
    return value
  }

  const mount = normalized.match(WSL_MOUNT_RE)

  if (mount) {
    const tail = (mount[2] || '').replace(/\//g, '\\')

    return tail ? `${mount[1].toUpperCase()}:\\${tail}` : `${mount[1].toUpperCase()}:\\`
  }

  const relative = normalized.replace(/^\/+/, '').replace(/\//g, '\\')

  return `${wslUncBase(distro)}\\${relative}`
}

/** Native folder dialog `defaultPath`: open a WSL cwd in the Windows picker. */
export function resolvePickerDefaultPath(
  defaultPath: string | undefined,
  distro: string = resolveDefaultWslDistro()
): string | undefined {
  if (!defaultPath) {
    return undefined
  }

  const value = String(defaultPath).trim()

  return value.startsWith('/') && !WIN_DRIVE_RE.test(value) ? wslPosixToWindowsAccessible(value, distro) : defaultPath
}

/** fs read path: on Windows, make a WSL cwd readable via its UNC / drive form. */
export function resolveLocalReadPath(dirPath: string, distro: string = resolveDefaultWslDistro()): string {
  const value = String(dirPath || '').trim()

  return IS_WINDOWS && value.startsWith('/') && !WIN_DRIVE_RE.test(value)
    ? wslPosixToWindowsAccessible(value, distro)
    : value
}
