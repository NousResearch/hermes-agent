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
let cachedWslAvailability: null | boolean = null

function isWslAvailable(): boolean {
  if (cachedWslAvailability !== null) {
    return cachedWslAvailability
  }

  if (!IS_WINDOWS) {
    cachedWslAvailability = true

    return cachedWslAvailability
  }

  // Distributions normally belong to HKCU. Neither an empty Lxss shell nor
  // a stale DefaultDistribution value proves a registered distribution exists.
  // Query registry data only: never launch the WSL stub or touch its UNC shares
  // to establish availability (either can activate WSL installation machinery).
  cachedWslAvailability = false
  for (const hive of ['HKCU', 'HKLM']) {
    const root = `${hive}\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Lxss`
    try {
      const out = execFileSync('reg.exe', ['query', root, '/s'], {
        encoding: 'utf8',
        stdio: ['ignore', 'pipe', 'ignore'],
        timeout: 1000,
        windowsHide: true
      })
      // reg.exe prints full HKEY_* names even when queried with abbreviations.
      const fullRoot = root.replace(hive, hive === 'HKCU' ? 'HKEY_CURRENT_USER' : 'HKEY_LOCAL_MACHINE')
      let distributionSubkey = false
      for (const line of out.split(/\r?\n/)) {
        const key = line.trim()
        if (/^(?:HKEY_|HKCU\\|HKLM\\)/i.test(key)) {
          distributionSubkey = [root, fullRoot].some(prefix => {
            const tail = key.slice(prefix.length + 1)
            return key.toLowerCase().startsWith(`${prefix.toLowerCase()}\\`) && Boolean(tail) && !tail.includes('\\')
          })
        } else if (distributionSubkey && /^\s*DistributionName\s+REG_SZ\s+\S/i.test(line)) {
          cachedWslAvailability = true
          break
        }
      }
      if (cachedWslAvailability) break
    } catch {
      // Missing keys, inaccessible registry data and timeouts are not evidence.
    }
  }

  return cachedWslAvailability
}

/**
 * WSL path eligibility belongs to the backend profile that produced the path.
 * A single desktop process can keep a local primary backend and a remote pool
 * backend alive simultaneously, so a process-global boolean can bleed between
 * them. Unknown profiles retain the historical local default until Electron
 * resolves and records their actual backend mode.
 */
const DEFAULT_WSL_BRIDGE_PROFILE = 'default'
const wslBridgeProfiles = new Map<string, boolean>()
let activeWslBridgeProfile = DEFAULT_WSL_BRIDGE_PROFILE

function normalizeWslBridgeProfile(profile?: null | string): string {
  return String(profile || '').trim() || DEFAULT_WSL_BRIDGE_PROFILE
}

/** Select the profile used by legacy callers that cannot pass one explicitly. */
export function setActiveGatewayProfile(profile?: null | string): void {
  activeWslBridgeProfile = normalizeWslBridgeProfile(profile)
}

/** Record whether paths returned by one profile belong to this host's WSL. */
export function setWslBridgeProfileState(profile: null | string, active: boolean): void {
  wslBridgeProfiles.set(normalizeWslBridgeProfile(profile), Boolean(active))
}

/** Backward-compatible toggle: update only the current fallback profile. */
export function setWslBridgeActive(active: boolean): void {
  setWslBridgeProfileState(activeWslBridgeProfile, active)
}

export function isWslBridgeActive(profile?: null | string): boolean {
  const key = profile == null ? activeWslBridgeProfile : normalizeWslBridgeProfile(profile)
  const configured = wslBridgeProfiles.get(key)

  if (configured !== undefined) {
    // Local backend intent alone does not imply this Windows host has WSL.
    // Remote profiles still short-circuit without even a registry query.
    return configured && isWslAvailable()
  }

  const detected = isWslAvailable()
  wslBridgeProfiles.set(key, detected)

  return detected
}

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
      // On WSL-less machines wsl.exe prints "The Windows Subsystem for Linux
      // is not installed..." to stderr; stderr is inherited by default, so
      // that banner leaks into whatever console the app is attached to
      // (visible e.g. during the update hand-off). Discard it. (#80184)
      stdio: ['ignore', 'pipe', 'ignore'],
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
export function wslPosixToWindowsAccessible(
  posixPath: string,
  distro?: string,
  resolveDistro: () => string = resolveDefaultWslDistro
): string {
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

  return `${wslUncBase(distro ?? resolveDistro())}\\${relative}`
}

/** Native folder dialog `defaultPath`: open a WSL cwd in the Windows picker. */
export function resolvePickerDefaultPath(
  defaultPath: string | undefined,
  distro?: string,
  profile?: null | string
): string | undefined {
  if (!defaultPath) {
    return undefined
  }

  // Remote-gateway POSIX paths can't be opened via wsl.exe — no-op the bridge
  // so the native dialog gets the raw path (it falls back gracefully) instead
  // of triggering a wsl.exe spawn / install prompt. (#66433)
  if (!isWslBridgeActive(profile)) {
    return defaultPath
  }

  const value = String(defaultPath).trim()

  return value.startsWith('/') && !WIN_DRIVE_RE.test(value) ? wslPosixToWindowsAccessible(value, distro) : defaultPath
}

/** fs read path: on Windows, make a WSL cwd readable via its UNC / drive form. */
export function resolveLocalReadPath(dirPath: string, distro?: string, profile?: null | string): string {
  const value = String(dirPath || '').trim()

  // In remote-gateway mode the POSIX paths belong to a host the Windows
  // desktop cannot open locally — skip the WSL bridge entirely (no distro
  // probe, no wsl.exe) so the file panel never spawns the install prompt on
  // WSL-less machines. (#66433)
  if (!isWslBridgeActive(profile)) {
    return value
  }

  return IS_WINDOWS && value.startsWith('/') && !WIN_DRIVE_RE.test(value)
    ? wslPosixToWindowsAccessible(value, distro)
    : value
}
