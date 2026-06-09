/**
 * Resolve a hostname or device identifier to a human-friendly nickname.
 *
 * Falls back through:
 *  1. User-defined nicknames (from localStorage)
 *  2. Heuristic shortening of common hostname patterns
 *  3. Raw host as last resort
 */

const NICKNAME_STORAGE_KEY = 'hermes:device-nicknames'

type NicknameMap = Record<string, string>

function loadNicknames(): NicknameMap {
  try {
    const raw = localStorage.getItem(NICKNAME_STORAGE_KEY)
    return raw ? JSON.parse(raw) : {}
  } catch {
    return {}
  }
}

function saveNicknames(nm: NicknameMap) {
  try {
    localStorage.setItem(NICKNAME_STORAGE_KEY, JSON.stringify(nm))
  } catch {
    // quota exceeded — non-fatal
  }
}

/**
 * Derive a short name from common hostname patterns:
 *   Omars-MacBook-Pro-3.local → MacBook Pro
 *   DESKTOP-ABC123 → Desktop
 *   Omars-MacBook-Air.local → MacBook Air
 *   Taro → Taro
 *   KO-NAS.local → NAS
 */
function heuristicNickname(host: string): string {
  const base = host.replace(/\.local$/, '').replace(/\.lan$/, '')

  // "Omars-MacBook-Pro-3" → "MacBook Pro"
  const m = base.match(/[-_](MacBook[\s_-]?[A-Za-z]+)[-_]?\d*$/)
  if (m) {
    return m[1].replace(/[-_]/g, ' ')
  }

  // "DESKTOP-ABC123" → "Desktop"
  if (/^[-_]?desktop[-_]/i.test(base)) {
    return 'Desktop'
  }

  // "OMARS-WIN11" → "Win11"
  if (/win\d{2}/i.test(base)) {
    const wm = base.match(/(win\d{2})/i)
    return wm ? wm[1].replace('win', 'Win') : base
  }

  // "KO-NAS" / "nas.local" → "NAS"
  if (/[-_]nas[-_]/i.test(base) || /^nas$/i.test(base)) {
    return 'NAS'
  }

  // Single-word hostnames (e.g., "Taro") pass through
  if (!base.includes('-') && !base.includes('_')) {
    return base
  }

  // Take the last segment for hyphenated names: "Omars-MacBook-Pro-3" → "3" (bad)
  // Instead take the longest meaningful segment
  const parts = base.split(/[-_]/)
  const meaningful = parts.filter(p => !/^\d+$/.test(p))
  if (meaningful.length >= 2) {
    return meaningful.slice(-2).join(' ')
  }

  return base
}

export function resolveDeviceNickname(host: string | undefined): string {
  if (!host) {
    return 'unknown'
  }

  const nicknames = loadNicknames()
  const stored = nicknames[host]
  if (stored) {
    return stored
  }

  return heuristicNickname(host)
}

/**
 * Set a custom nickname for a device. Persisted in localStorage.
 */
export function setDeviceNickname(host: string, nickname: string) {
  const nicknames = loadNicknames()
  nicknames[host] = nickname
  saveNicknames(nicknames)
}

/** Cached hostname lookup — avoids repeated async calls. */
let hostnameCache: string | null = null

function getHostname(): string {
  if (hostnameCache) return hostnameCache
  // Use Electron's remote hostname if available
  if (typeof window !== 'undefined' && (window as any).getHostname) {
    hostnameCache = (window as any).getHostname()
  } else if (typeof navigator !== 'undefined') {
    // Fallback: use platform from navigator (e.g., "MacIntel", "Linux x86_64")
    hostnameCache = navigator.platform || 'this-machine'
  } else {
    hostnameCache = 'this-machine'
  }
  return hostnameCache as string
}

/**
 * Get the nickname for the current device running this app.
 * Used to label local sessions in the sidebar.
 */
export function currentDeviceNickname(): string {
  return resolveDeviceNickname(getHostname())
}

/**
 * Check if the nickname came from user customization or heuristic.
 */
export function isCustomNickname(host: string): boolean {
  const nicknames = loadNicknames()
  return host in nicknames
}

/**
 * Get the operating system label from a presence record's client or os field.
 */
export function osLabel(os: string | undefined): string | null {
  if (!os) return null
  const l = os.toLowerCase()
  if (l.startsWith('mac') || l.startsWith('darwin')) return 'macOS'
  if (l.startsWith('linux')) return 'Linux'
  if (l.startsWith('windows') || l.startsWith('win')) return 'Windows'
  return os
}
