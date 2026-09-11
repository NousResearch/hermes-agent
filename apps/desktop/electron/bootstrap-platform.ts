import fs from 'node:fs'

function isWslEnvironment(env = process.env, platform = process.platform, kernelRelease = null) {
  if (platform !== 'linux') {
    return false
  }

  if (env.WSL_DISTRO_NAME || env.WSL_INTEROP) {
    return true
  }

  try {
    const release = kernelRelease ?? fs.readFileSync('/proc/sys/kernel/osrelease', 'utf8')

    return /microsoft|wsl/i.test(release)
  } catch {
    return false
  }
}

function isWindowsBinaryPathInWsl(
  filePath,
  options: { isWsl?: boolean; env?: NodeJS.ProcessEnv; platform?: NodeJS.Platform } = {}
) {
  const isWsl = options.isWsl ?? isWslEnvironment(options.env, options.platform)

  if (!isWsl) {
    return false
  }

  const normalized = String(filePath || '')
    .replace(/\\/g, '/')
    .toLowerCase()

  return (
    normalized.endsWith('.exe') ||
    normalized.endsWith('.cmd') ||
    normalized.endsWith('.bat') ||
    normalized.endsWith('.ps1')
  )
}

function bundledRuntimeImportCheck(platform = process.platform) {
  return platform === 'win32' ? 'import fastapi, uvicorn, winpty' : 'import fastapi, uvicorn, ptyprocess'
}

const GPU_OVERRIDE_ON = new Set(['1', 'true', 'yes', 'on'])
const GPU_OVERRIDE_OFF = new Set(['0', 'false', 'no', 'off'])

/**
 * Decide whether the app is being shown over a remote/forwarded display, where
 * Chromium's GPU compositor produces an unstable, flickering surface (it can't
 * present accelerated layers cleanly over the wire). Native local Windows/macOS
 * sessions composite locally and never hit this, so we only fall back to
 * software rendering when a remote display is detected.
 *
 * Returns a short reason string when GPU acceleration should be disabled, or
 * null to keep it enabled. `HERMES_DESKTOP_DISABLE_GPU` overrides detection
 * both ways (1/true/yes/on → always disable, 0/false/no/off → never disable).
 *
 * Pure + dependency-free so it can be unit-tested and called before app ready.
 */
function detectRemoteDisplay(options: { env?: NodeJS.ProcessEnv; platform?: NodeJS.Platform } = {}) {
  const env = options.env ?? process.env
  const platform = options.platform ?? process.platform

  const override = String(env.HERMES_DESKTOP_DISABLE_GPU || '')
    .trim()
    .toLowerCase()

  if (GPU_OVERRIDE_ON.has(override)) {
    return 'override (HERMES_DESKTOP_DISABLE_GPU)'
  }

  if (GPU_OVERRIDE_OFF.has(override)) {
    return null
  }

  // Launched from an SSH session → the display is X11-forwarded or otherwise
  // remote. Covers the common `ssh user@box` + GUI-forwarding case.
  if (env.SSH_CONNECTION || env.SSH_CLIENT || env.SSH_TTY) {
    return 'ssh-session'
  }

  if (platform === 'linux') {
    // X11 forwarding sets DISPLAY to "<host>:N" (e.g. "localhost:10.0"); a
    // local X server is ":0"/":1" with no host part before the colon.
    // NB: WSLg deliberately isn't treated as remote — it reports
    // GPU-accelerated vGPU surfaces locally and doesn't show the flicker.
    const display = String(env.DISPLAY || '')

    if (display.includes(':') && display.split(':')[0]) {
      return `x11-forwarding (DISPLAY=${display})`
    }
  }

  if (platform === 'win32') {
    // RDP sessions report SESSIONNAME like "RDP-Tcp#7"; the local console is
    // "Console".
    const sessionName = String(env.SESSIONNAME || '')

    if (/^rdp-/i.test(sessionName)) {
      return `rdp (SESSIONNAME=${sessionName})`
    }
  }

  return null
}

const LINUX_PASSWORD_STORES = new Set(['gnome-libsecret', 'kwallet', 'kwallet5', 'kwallet6', 'basic'])

/**
 * Resolve the Chromium `--password-store` switch for Linux safeStorage.
 *
 * Without the switch Chromium often fails to pick a keychain backend when the
 * app is launched outside a full desktop session, safeStorage reports
 * encryption as unavailable, and hardening.ts refuses to persist remote
 * gateway tokens. The `hermes desktop` launcher detects the session keychain
 * (or reads `desktop.password_store` from config.yaml) and bridges the value
 * in via HERMES_DESKTOP_PASSWORD_STORE.
 *
 * Returns `{ store, warning }`: `store` is the validated backend to apply (or
 * null to leave Chromium's default), `warning` is a message to log for
 * unrecognized values. Pure + dependency-free so it can be unit-tested and
 * called before app ready.
 */
function resolveLinuxPasswordStore(options: { env?: NodeJS.ProcessEnv; platform?: NodeJS.Platform } = {}) {
  const env = options.env ?? process.env
  const platform = options.platform ?? process.platform

  const requested = String(env.HERMES_DESKTOP_PASSWORD_STORE || '').trim()

  if (platform !== 'linux' || !requested) {
    return { store: null, warning: null }
  }

  if (!LINUX_PASSWORD_STORES.has(requested)) {
    return { store: null, warning: `ignoring unknown HERMES_DESKTOP_PASSWORD_STORE value: ${requested}` }
  }

  return { store: requested, warning: null }
}

function requestedOzonePlatform(env: NodeJS.ProcessEnv, argv: readonly string[]): string | null {
  let explicit: string | null = null
  let hint: string | null = null

  for (const arg of argv) {
    const match = /^--ozone-platform(-hint)?=(.+)$/.exec(arg)

    if (!match) {
      continue
    }

    if (match[1]) {
      hint = match[2].toLowerCase()
    } else {
      explicit = match[2].toLowerCase()
    }
  }

  return explicit ?? hint ?? env.ELECTRON_OZONE_PLATFORM_HINT?.toLowerCase() ?? null
}

function linuxSessionIsWayland(env: NodeJS.ProcessEnv): boolean {
  return env.XDG_SESSION_TYPE === 'wayland' || (Boolean(env.WAYLAND_DISPLAY) && !env.DISPLAY)
}

/**
 * Merge Chromium `--disable-features` lists without dropping caller values.
 */
function mergeDisableFeatures(existing: string | undefined, extra: readonly string[]): string {
  const seen = new Set<string>()
  const out: string[] = []

  for (const part of [...String(existing || '').split(','), ...extra]) {
    const name = part.trim()

    if (!name) {
      continue
    }

    const key = name.toLowerCase()

    if (seen.has(key)) {
      continue
    }

    seen.add(key)
    out.push(name)
  }

  return out.join(',')
}

/**
 * Chromium's Wayland ozone backend is incompatible with Vulkan. On that path
 * the GPU process dies after ~30s (`GPU process isn't usable. Goodbye.`)
 * even when `app.disableHardwareAcceleration()` / `HERMES_DESKTOP_DISABLE_GPU`
 * already ran. `--disable-features=Vulkan` keeps GL acceleration and avoids
 * the unsupported path.
 *
 * Returns the merged `--disable-features` value when Linux ozone will be
 * Wayland, or null when the switch is not needed.
 */
function linuxWaylandVulkanDisableFeatures(
  options: {
    env?: NodeJS.ProcessEnv
    platform?: NodeJS.Platform
    argv?: readonly string[]
    existingDisableFeatures?: string
  } = {}
): string | null {
  const env = options.env ?? process.env
  const platform = options.platform ?? process.platform
  const argv = options.argv ?? []

  if (platform !== 'linux') {
    return null
  }

  const requested = requestedOzonePlatform(env, argv)
  const ozoneIsWayland =
    requested === 'wayland' || ((requested === null || requested === 'auto') && linuxSessionIsWayland(env))

  if (!ozoneIsWayland) {
    return null
  }

  return mergeDisableFeatures(options.existingDisableFeatures, ['Vulkan'])
}

export {
  bundledRuntimeImportCheck,
  detectRemoteDisplay,
  isWindowsBinaryPathInWsl,
  isWslEnvironment,
  linuxWaylandVulkanDisableFeatures,
  mergeDisableFeatures,
  resolveLinuxPasswordStore
}
