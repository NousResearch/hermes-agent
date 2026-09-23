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

/**
 * Detect a locally-numbered $DISPLAY (":N", no host part) with no backing X11
 * socket. When this happens, Electron's Ozone/X11 backend fails deep in
 * native code with a cryptic "Missing X server or $DISPLAY" crash and no
 * actionable message — this lets the caller print something useful and exit
 * cleanly instead. The common cause is a terminal multiplexer (tmux/screen)
 * pane that outlives the X session that set $DISPLAY (e.g. an Xwayland
 * restart, or an SSH X11 forward from a connection that's since closed) —
 * new panes pick up the current display, but panes already open keep the
 * stale value.
 *
 * Forwarded displays ("host:N", e.g. SSH X11 forwarding over TCP) aren't
 * covered here since verifying those needs a network round trip; those are
 * left to Electron's own (unfriendly) failure.
 *
 * Returns a description of the problem, or null if the display looks fine
 * (or isn't ours to check — Wayland, non-Linux, forwarded, or unset).
 *
 * Pure + dependency-free so it can be unit-tested and called before app ready.
 */
function detectDeadLocalDisplay(
  options: { env?: NodeJS.ProcessEnv; platform?: NodeJS.Platform; existsSync?: (path: string) => boolean } = {}
) {
  const env = options.env ?? process.env
  const platform = options.platform ?? process.platform
  const existsSync = options.existsSync ?? fs.existsSync

  if (platform !== 'linux' || env.WAYLAND_DISPLAY) {
    return null
  }

  const match = /^:(\d+)(?:\.\d+)?$/.exec(String(env.DISPLAY || ''))

  if (!match) {
    return null
  }

  const socketPath = `/tmp/.X11-unix/X${match[1]}`

  if (existsSync(socketPath)) {
    return null
  }

  return `DISPLAY=${env.DISPLAY} set, but ${socketPath} doesn't exist`
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

export {
  bundledRuntimeImportCheck,
  detectDeadLocalDisplay,
  detectRemoteDisplay,
  isWindowsBinaryPathInWsl,
  isWslEnvironment,
  resolveLinuxPasswordStore
}
