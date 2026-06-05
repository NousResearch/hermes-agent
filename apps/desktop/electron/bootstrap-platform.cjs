const fs = require('node:fs')
const childProcess = require('node:child_process')

function isWslEnvironment(env = process.env, platform = process.platform, kernelRelease = null) {
  if (platform !== 'linux') return false
  if (env.WSL_DISTRO_NAME || env.WSL_INTEROP) return true

  try {
    const release = kernelRelease ?? fs.readFileSync('/proc/sys/kernel/osrelease', 'utf8')
    return /microsoft|wsl/i.test(release)
  } catch {
    return false
  }
}

function isWindowsBinaryPathInWsl(filePath, options = {}) {
  const isWsl = options.isWsl ?? isWslEnvironment(options.env, options.platform)
  if (!isWsl) return false

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

function nvidiaDriverMajor(version) {
  const match = String(version || '').trim().match(/^(\d+)(?:\.|$)/)
  if (!match) return null
  const major = Number.parseInt(match[1], 10)
  return Number.isFinite(major) ? major : null
}

function detectNvidiaDriverVersion(options = {}) {
  const execFileSync = options.execFileSync ?? childProcess.execFileSync

  try {
    const output = execFileSync('nvidia-smi', ['--query-gpu=driver_version', '--format=csv,noheader'], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
      timeout: 1000
    })
    return String(output || '').split(/\r?\n/).map(line => line.trim()).find(Boolean) || null
  } catch {
    return null
  }
}

/**
 * Decide whether the desktop shell should fall back to software rendering.
 * Remote/forwarded displays can make Chromium's GPU compositor flicker, and
 * NVIDIA 580.x Linux drivers can crash Electron/ANGLE during GPU startup.
 *
 * Returns a short reason string when GPU acceleration should be disabled, or
 * null to keep it enabled. `HERMES_DESKTOP_DISABLE_GPU` overrides detection
 * both ways (1/true/yes/on → always disable, 0/false/no/off → never disable).
 *
 * Pure + dependency-free so it can be unit-tested and called before app ready.
 */
function detectRemoteDisplay(options = {}) {
  const env = options.env ?? process.env
  const platform = options.platform ?? process.platform

  const override = String(env.HERMES_DESKTOP_DISABLE_GPU || '')
    .trim()
    .toLowerCase()
  if (GPU_OVERRIDE_ON.has(override)) return 'override (HERMES_DESKTOP_DISABLE_GPU)'
  if (GPU_OVERRIDE_OFF.has(override)) return null

  // Launched from an SSH session → the display is X11-forwarded or otherwise
  // remote. Covers the common `ssh user@box` + GUI-forwarding case.
  if (env.SSH_CONNECTION || env.SSH_CLIENT || env.SSH_TTY) return 'ssh-session'

  if (platform === 'linux') {
    // X11 forwarding sets DISPLAY to "<host>:N" (e.g. "localhost:10.0"); a
    // local X server is ":0"/":1" with no host part before the colon.
    // NB: WSLg deliberately isn't treated as remote — it reports
    // GPU-accelerated vGPU surfaces locally and doesn't show the flicker.
    const display = String(env.DISPLAY || '')
    if (display.includes(':') && display.split(':')[0]) {
      return `x11-forwarding (DISPLAY=${display})`
    }

    const isWsl = options.isWsl ?? isWslEnvironment(env, platform, options.kernelRelease)
    const nvidiaVersion = options.nvidiaDriverVersion ?? (isWsl ? null : detectNvidiaDriverVersion(options))
    const nvidiaMajor = nvidiaDriverMajor(nvidiaVersion)
    if (nvidiaMajor !== null && nvidiaMajor >= 580) {
      return `nvidia-driver-${nvidiaVersion}`
    }
  }

  if (platform === 'win32') {
    // RDP sessions report SESSIONNAME like "RDP-Tcp#7"; the local console is
    // "Console".
    const sessionName = String(env.SESSIONNAME || '')
    if (/^rdp-/i.test(sessionName)) return `rdp (SESSIONNAME=${sessionName})`
  }

  return null
}

module.exports = {
  bundledRuntimeImportCheck,
  detectNvidiaDriverVersion,
  detectRemoteDisplay,
  isWindowsBinaryPathInWsl,
  isWslEnvironment,
  nvidiaDriverMajor
}
