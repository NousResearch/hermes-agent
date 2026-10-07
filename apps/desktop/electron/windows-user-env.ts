// windows-user-env.ts
//
// Read a User-scoped environment variable straight from the Windows registry
// (HKCU\Environment).
//
// A GUI app launched from Explorer inherits the environment block captured at
// login, so a variable set via `setx` AFTER login is invisible in process.env
// even though a fresh shell — and the Hermes CLI — sees it immediately. The
// desktop's HERMES_HOME resolution relies on process.env, so that stale-snapshot
// gap silently sends the backend to the default %LOCALAPPDATA%\hermes. Reading
// the live registry value closes the gap. See #45471.

import { execFileSync } from 'node:child_process'

// Parse the output of `reg query HKCU\Environment /v <name>`, which looks like:
//
//   HKEY_CURRENT_USER\Environment
//       HERMES_HOME    REG_SZ    F:\Hermes\data
//
// Returns the raw value string (spaces inside the value preserved), or null when
// the requested value line isn't present.
function parseRegQueryValue(stdout, name) {
  if (!stdout || !name) {
    return null
  }

  const typePattern = /^(\S+)\s+(?:REG_SZ|REG_EXPAND_SZ|REG_MULTI_SZ|REG_DWORD|REG_QWORD|REG_BINARY|REG_NONE)\s+(.*)$/

  for (const rawLine of String(stdout).split(/\r?\n/)) {
    const line = rawLine.trim()
    const match = line.match(typePattern)

    if (match && match[1].toLowerCase() === name.toLowerCase()) {
      return match[2]
    }
  }

  return null
}

// Expand %VAR% references against an env map. REG_EXPAND_SZ values store
// unexpanded references; plain REG_SZ paths have none, so this is a no-op for
// the common F:\... case. Unknown references are left verbatim.
function expandWindowsEnvRefs(value, env = process.env) {
  if (!value) {
    return value
  }

  return value.replace(/%([^%]+)%/g, (whole, name) => {
    const key = Object.keys(env).find(k => k.toUpperCase() === String(name).toUpperCase())

    return key != null && env[key] != null ? env[key] : whole
  })
}

// True when every byte is below 0x80.
//
// `reg.exe` is a native console program: its stdout carries the machine code
// page, not UTF-8. While the bytes are pure ASCII the two agree and decoding as
// UTF-8 is exact. Once any byte is >= 0x80 they do not, and Node exposes no API
// for the host code page — so rather than guess, we re-read the value through a
// channel that carries its own encoding (below). On CP932 the guess would not
// merely be lossy: a two-byte character whose trail byte is 0x5C survives a
// UTF-8 decode as a literal backslash, so `C:\<kanji>\hermes` comes back with
// MORE path separators than it started with, and the caller treats that
// non-empty string as a valid HERMES_HOME instead of falling back (#45471).
function isAsciiBytes(buf) {
  for (let i = 0; i < buf.length; i += 1) {
    if (buf[i] >= 0x80) {
      return false
    }
  }

  return true
}

const SAFE_ENV_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/

// Re-read the value via PowerShell, base64-encoded from UTF-8 inside the child
// so the bytes on the wire are ASCII and no code page is involved. Mirrors the
// approach already used for clipboard text in ui-tui/src/lib/clipboard.ts.
function readUserEnvVarAsBase64(name, exec) {
  if (!SAFE_ENV_NAME.test(name)) {
    return null
  }

  const script =
    '[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(' +
    "[Environment]::GetEnvironmentVariable('" +
    name +
    "','User')))"

  let out

  try {
    out = exec('powershell', ['-NoProfile', '-NonInteractive', '-Command', script], {
      encoding: 'utf8',
      windowsHide: true,
      timeout: 5000
    })
  } catch {
    return null
  }

  const b64 = String(out == null ? '' : out).trim()

  if (!b64) {
    return null
  }

  return Buffer.from(b64, 'base64').toString('utf8') || null
}

// Read a User-scoped env var from HKCU\Environment. Windows-only: returns null
// off-Windows (without spawning), on any spawn error, when `reg` exits non-zero
// (the value doesn't exist), or when the value is empty.
function readWindowsUserEnvVar(
  name,
  {
    platform = process.platform,
    env = process.env,
    exec = execFileSync
  }: {
    platform?: NodeJS.Platform
    env?: NodeJS.ProcessEnv
    exec?: typeof execFileSync | ((file?: string, args?: any) => string)
  } = {}
) {
  if (platform !== 'win32' || !name) {
    return null
  }

  let stdout

  try {
    // No `encoding` — take the bytes, decide how to read them below.
    const out = exec('reg', ['query', 'HKCU\\Environment', '/v', name], {
      windowsHide: true,
      timeout: 5000
    })
    const buf = Buffer.isBuffer(out) ? out : Buffer.from(String(out == null ? '' : out), 'utf8')

    if (!isAsciiBytes(buf)) {
      const viaBase64 = readUserEnvVarAsBase64(name, exec)

      if (viaBase64 == null) {
        return null
      }

      return expandWindowsEnvRefs(viaBase64, env).trim() || null
    }

    stdout = buf.toString('utf8')
  } catch {
    // `reg` missing, or value absent (reg exits 1) — caller falls back.
    return null
  }

  const raw = parseRegQueryValue(stdout, name)

  if (raw == null) {
    return null
  }

  const expanded = expandWindowsEnvRefs(raw, env).trim()

  return expanded || null
}

export { expandWindowsEnvRefs, parseRegQueryValue, readWindowsUserEnvVar }
