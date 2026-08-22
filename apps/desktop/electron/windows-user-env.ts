// windows-user-env.ts
//
// Read environment variables straight from the live Windows registry.
//
// A GUI app launched from Explorer inherits the environment block captured at
// login, so a variable set via `setx` AFTER login is invisible in process.env
// even though a fresh shell — and the Hermes CLI — sees it immediately. The
// Desktop process resolution relies on process.env, so that stale snapshot can
// hide a custom HERMES_HOME or PATH entries written after launch. Expansion
// still uses the inherited environment. See #45471.

import { execFileSync } from 'node:child_process'

// Parse the output of `reg query <key> /v <name>`, which looks like:
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

type WindowsEnvReadOptions = {
  platform?: NodeJS.Platform
  env?: NodeJS.ProcessEnv
  exec?: typeof execFileSync | ((file?: string, args?: any) => string)
}

function readWindowsRegistryEnvVar(
  keyPath: string,
  name,
  { platform = process.platform, env = process.env, exec = execFileSync }: WindowsEnvReadOptions = {}
) {
  if (platform !== 'win32' || !name) {
    return null
  }

  let stdout

  try {
    stdout = exec('reg', ['query', keyPath, '/v', name], {
      encoding: 'utf8',
      windowsHide: true,
      timeout: 5000
    })
  } catch {
    return null
  }

  const raw = parseRegQueryValue(stdout, name)

  if (raw == null) {
    return null
  }

  const expanded = expandWindowsEnvRefs(raw, env).trim()

  return expanded || null
}

// Read a User-scoped env var from HKCU\Environment. Windows-only: returns null
// off-Windows (without spawning), on any spawn error, when `reg` exits non-zero
// (the value doesn't exist), or when the value is empty.
function readWindowsUserEnvVar(name, options: WindowsEnvReadOptions = {}) {
  return readWindowsRegistryEnvVar('HKCU\\Environment', name, options)
}

const WINDOWS_MACHINE_ENV_KEY = 'HKLM\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment'

// Return the live effective Windows PATH rather than the environment snapshot
// inherited by an Explorer-launched Electron process. Windows composes the
// machine PATH before the user PATH; keeping that order preserves normal
// command-resolution precedence while exposing newly added literal PATH
// entries without relying on the stale process snapshot.
function readWindowsHostPath({
  platform = process.platform,
  env = process.env,
  exec = execFileSync
}: WindowsEnvReadOptions = {}) {
  if (platform !== 'win32') {
    return null
  }

  const machinePath = readWindowsRegistryEnvVar(WINDOWS_MACHINE_ENV_KEY, 'Path', { platform, env, exec })
  if (!machinePath) {
    return null
  }

  const userPath = readWindowsRegistryEnvVar('HKCU\\Environment', 'Path', { platform, env, exec })

  return userPath ? `${machinePath};${userPath}` : machinePath
}

export { expandWindowsEnvRefs, parseRegQueryValue, readWindowsHostPath, readWindowsUserEnvVar }
