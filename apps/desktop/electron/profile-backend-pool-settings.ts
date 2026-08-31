import fs from 'node:fs'
import path from 'node:path'

export const DEFAULT_PROFILE_BACKEND_POOL_MAX = 4
export const PROFILE_BACKEND_POOL_SETTINGS_FILE = 'profile-backend-settings.json'

export interface ProfileBackendPoolSettings {
  maxBackends: number
}

export function normalizeProfileBackendPoolMax(value: unknown, fallback = DEFAULT_PROFILE_BACKEND_POOL_MAX): number {
  const numeric = typeof value === 'string' && value.trim() !== '' ? Number(value.trim()) : value

  if (typeof numeric === 'number' && Number.isSafeInteger(numeric) && numeric >= 0) {
    return numeric
  }

  return fallback
}

export function profileBackendPoolSettingsPath(userDataDir: string): string {
  return path.join(userDataDir, PROFILE_BACKEND_POOL_SETTINGS_FILE)
}

export function readProfileBackendPoolSettings(
  userDataDir: string,
  readFile: (filePath: string, encoding: BufferEncoding) => string = (filePath, encoding) =>
    fs.readFileSync(filePath, encoding)
): ProfileBackendPoolSettings {
  try {
    const parsed: unknown = JSON.parse(readFile(profileBackendPoolSettingsPath(userDataDir), 'utf8'))
    const maxBackends =
      parsed && typeof parsed === 'object' && 'max_backends' in parsed
        ? (parsed as { max_backends?: unknown }).max_backends
        : undefined

    return { maxBackends: normalizeProfileBackendPoolMax(maxBackends) }
  } catch {
    return { maxBackends: DEFAULT_PROFILE_BACKEND_POOL_MAX }
  }
}

export function writeProfileBackendPoolSettings(
  userDataDir: string,
  maxBackends: unknown,
  writeFile: (filePath: string, content: string, options: { encoding: BufferEncoding; mode: number }) => void = (
    filePath,
    content,
    options
  ) => fs.writeFileSync(filePath, content, options),
  rename: (from: string, to: string) => void = (from, to) => fs.renameSync(from, to),
  mkdir: (dir: string) => void = dir => fs.mkdirSync(dir, { recursive: true, mode: 0o700 })
): ProfileBackendPoolSettings {
  const settings = { maxBackends: normalizeProfileBackendPoolMax(maxBackends) }
  const target = profileBackendPoolSettingsPath(userDataDir)
  const temporary = `${target}.${process.pid}.tmp`

  mkdir(userDataDir)
  writeFile(
    temporary,
    `${JSON.stringify({ version: 1, max_backends: settings.maxBackends }, null, 2)}\n`,
    { encoding: 'utf8', mode: 0o600 }
  )
  rename(temporary, target)

  return settings
}
