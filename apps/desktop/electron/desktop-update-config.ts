import fs from 'node:fs'
import path from 'node:path'

export const DESKTOP_UPDATE_CONFIG_SCHEMA_VERSION = 2
export const DEFAULT_DESKTOP_UPDATE_BRANCH = 'main'

export type DesktopUpdateTrack = 'release' | 'main'
export type DesktopUpdateTrackSource = 'default' | 'established' | 'explicit' | 'migration'

export interface DesktopUpdateConfig {
  schemaVersion: typeof DESKTOP_UPDATE_CONFIG_SCHEMA_VERSION
  track: DesktopUpdateTrack
  trackSource: DesktopUpdateTrackSource
  branch: string
}

interface LoadDesktopUpdateConfigOptions {
  configPath: string
  installationAlreadyExisted: boolean
  defaultBranch?: string
}

function normaliseBranch(value: unknown, fallback = DEFAULT_DESKTOP_UPDATE_BRANCH): string {
  return typeof value === 'string' && value.trim() ? value.trim() : fallback
}

function normaliseTrack(value: unknown): DesktopUpdateTrack | null {
  return value === 'release' || value === 'main' ? value : null
}

function normaliseTrackSource(value: unknown): DesktopUpdateTrackSource | null {
  return value === 'default' || value === 'established' || value === 'explicit' || value === 'migration'
    ? value
    : null
}

function parseDesktopUpdateConfig(
  raw: string,
  defaultBranch = DEFAULT_DESKTOP_UPDATE_BRANCH
): DesktopUpdateConfig | null {
  try {
    const parsed = JSON.parse(raw)
    const track = normaliseTrack(parsed?.track)

    if (track) {
      return {
        schemaVersion: DESKTOP_UPDATE_CONFIG_SCHEMA_VERSION,
        track,
        // Configs written before this field existed represented a persisted
        // selection. Treat them as explicit so recovery fails closed rather
        // than silently repairing a release install from main.
        trackSource: normaliseTrackSource(parsed?.trackSource) || 'explicit',
        branch: normaliseBranch(parsed?.branch, defaultBranch)
      }
    }

    // Legacy Desktop configurations only stored a branch. Those users were
    // already following every commit, so preserve that behaviour during the
    // one-way v2 migration.
    if (typeof parsed?.branch === 'string' && parsed.branch.trim()) {
      return {
        schemaVersion: DESKTOP_UPDATE_CONFIG_SCHEMA_VERSION,
        track: 'main',
        trackSource: 'migration',
        branch: parsed.branch.trim()
      }
    }
  } catch {
    return null
  }

  return null
}

function writeDesktopUpdateConfig(configPath: string, config: DesktopUpdateConfig): DesktopUpdateConfig {
  const normalised: DesktopUpdateConfig = {
    schemaVersion: DESKTOP_UPDATE_CONFIG_SCHEMA_VERSION,
    track: normaliseTrack(config.track) || 'release',
    trackSource: normaliseTrackSource(config.trackSource) || 'explicit',
    branch: normaliseBranch(config.branch)
  }

  const temporaryPath = `${configPath}.tmp`

  fs.mkdirSync(path.dirname(configPath), { recursive: true })
  fs.writeFileSync(temporaryPath, JSON.stringify(normalised, null, 2), 'utf8')
  fs.renameSync(temporaryPath, configPath)

  return normalised
}

export function establishDesktopUpdateTrack(config: DesktopUpdateConfig): DesktopUpdateConfig {
  if (config.track !== 'release' || config.trackSource !== 'default') {
    return config
  }

  return { ...config, trackSource: 'established' }
}

function loadDesktopUpdateConfig({
  configPath,
  installationAlreadyExisted,
  defaultBranch = DEFAULT_DESKTOP_UPDATE_BRANCH
}: LoadDesktopUpdateConfigOptions): DesktopUpdateConfig {
  let config: DesktopUpdateConfig | null = null

  try {
    config = parseDesktopUpdateConfig(fs.readFileSync(configPath, 'utf8'), defaultBranch)
  } catch {
    config = null
  }

  if (!config) {
    config = {
      schemaVersion: DESKTOP_UPDATE_CONFIG_SCHEMA_VERSION,
      track: installationAlreadyExisted ? 'main' : 'release',
      trackSource: installationAlreadyExisted ? 'migration' : 'default',
      branch: defaultBranch
    }
  }

  // Persist migrations and first-run inference immediately. In particular,
  // this must happen before desktop-installation.json is created or a fresh
  // release-track install would be misclassified as an existing install on
  // its second launch.
  return writeDesktopUpdateConfig(configPath, config)
}

export { loadDesktopUpdateConfig, parseDesktopUpdateConfig, writeDesktopUpdateConfig }
