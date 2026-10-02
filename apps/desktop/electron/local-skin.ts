// Read the local display skin without asking the active gateway. This is used
// at renderer boot so a Desktop window can still paint its configured local
// skin when its primary connection is a remote gateway that is offline.
import fs from 'node:fs'
import path from 'node:path'

import type { HermesSkin, SkinColors } from '@hermes/shared/skin'
import { parse } from 'yaml'

const PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/
const SKIN_FILE_NAME_RE = /^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$/
const MAX_CONFIG_BYTES = 1_000_000
const MAX_SKIN_BYTES = 256_000

// Custom skin files are overlays in hermes_cli/skin_engine.py. Keep this in
// step with its default `colors` block so a partial local skin paints exactly
// like the resolved gateway payload when Desktop starts offline.
const DEFAULT_SKIN_COLORS: SkinColors = {
  banner_border: '#CD7F32',
  banner_title: '#FFD700',
  banner_accent: '#FFBF00',
  banner_dim: '#B8860B',
  banner_text: '#FFF8DC',
  ui_accent: '#FFBF00',
  ui_label: '#DAA520',
  ui_ok: '#4caf50',
  ui_error: '#ef5350',
  ui_warn: '#ffa726',
  prompt: '#FFF8DC',
  input_rule: '#CD7F32',
  response_border: '#FFD700',
  status_bar_bg: '#1a1a2e',
  status_bar_text: '#C0C0C0',
  status_bar_strong: '#FFD700',
  status_bar_dim: '#8A7A4A',
  status_bar_good: '#8FBC8F',
  status_bar_warn: '#FFD700',
  status_bar_bad: '#FF8C00',
  status_bar_critical: '#FF6B6B',
  session_label: '#DAA520',
  session_border: '#8B8682',
  completion_menu_bg: '#1a1a2e',
  completion_menu_current_bg: '#333355',
  selection_bg: '#3a3a55',
  shell_dollar: '#4dabf7',
  voice_status_bg: '#1a1a2e'
}

type UnknownRecord = Record<string, unknown>

const isRecord = (value: unknown): value is UnknownRecord =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value)

const readText = (filePath: string, maxBytes: number): string | null => {
  try {
    const stat = fs.statSync(filePath)

    if (!stat.isFile() || stat.size > maxBytes) {
      return null
    }

    return fs.readFileSync(filePath, 'utf8')
  } catch {
    return null
  }
}

const parseRecord = (source: string | null): UnknownRecord | null => {
  if (!source) {
    return null
  }

  try {
    const parsed = parse(source, { maxAliasCount: 20 })

    return isRecord(parsed) ? parsed : null
  } catch {
    return null
  }
}

const text = (value: unknown): string | null => {
  if (typeof value !== 'string') {
    return null
  }

  const trimmed = value.trim()

  return trimmed && trimmed.length <= 128 ? trimmed : null
}

const stringMap = (value: unknown): SkinColors | undefined => {
  if (!isRecord(value)) {
    return undefined
  }

  const entries: SkinColors = {}

  for (const [key, entry] of Object.entries(value)) {
    if (key.length <= 128 && typeof entry === 'string' && entry.length <= 1_024) {
      entries[key] = entry
    }
  }

  return entries
}

/** The profile key mirrors `hermes --profile <name>` and is safe for a path. */
export function localSkinProfileKey(profile: null | string | undefined): string {
  const name = typeof profile === 'string' ? profile.trim() : ''

  return name && name !== 'default' && PROFILE_NAME_RE.test(name) ? name : 'default'
}

/** The profile home mirrors `hermes --profile <name>`: `<root>/profiles/<name>`. */
export function localSkinHome(hermesHome: string, profile: null | string): string {
  const name = localSkinProfileKey(profile)

  return name === 'default' ? hermesHome : path.join(hermesHome, 'profiles', name)
}

function skinFromFile(filePath: string, configuredName: string): HermesSkin | null {
  const parsed = parseRecord(readText(filePath, MAX_SKIN_BYTES))

  if (!parsed) {
    return null
  }

  const name = text(parsed.name) ?? configuredName
  const description = text(parsed.description)
  const colors = stringMap(parsed.colors)
  const darkColors = stringMap(parsed.dark_colors)
  const lightColors = stringMap(parsed.light_colors)

  return {
    name,
    ...(description ? { description } : {}),
    // The backend emits this skin after merging its base palette. Do the same
    // here because custom YAMLs are allowed to only override one token.
    colors: { ...DEFAULT_SKIN_COLORS, ...colors },
    ...(darkColors ? { dark_colors: darkColors } : {}),
    ...(lightColors ? { light_colors: lightColors } : {})
  }
}

/**
 * Return the configured local skin as a small renderer-safe payload. The
 * renderer never gets arbitrary file paths or config contents, and a broken
 * local config simply leaves the normal desktop theme in place.
 */
export function readLocalDisplaySkin(hermesHome: string, profile: null | string): HermesSkin | null {
  const home = localSkinHome(hermesHome, profile)
  const config = parseRecord(readText(path.join(home, 'config.yaml'), MAX_CONFIG_BYTES))
  const display = config && isRecord(config.display) ? config.display : null
  const configuredName = text(display?.skin)

  if (!configuredName || !SKIN_FILE_NAME_RE.test(configuredName)) {
    return null
  }

  // Built-ins do not have a local YAML file. The renderer already owns those
  // palettes, so the name alone is enough for the boot fallback.
  const skinsRoot = path.join(home, 'skins')
  const filePath = path.join(skinsRoot, `${configuredName}.yaml`)

  try {
    const realRoot = fs.realpathSync(skinsRoot)
    const realFile = fs.realpathSync(filePath)

    // Do not turn a symlink in skins/ into a generic file-reading bridge for
    // renderer code. Custom skins are direct children of the local skins root.
    if (path.dirname(realFile) !== realRoot) {
      return { name: configuredName }
    }
  } catch {
    return { name: configuredName }
  }

  return skinFromFile(filePath, configuredName) ?? { name: configuredName }
}

/** The one small payload preload needs for a window's first theme paint. */
export function readLocalSkinPayload(
  hermesHome: string,
  routedProfile: null | string | undefined,
  fallbackProfile: null | string | undefined
): { profile: string; skin: HermesSkin } | null {
  const profile = localSkinProfileKey(routedProfile ?? fallbackProfile)
  const skin = readLocalDisplaySkin(hermesHome, profile)

  return skin ? { profile, skin } : null
}
