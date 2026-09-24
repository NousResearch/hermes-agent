// The active skin read straight off the LOCAL disk (#118942).
//
// The renderer otherwise learns its skin only from a live gateway
// (`gateway.ready` / `skin.changed`), so a launch whose primary connection is
// unreachable never saw `display.skin` and painted the built-in default. This
// mirrors the part of `hermes_cli/skin_engine.py` the desktop converter reads —
// `display.skin` → `skins/<name>.yaml` → `colors` merged over the default
// skin's — so the ThemeProvider can paint it before (or without) a gateway.
// The gateway stays authoritative the moment it speaks.
//
// Pure (no Electron import) so it is vitest-able. Like
// `readDesktopLaunchConfig` it parses without a YAML library (the main bundle
// ships none), and deliberately narrowly: anything outside plain block
// mappings of scalars returns null rather than a guessed palette.

import fs from 'node:fs'
import path from 'node:path'

export interface LocalSkin {
  name: string
  colors?: Record<string, string>
}

// `_BUILTIN_SKINS["default"]["colors"]` in hermes_cli/skin_engine.py, which
// `_build_skin_config` merges under every skin's own `colors`. Kept in lockstep
// by local-skin.test.ts, which reads the Python source.
export const DEFAULT_SKIN_COLORS: Readonly<Record<string, string>> = {
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

const UNPARSEABLE = Symbol('unparseable')

type Scalar = null | string | typeof UNPARSEABLE

const isSkippable = (line: string) => /^\s*(#.*)?$/.test(line) || /^(---|\.\.\.)\s*$/.test(line)

/** A YAML scalar as PyYAML's safe_load would see it, for the shapes skins use. */
function parseScalar(raw: string): Scalar {
  const text = raw.trim()

  if (!text || text.startsWith('#')) {
    return null // empty, or a bare `#abc` — YAML reads that as a comment
  }

  const single = /^'((?:[^']|'')*)'\s*(#.*)?$/.exec(text)

  if (single) {
    return single[1].replace(/''/g, "'")
  }

  const double = /^"([^"\\]*)"\s*(#.*)?$/.exec(text)

  if (double) {
    return double[1]
  }

  if (/^["'{[&*!|>]/.test(text)) {
    return UNPARSEABLE // escapes, flow collections, anchors, tags, block scalars
  }

  const plain = text.replace(/\s+#.*$/, '').trim()

  return plain === '~' || plain === 'null' || plain === 'Null' || plain === 'NULL' ? null : plain
}

/**
 * Top-level `key: value` pairs, plus the children of the requested block
 * mappings. Lines outside those (other keys' nested values, PyYAML's
 * indentless `- item` sequences, deeper nesting) are skipped; a requested
 * mapping whose indentation breaks returns null.
 */
function parseTopLevel(
  text: string,
  blocks: readonly string[]
): null | { scalars: Map<string, Scalar>; mappings: Map<string, Map<string, Scalar>> } {
  const scalars = new Map<string, Scalar>()
  const mappings = new Map<string, Map<string, Scalar>>()
  let open: Map<string, Scalar> | null = null
  let childIndent = 0

  for (const line of text.replace(/^\uFEFF/, '').split(/\r?\n/)) {
    if (isSkippable(line)) {
      continue
    }

    if (/^\S/.test(line)) {
      open = null
      const keyed = /^([A-Za-z_][\w-]*):(?:\s+(.*))?$/.exec(line)

      if (!keyed) {
        continue // a sequence item or a key shape we never read
      }

      const [, key, rest = ''] = keyed

      if (blocks.includes(key) && parseScalar(rest) === null) {
        open = new Map()
        childIndent = 0
        mappings.set(key, open)
      } else {
        scalars.set(key, parseScalar(rest))
      }

      continue
    }

    if (!open) {
      continue // nested under a key we don't read
    }

    const indent = line.length - line.trimStart().length

    if (!childIndent) {
      childIndent = indent
    }

    if (indent < childIndent) {
      return null
    }

    const child = /^\s+([A-Za-z_][\w-]*):(?:\s+(.*))?$/.exec(line)

    if (indent === childIndent && child) {
      open.set(child[1], parseScalar(child[2] ?? ''))
    }
  }

  return { scalars, mappings }
}

/** `display.skin` from config.yaml text; `default` when unset (as init_skin_from_config). */
export function readConfiguredSkinName(configYaml: string): null | string {
  const parsed = parseTopLevel(configYaml, ['display'])

  if (!parsed) {
    return null
  }

  const skin = parsed.mappings.get('display')?.get('skin')

  if (skin === UNPARSEABLE) {
    return null
  }

  return typeof skin === 'string' && skin.trim() ? skin.trim() : 'default'
}

/**
 * A user skin file → the payload `resolve_skin()` would send for it (name +
 * default-merged colors). Null when the file is not a skin `_load_skin_from_yaml`
 * accepts, or not one this narrow parser can read faithfully.
 */
export function readSkinFile(skinYaml: string): LocalSkin | null {
  const parsed = parseTopLevel(skinYaml, ['colors'])
  const name = parsed?.scalars.get('name')

  if (!parsed || typeof name !== 'string' || !name.trim()) {
    return null
  }

  const colors: Record<string, string> = { ...DEFAULT_SKIN_COLORS }
  const own = parsed.mappings.get('colors')

  if (parsed.scalars.get('colors') === UNPARSEABLE) {
    return null
  }

  for (const [key, value] of own ?? []) {
    if (value === UNPARSEABLE) {
      return null
    }

    if (value === null) {
      delete colors[key]
    } else {
      colors[key] = value
    }
  }

  return { name: name.trim(), colors }
}

/**
 * The skin the local backend would resolve for `profile`, read from disk.
 * A user file wins (as `load_skin`); otherwise only the name is returned and
 * the renderer paints it if it has a palette for it (a desktop built-in or a
 * cached backend skin). Null when config.yaml is unreadable as YAML we
 * understand, or the skin is `default`.
 */
export function readLocalSkin(hermesHome: string, profile: null | string): LocalSkin | null {
  const home = profile && profile !== 'default' ? path.join(hermesHome, 'profiles', profile) : hermesHome

  const read = (file: string) => {
    try {
      return fs.readFileSync(file, 'utf8')
    } catch {
      return null
    }
  }

  const config = read(path.join(home, 'config.yaml'))
  const name = config === null ? 'default' : readConfiguredSkinName(config)

  if (!name || name === 'default' || /[\\/]/.test(name)) {
    return null
  }

  const userFile = read(path.join(home, 'skins', `${name}.yaml`))

  // A file we can't read faithfully degrades to the bare name: the renderer then
  // paints a built-in or its cached palette for it, exactly as before this path.
  return (userFile === null ? null : readSkinFile(userFile)) ?? { name }
}
