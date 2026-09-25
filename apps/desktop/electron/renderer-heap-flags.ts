// Renderer heap ceiling + `desktop.electron_flags` for packaged launches (#77311).
//
// Chromium only forwards `--js-flags` to renderer processes when the BROWSER
// process has the switch on its own command line before `ready`. The
// `hermes desktop` launcher appends `desktop.electron_flags` to argv, but a
// packaged app started from its Start-menu / .desktop entry never runs that
// launcher, so the config key was dead there and the renderer had no heap
// ceiling at all. This module is pure (no Electron import) so the switch
// planning is vitest-able; main.ts applies the result via app.commandLine.

export interface DesktopLaunchConfig {
  electronFlags: string[]
  rendererMaxOldSpaceMb: number
  /** `desktop.renderer_accessibility`; unset when the key is absent or unreadable. */
  rendererAccessibility?: boolean
}

export interface PlannedSwitch {
  name: string
  value?: string
}

/**
 * Read the two launch keys out of `config.yaml` text without a YAML parser
 * (the main bundle ships none). Deliberately narrow: a top-level `desktop:`
 * mapping with two-space-indented `electron_flags` (flow list, block list with
 * four-space `- ` items, or one string) and `renderer_max_old_space_mb`
 * (integer). Anything else keeps the defaults — a malformed config must never
 * block the launch — but a `desktop:` block that clearly MEANT to set one of
 * them and produced nothing warns once, because the failure is otherwise
 * invisible (the app just launches with no heap ceiling). The supported shape
 * is documented in `website/docs/user-guide/desktop.md`.
 */
export function readDesktopLaunchConfig(yamlText: string): DesktopLaunchConfig {
  const out: DesktopLaunchConfig = { electronFlags: [], rendererMaxOldSpaceMb: 0 }
  const lines = String(yamlText ?? '').split(/\r?\n/)
  const start = lines.findIndex(line => /^desktop:\s*(#.*)?$/.test(line))

  if (start < 0) {
    return out
  }

  const unquote = (raw: string) => raw.trim().replace(/^(['"])(.*)\1$/, '$2')

  const blockLines: string[] = []

  const splitFlow = (raw: string) => raw.slice(1, -1).split(',').map(unquote).filter(Boolean)

  for (let i = start + 1; i < lines.length; i += 1) {
    const line = lines[i]

    if (/^\S/.test(line)) {
      break // next top-level key
    }

    blockLines.push(line)

    const keyed = /^ {2}([a-z_]+):\s*(.*?)\s*$/.exec(line)

    if (!keyed) {
      continue
    }

    const [, key, rawValue] = keyed
    const value = rawValue.replace(/\s+#.*$/, '')

    if (key === 'renderer_accessibility') {
      const word = unquote(value).toLowerCase()

      if (['0', 'false', 'no', 'off'].includes(word)) {
        out.rendererAccessibility = false
      } else if (['1', 'true', 'yes', 'on'].includes(word)) {
        out.rendererAccessibility = true
      }
    } else if (key === 'renderer_max_old_space_mb') {
      const mb = Number.parseInt(unquote(value), 10)
      out.rendererMaxOldSpaceMb = Number.isFinite(mb) && mb > 0 ? mb : 0
    } else if (key === 'electron_flags') {
      if (value.startsWith('[') && value.endsWith(']')) {
        out.electronFlags = splitFlow(value)
      } else if (value) {
        out.electronFlags = unquote(value).split(/\s+/).filter(Boolean)
      } else {
        const items: string[] = []

        for (let j = i + 1; j < lines.length; j += 1) {
          const item = /^ {4}- (.*)$/.exec(lines[j])

          if (!item) {
            break
          }

          items.push(unquote(item[1]))
        }

        out.electronFlags = items.filter(Boolean)
      }
    }
  }

  if (out.electronFlags.length === 0 && out.rendererMaxOldSpaceMb === 0) {
    // The parser is a deliberate subset, so a valid-YAML-but-unsupported shape
    // (any indentation other than two-space keys / four-space `- ` items)
    // yields nothing at all. Silence there reads as "the knob does not work";
    // one line naming the supported shape is the difference between a five
    // minute fix and a bug report.
    const meantToSet = blockLines.some(line => /(electron_flags|renderer_max_old_space_mb)\s*:/.test(line))

    if (meantToSet) {
      console.warn(
        '[hermes] config.yaml: desktop.electron_flags / desktop.renderer_max_old_space_mb were ignored — ' +
          'the launch reader supports only two-space-indented keys under a top-level `desktop:` with ' +
          'four-space `- ` list items (see the Desktop docs). Launching with Chromium defaults.'
      )
    }
  }

  return out
}

/**
 * Plan the `app.commandLine.appendSwitch` calls for a launch.
 *
 * Every `--js-flags` source (an existing switch on `argv`, entries in
 * `electron_flags`, and the typed `renderer_max_old_space_mb`) is MERGED into a
 * single `js-flags` switch — appendSwitch replaces, so applying them one by
 * one would silently drop all but the last. Other `--name[=value]` entries
 * pass through as their own switches. Flags already present on `argv` are not
 * re-applied (the `hermes desktop` launcher put them there).
 */
export function planLaunchSwitches(cfg: DesktopLaunchConfig, argv: readonly string[] = []): PlannedSwitch[] {
  const jsFlagParts: string[] = []
  const others: PlannedSwitch[] = []
  const argvSwitches = new Set<string>()

  const consider = (flag: string, fromArgv: boolean) => {
    const match = /^--([^=\s]+)(?:=(.*))?$/.exec(flag.trim())

    if (!match) {
      return
    }

    const [, name, value] = match

    if (name === 'js-flags') {
      jsFlagParts.push(
        ...String(value ?? '')
          .split(/\s+/)
          .filter(Boolean)
      )

      return
    }

    if (fromArgv) {
      argvSwitches.add(name)
    } else if (!argvSwitches.has(name)) {
      others.push(value === undefined ? { name } : { name, value })
    }
  }

  for (let i = 0; i < argv.length; i += 1) {
    const arg = String(argv[i] ?? '').trim()
    const next = i + 1 < argv.length ? String(argv[i + 1] ?? '').trim() : ''
    // `--name value` is as valid on a command line as `--name=value`. Scanning
    // the two tokens independently dropped the value: `--js-flags
    // --max-old-space-size=4096` registered a bare `js-flags` with no parts
    // and a bogus `max-old-space-size` switch, so the config's ceiling
    // replaced the launcher's instead of merging with it. `--js-flags` always
    // takes a value (its own value is dash-prefixed), any other switch pairs
    // only with a non-flag token.
    const bareName = /^--([^=\s]+)$/.exec(arg)?.[1]

    if (bareName && next && (bareName === 'js-flags' || !next.startsWith('-'))) {
      consider(`${arg}=${next}`, true)
      i += 1

      continue
    }

    consider(arg, true)
  }

  for (const flag of cfg.electronFlags) {
    consider(flag, false)
  }

  if (cfg.rendererMaxOldSpaceMb > 0 && !jsFlagParts.some(part => part.startsWith('--max-old-space-size='))) {
    jsFlagParts.push(`--max-old-space-size=${Math.floor(cfg.rendererMaxOldSpaceMb)}`)
  }

  const planned = [...others]

  if (jsFlagParts.length > 0) {
    planned.push({ name: 'js-flags', value: Array.from(new Set(jsFlagParts)).join(' ') })
  }

  return planned
}
