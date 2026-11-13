/**
 * The lockstep test for the three hand-maintained display-toggle registries.
 *
 * A mirrored key works end to end only if it is registered in every one of
 * them, and missing any one dies SILENTLY: leave it out of the desktop
 * canonical gate (`canonical-protocol.ts::CANONICAL_CONFIG_KEYS`) and the
 * mirror push throws before the wire (canonicalProducerFrame refuses
 * synchronously, escaping display-toggles.ts's `.catch()`); leave it out of
 * either backend frozenset (`gateway/session_tui_config.py` and
 * `tui_gateway/server.py`, both `_DISPLAY_TOGGLE_KEYS`) and `config.set`
 * answers 4002 — the preference never lands in config.yaml. The reactions
 * toggle shipped broken for years over exactly this class of drift.
 *
 * This test enumerates the desktop side from the `mirrorDisplayToggle` call
 * sites themselves (a new store registering a mirror is covered the moment it
 * lands) and pins the set against all three registries. The backend
 * frozensets exist solely to serve these mirrors, so they must be EQUAL to the
 * mirrored set — a stale key there is drift too. The canonical gate serves
 * more than toggles, so it only has to contain them.
 */

import { readdirSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'

import { describe, expect, it } from 'vitest'

// Anchored on this file (…/apps/desktop/src/store/) rather than a process cwd,
// which different vitest invocations don't agree on.
const DESKTOP = dirname(dirname(dirname(import.meta.filename)))
const REPO = dirname(dirname(DESKTOP))

const sourceFiles = (dir: string): string[] =>
  readdirSync(dir, { withFileTypes: true, recursive: true }).flatMap(entry =>
    entry.isFile() && /\.(ts|tsx)$/.test(entry.name) && !/\.test\./.test(entry.name)
      ? [join(entry.parentPath ?? entry.path, entry.name)]
      : []
  )

// Every mirrored key, read off the registration calls — not a copy of them.
// A call site that passes anything but a literal `display.<key>` breaks this
// scan, which is deliberate: rewrite the scan before it can miss a key.
const mirroredKeys = (): string[] => {
  const keys: string[] = []

  for (const file of sourceFiles(join(DESKTOP, 'src'))) {
    const source = readFileSync(file, 'utf8')
    const calls = [...source.matchAll(/mirrorDisplayToggle\(\s*'([^']+)'/g)].map(match => match[1])
    const occurrences = (source.match(/\bmirrorDisplayToggle\s*\(/g) ?? []).length
    const definitions = (source.match(/\bfunction\s+mirrorDisplayToggle\s*\(/g) ?? []).length

    expect(
      calls.length + definitions,
      `${file}: a mirrorDisplayToggle call does not pass a literal 'display.<key>' — teach this scan its shape`
    ).toBe(occurrences)
    keys.push(...calls)
  }

  return keys
}

// The quoted strings inside one registry's declaration.
const registryKeys = (file: string, marker: string, error: string): string[] => {
  const source = readFileSync(join(REPO, file), 'utf8')
  const start = source.indexOf(marker)

  expect(start, error).toBeGreaterThanOrEqual(0)

  const end = source.indexOf(marker.endsWith('([') ? '])' : '})', start)

  return [...source.slice(start, end).matchAll(/['"]([^'"]+)['"]/g)].map(match => match[1])
}

describe('display toggle registries stay in lockstep', () => {
  const keys = mirroredKeys()

  it('finds the mirrored keys to begin with (the scan did not rot)', () => {
    expect(keys.length).toBeGreaterThanOrEqual(4)
    expect(new Set(keys).size).toBe(keys.length)
  })

  it('registers every mirrored key in all three places', () => {
    const canonical = registryKeys(
      'apps/desktop/src/api/canonical-protocol.ts',
      'const CANONICAL_CONFIG_KEYS = new Set([',
      'canonical-protocol.ts: CANONICAL_CONFIG_KEYS disappeared'
    )

    const tuiGateway = registryKeys(
      'tui_gateway/server.py',
      '_DISPLAY_TOGGLE_KEYS = frozenset(',
      'server.py: _DISPLAY_TOGGLE_KEYS disappeared'
    )

    const gateway = registryKeys(
      'gateway/session_tui_config.py',
      '_DISPLAY_TOGGLE_KEYS = frozenset(',
      'session_tui_config.py: _DISPLAY_TOGGLE_KEYS disappeared'
    )

    for (const key of keys) {
      expect(canonical, `${key} missing from the desktop canonical gate`).toEqual(
        expect.arrayContaining([key])
      )
      expect(tuiGateway, `${key} missing from tui_gateway/server.py`).toContain(key)
      expect(gateway, `${key} missing from gateway/session_tui_config.py`).toContain(key)
    }
  })

  it('keeps the backend frozensets exactly the mirrored set, so no stale keys linger there', () => {
    const mirrored = [...keys].sort()

    for (const file of ['tui_gateway/server.py', 'gateway/session_tui_config.py']) {
      const served = registryKeys(file, '_DISPLAY_TOGGLE_KEYS = frozenset(', `${file}: registry disappeared`)

      expect(served.sort(), `${file} serves keys the renderer does not mirror (or dropped one)`).toEqual(mirrored)
    }
  })
})
