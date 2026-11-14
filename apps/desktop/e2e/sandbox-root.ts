// sandbox-root.ts — where an e2e sandbox may live.
//
// A sandbox root must NOT sit under a Hermes `<home>/profiles/<name>` tree. The gateway resolves
// profile ownership by walking a path's parents for a real `profiles/` root
// (`hermes_constants.named_profile_home`), so a sandbox under
// `~/.hermes/profiles/<bot>/cache/…` reads as that bot's own profile home: `hermes gateway ensure`
// then treats the cache dir as the host root, starts a gateway there that records
// "only one profile exists (nothing to multiplex)", and answers the app with the parked verdict
// `Gateway inaccessible (profile_parked)` — the app never reaches ready and every boot assertion
// times out on a product that is fine.
//
// Hermes agent sessions set TMPDIR to exactly that kind of path (their profile scratch dir), so the
// default `os.tmpdir()` cannot be trusted. Pure node, no playwright: proven by
// sandbox-root.unit.test.ts.

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

/** Files that mark a directory as a real Hermes home (not merely one named `profiles`). */
const HERMES_HOME_MARKERS = ['config.yaml', '.env', 'state.db']

function looksLikeHermesHome(dir: string): boolean {
  if (path.basename(dir) === '.hermes') {
    return true
  }

  return HERMES_HOME_MARKERS.some(marker => {
    try {
      return fs.statSync(path.join(dir, marker)).isFile()
    } catch {
      return false
    }
  })
}

/** True when *candidate* is `<hermes home>/profiles/<name>` or under it. */
export function insideHermesProfileHome(candidate: string): boolean {
  let dir = path.resolve(candidate)

  for (;;) {
    const parent = path.dirname(dir)

    if (
      path.basename(parent) === 'profiles'
      && !path.basename(dir).startsWith('.')
      && looksLikeHermesHome(path.dirname(parent))
    ) {
      return true
    }

    if (parent === dir) {
      return false
    }

    dir = parent
  }
}

/** A base directory that is not a profile home, or *candidate* when it is already safe. */
function neutralBase(): string {
  const override = process.env.HERMES_E2E_SANDBOX_ROOT

  if (override && path.isAbsolute(override)) {
    return override
  }

  return process.platform === 'win32' ? os.tmpdir() : '/tmp'
}

/**
 * The directory e2e sandboxes are created in: `$HERMES_E2E_SANDBOX_ROOT`, else *candidate*
 * (`os.tmpdir()` by default) — replaced by a neutral temp root when the candidate is a profile home.
 */
export function sandboxBaseDir(candidate: string = os.tmpdir()): string {
  const resolved = path.resolve(candidate)

  return insideHermesProfileHome(resolved) ? neutralBase() : resolved
}
