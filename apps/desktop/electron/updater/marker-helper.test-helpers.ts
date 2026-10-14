/** A checkout with a REAL fake hand-off script that answers marker-helper ops. */

import fs from 'fs'
import os from 'os'
import path from 'path'

const roots: string[] = []

/** Call from afterEach. */
export function cleanupFakeCheckouts(): void {
  for (const root of roots.splice(0)) {
    fs.rmSync(root, { recursive: true, force: true })
  }
}

/** A scratch directory removed by cleanupFakeCheckouts. */
export function scratchRoot(tag: string): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), `${tag}-`))
  roots.push(root)

  return root
}

/**
 * Text of a REAL fake hand-off script: it records its argv to
 * `$HERMES_HOME/helper-calls.log`, prints the verdict staged in
 * `$HERMES_HOME/helper-verdict`, exits with `$HERMES_HOME/helper-exit`
 * (default 0), and — like the real helper — removes the marker when it
 * answers `reclaimed`/`withdrawn`.
 */
export function fakeHelperScript(protocolLine = '# hermes-handoff-protocol: 2'): string {
  return [
    '#!/usr/bin/env bash',
    protocolLine,
    'printf "%s\\n" "$*" >> "$HERMES_HOME/helper-calls.log"',
    'if [ "$1" != "--marker-op" ]; then exit 3; fi',
    '[ -f "$HERMES_HOME/helper-sleep" ] && sleep "$(cat "$HERMES_HOME/helper-sleep")"',
    'verdict="$(cat "$HERMES_HOME/helper-verdict" 2>/dev/null)"',
    'case "$verdict" in reclaimed|withdrawn) rm -f "$HERMES_HOME/.hermes-update-in-progress" ;; esac',
    'printf "%s\\n" "$verdict"',
    'exit "$(cat "$HERMES_HOME/helper-exit" 2>/dev/null || echo 0)"',
    ''
  ].join('\n')
}

/** A checkout whose posix.sh is `fakeHelperScript`. */
export function fakeHelperCheckout(protocolLine = '# hermes-handoff-protocol: 2'): { root: string; home: string } {
  const root = scratchRoot('marker-helper')
  const home = path.join(root, 'home')
  fs.mkdirSync(home)
  fs.mkdirSync(path.join(root, 'scripts', 'desktop-update'), { recursive: true })
  fs.writeFileSync(path.join(root, 'scripts', 'desktop-update', 'posix.sh'), fakeHelperScript(protocolLine))

  return { root, home }
}
