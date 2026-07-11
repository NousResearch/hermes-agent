// Regression guards for Windows `hermes` resolution in main.ts.
//
// main.ts has no module.exports, so these follow the repo's source-assertion
// test pattern (see windows-child-process.test.ts). They pin the two Windows
// resolution bugs that caused desktop reinstall loops:
//   1. findOnPath() tried the empty extension FIRST, so an extensionless
//      Git-Bash `hermes` shim shadowed the real hermes.cmd/hermes.exe; the
//      shim then failed the --version probe and the desktop fell through to a
//      spurious bootstrap/repair.
//   2. handOffWindowsBootstrapRecovery() chose --update vs the destructive
//      --repair by checking ONLY venv\Scripts\hermes.exe (the console-script
//      shim, written at the END of venv setup and absent in interrupted
//      states), so it escalated to a full venv recreate even on healthy
//      installs.
//   3. unwrapWindowsVenvHermesCommand() returned the venv python with NO
//      runtime probe (bypassing the caller's --version check too), so a venv
//      broken mid-update (e.g. missing python-dotenv) was re-selected forever:
//      Retry / "Repair install" resolved the same dead interpreter instead of
//      falling through to the bootstrap installer.

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

function readMain() {
  return fs.readFileSync(path.join(__dirname, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')
}

test('findOnPath tries PATHEXT extensions before the bare (empty) name on Windows', () => {
  const source = readMain()
  // Fixed order: PATHEXT first, empty string LAST.
  assert.match(
    source,
    /\(process\.env\.PATHEXT \|\| '\.COM;\.EXE;\.BAT;\.CMD'\)\.split\(';'\)\.filter\(Boolean\), ''\]/,
    'extensions array must end with the empty string, not start with it'
  )
  // The buggy empty-first order must not return.
  assert.doesNotMatch(
    source,
    /\['', \.\.\.\(process\.env\.PATHEXT/,
    'empty-extension-first order regressed: an extensionless shim can shadow hermes.cmd/.exe'
  )
})

test('Windows bootstrap recovery chooses --update when any real-install signal is present', () => {
  const source = readMain()
  assert.match(source, /const haveRealInstall =/, 'recovery must compute haveRealInstall')
  assert.match(source, /fileExists\(venvPython\)/, 'recovery must accept the venv interpreter as a real-install signal')
  assert.match(
    source,
    /\.hermes-bootstrap-complete/,
    'recovery must accept the bootstrap-complete marker as a real-install signal'
  )
  assert.match(source, /updaterArgs = haveRealInstall \? \['--update'/, 'updaterArgs must gate on haveRealInstall')
  // The old too-narrow check (only venv\Scripts\hermes.exe) must not return.
  assert.doesNotMatch(
    source,
    /updaterArgs = fileExists\(venvHermes\) \?/,
    'recovery regressed to gating only on the hermes.exe shim, which forces destructive --repair'
  )
})

test('unwrapWindowsVenvHermesCommand smoke-tests the venv python before trusting it', () => {
  const source = readMain()
  const fnStart = source.indexOf('function unwrapWindowsVenvHermesCommand(')
  assert.notEqual(fnStart, -1, 'unwrapWindowsVenvHermesCommand must exist in main.ts')
  // Slice out just the function body (up to the next top-level function decl)
  const fnEnd = source.indexOf('\nfunction ', fnStart + 1)
  const body = source.slice(fnStart, fnEnd === -1 ? undefined : fnEnd)
  assert.match(
    body,
    /canImportHermesCli\(python/,
    'unwrap must probe the venv interpreter; returning it unprobed re-selects a broken venv ' +
      'forever (Retry/Repair loop on a mid-update venv missing e.g. python-dotenv)'
  )
  assert.match(body, /effectivePython/, 'unwrap must prefer base Python when pyvenv.cfg has a home key')
  assert.match(
    body,
    /return null\s*\n\s*\}\s*\n\s*return \{/,
    'a failed probe must fall through (return null) so the resolver reaches the bootstrap rung'
  )
})

test('resolveBasePythonFromVenvCfg reads pyvenv.cfg home and resolves base Python', () => {
  const source = readMain()
  assert.match(
    source,
    /function resolveBasePythonFromVenvCfg/,
    'resolveBasePythonFromVenvCfg must exist in main.ts'
  )
  // Must parse the `home` key from pyvenv.cfg.
  // Use indexOf to verify the regex pattern presence without regex-in-regex escaping.
  const fnStart = source.indexOf('function resolveBasePythonFromVenvCfg')
  const fnEnd = source.indexOf('\nfunction ', fnStart + 1)
  const body = source.slice(fnStart, fnEnd === -1 ? undefined : fnEnd)
  assert.ok(body.includes("cfg.match(/^home\\s*=\\s*(.+)$/im)"), "must parse the `home` key from pyvenv.cfg")
  assert.ok(body.includes("path.join(baseDir, 'python.exe')"), "must join baseDir with python.exe on Windows")
  assert.ok(body.includes('return null') && body.includes('catch'), "must return null on errors to fall back gracefully")
})

test('createActiveBackend calls resolveBasePythonFromVenvCfg on Windows', () => {
  const source = readMain()
  const fnStart = source.indexOf('function createActiveBackend(')
  assert.notEqual(fnStart, -1, 'createActiveBackend must exist in main.ts')
  const fnEnd = source.indexOf('\nfunction ', fnStart + 1)
  const body = source.slice(fnStart, fnEnd === -1 ? undefined : fnEnd)
  assert.match(
    body,
    /const basePython = IS_WINDOWS \? resolveBasePythonFromVenvCfg\(VENV_ROOT\) : null/,
    'createActiveBackend must call resolveBasePythonFromVenvCfg on Windows'
  )
  assert.match(
    body,
    /const command = basePython \|\| \(fileExists\(venvPython\) \? venvPython : findSystemPython\(\)\)/,
    'createActiveBackend must use basePython when available, falling back to venvPython then system Python'
  )
})
