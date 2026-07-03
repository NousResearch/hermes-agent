'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ELECTRON_DIR = __dirname

function readElectronFile(name) {
  return fs.readFileSync(path.join(ELECTRON_DIR, name), 'utf8').replace(/\r\n/g, '\n')
}

// ---------------------------------------------------------------------------
// primaryProfileKey() must fall back to the CLI's sticky ~/.hermes/active_profile
// before defaulting to "default", so a profile picked via `hermes profile use`
// is honored on the desktop's first launch even when the desktop has no
// stored preference of its own (#57757).
// ---------------------------------------------------------------------------

test('readCliActiveProfile reads active_profile from HERMES_HOME', () => {
  const source = readElectronFile('main.cjs')

  const fnStart = source.indexOf('function readCliActiveProfile(')
  assert.notEqual(fnStart, -1, 'readCliActiveProfile function not found')

  const fnBody = source.slice(fnStart, fnStart + 500)

  assert.match(
    fnBody,
    /path\.join\(HERMES_HOME, 'active_profile'\)/,
    'should read the active_profile file from HERMES_HOME, mirroring hermes_cli/profiles.py get_active_profile()'
  )

  // Missing/unreadable file must resolve to null, not throw — the CLI
  // treats "no file" as "default", and callers OR this into a fallback chain.
  assert.match(fnBody, /catch/, 'must swallow a missing/unreadable file')
  assert.match(fnBody, /return null/, 'must return null when no valid preference is found')
})

test('primaryProfileKey falls back through desktop preference, then CLI preference, then default', () => {
  const source = readElectronFile('main.cjs')

  const fnStart = source.indexOf('function primaryProfileKey(')
  assert.notEqual(fnStart, -1, 'primaryProfileKey function not found')

  const fnBody = source.slice(fnStart, fnStart + 200)

  assert.match(
    fnBody,
    /readActiveDesktopProfile\(\)\s*\|\|\s*readCliActiveProfile\(\)\s*\|\|\s*'default'/,
    'primaryProfileKey should try the desktop preference, then the CLI sticky preference, then "default"'
  )
})

test('startHermes pins --profile using primaryProfileKey, not just the desktop-local preference', () => {
  const source = readElectronFile('main.cjs')

  const fnStart = source.indexOf('async function startHermes(')
  assert.notEqual(fnStart, -1, 'startHermes function not found')

  const fnEnd = source.indexOf('\nasync function', fnStart + 1)
  const fnBody = source.slice(fnStart, fnEnd === -1 ? fnStart + 4000 : fnEnd)

  assert.match(
    fnBody,
    /const activeProfile = primaryProfileKey\(\)/,
    'startHermes should resolve the profile to launch via primaryProfileKey(), so it agrees with ' +
      'the routing decisions in ensureBackend()/resolveRemoteBackend() elsewhere in the file'
  )

  assert.match(
    fnBody,
    /if \(activeProfile !== 'default'\)/,
    'should only pass --profile when it resolves to something other than "default"'
  )
})

test('hermes:profile:get IPC handler reports primaryProfileKey, not just the desktop-local preference', () => {
  const source = readElectronFile('main.cjs')

  // The renderer's boot sequence (use-gateway-boot.ts) calls this handler to
  // seed $activeGatewayProfile, which drives what the profile switcher
  // displays. If this handler reports readActiveDesktopProfile() directly, the
  // switcher shows "default" on first launch even though the backend actually
  // booted into the CLI's sticky active_profile — the bug looks fixed
  // server-side but is still visibly broken in the UI.
  assert.match(
    source,
    /ipcMain\.handle\('hermes:profile:get', async \(\) => \(\{ profile: primaryProfileKey\(\) \}\)\)/,
    'hermes:profile:get should report primaryProfileKey(), which the renderer boot sequence trusts ' +
      'as the profile the primary backend came up as'
  )
})
