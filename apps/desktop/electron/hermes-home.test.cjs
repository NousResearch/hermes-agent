/**
 * Tests for electron/hermes-home.cjs.
 *
 * Run with: node --test electron/hermes-home.test.cjs
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Covers the Windows HERMES_HOME selection between %LOCALAPPDATA%\hermes and a
 * legacy ~/.hermes — in particular the #40178 regression where an installer
 * pre-creating an empty LOCALAPPDATA\hermes orphaned a CLI user's sessions.
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')

const { chooseWindowsHermesHome } = require('./hermes-home.cjs')

const LOCALAPPDATA = path.join('C:\\Users\\me\\AppData\\Local', 'hermes')
const LEGACY = path.join('C:\\Users\\me', '.hermes')

// Build dir/file probes from explicit sets of existing paths.
function probes({ dirs = [], files = [] } = {}) {
  const dirSet = new Set(dirs)
  const fileSet = new Set(files)
  return {
    dirExists: p => dirSet.has(p),
    fileExists: p => fileSet.has(p)
  }
}

const legacyDb = path.join(LEGACY, 'state.db')
const localappdataDb = path.join(LOCALAPPDATA, 'state.db')

test('honours legacy ~/.hermes when it has a DB and LOCALAPPDATA was pre-created empty (#40178)', () => {
  // The installer made LOCALAPPDATA\hermes (dir exists) but no state.db there;
  // the CLI user has real data in ~/.hermes.
  const p = probes({ dirs: [LOCALAPPDATA, LEGACY], files: [legacyDb] })
  assert.equal(chooseWindowsHermesHome(LOCALAPPDATA, LEGACY, p), LEGACY)
})

test('keeps LOCALAPPDATA when it already has a DB (established desktop install)', () => {
  const p = probes({ dirs: [LOCALAPPDATA, LEGACY], files: [localappdataDb, legacyDb] })
  assert.equal(chooseWindowsHermesHome(LOCALAPPDATA, LEGACY, p), LOCALAPPDATA)
})

test('uses LOCALAPPDATA for a fresh install (no DB on either side)', () => {
  const p = probes({ dirs: [LOCALAPPDATA] })
  assert.equal(chooseWindowsHermesHome(LOCALAPPDATA, LEGACY, p), LOCALAPPDATA)
})

test('falls back to legacy when LOCALAPPDATA does not exist yet (original heuristic)', () => {
  // No LOCALAPPDATA dir, legacy dir exists with only config (no state.db yet).
  const p = probes({ dirs: [LEGACY] })
  assert.equal(chooseWindowsHermesHome(LOCALAPPDATA, LEGACY, p), LEGACY)
})

test('uses LOCALAPPDATA when only LOCALAPPDATA has a DB', () => {
  const p = probes({ dirs: [LOCALAPPDATA], files: [localappdataDb] })
  assert.equal(chooseWindowsHermesHome(LOCALAPPDATA, LEGACY, p), LOCALAPPDATA)
})

test('uses LOCALAPPDATA when both DBs exist even if LOCALAPPDATA dir-check would fail', () => {
  // Defensive: a state.db implies the dir, so LOCALAPPDATA must win here.
  const p = probes({ dirs: [LEGACY], files: [localappdataDb, legacyDb] })
  assert.equal(chooseWindowsHermesHome(LOCALAPPDATA, LEGACY, p), LOCALAPPDATA)
})
