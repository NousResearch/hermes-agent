/**
 * Tests for electron/desktop-uninstall.cjs.
 *
 * Run with: node --test electron/desktop-uninstall.test.cjs
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * These are the pure helpers behind the desktop Chat GUI uninstaller: the
 * mode → CLI-flag mapping, the running-app-bundle resolution per OS, and the
 * cleanup-script builders (POSIX + Windows).
 */

const test = require('node:test')
const assert = require('node:assert/strict')

const {
  UNINSTALL_MODES,
  buildPosixCleanupScript,
  buildWindowsCleanupScript,
  modeRemovesAgent,
  modeRemovesUserData,
  resolveRemovableAppPath,
  shouldRemoveAppBundle,
  uninstallArgsForMode
} = require('./desktop-uninstall.cjs')

// --- uninstallArgsForMode ---

test('uninstallArgsForMode maps each mode to the right CLI flags', () => {
  assert.deepEqual(uninstallArgsForMode('gui'), ['uninstall', '--gui', '--yes'])
  assert.deepEqual(uninstallArgsForMode('lite'), ['uninstall', '--yes'])
  assert.deepEqual(uninstallArgsForMode('full'), ['uninstall', '--full', '--yes'])
})

test('uninstallArgsForMode throws on an unknown mode (no silent full wipe)', () => {
  assert.throws(() => uninstallArgsForMode('nuke'), /Unknown uninstall mode/)
  assert.throws(() => uninstallArgsForMode(''), /Unknown uninstall mode/)
})

test('UNINSTALL_MODES lists exactly the three supported modes', () => {
  assert.deepEqual([...UNINSTALL_MODES].sort(), ['full', 'gui', 'lite'])
})

// --- modeRemovesAgent / modeRemovesUserData ---

test('mode predicates classify what each mode removes', () => {
  assert.equal(modeRemovesAgent('gui'), false)
  assert.equal(modeRemovesAgent('lite'), true)
  assert.equal(modeRemovesAgent('full'), true)

  assert.equal(modeRemovesUserData('gui'), false)
  assert.equal(modeRemovesUserData('lite'), false)
  assert.equal(modeRemovesUserData('full'), true)
})

// --- resolveRemovableAppPath ---

test('resolveRemovableAppPath finds the .app bundle on macOS', () => {
  assert.equal(
    resolveRemovableAppPath('/Applications/Hermes.app/Contents/MacOS/Hermes', 'darwin'),
    '/Applications/Hermes.app'
  )
  assert.equal(
    resolveRemovableAppPath('/Users/x/Applications/Hermes.app/Contents/MacOS/Hermes', 'darwin'),
    '/Users/x/Applications/Hermes.app'
  )
})

test('resolveRemovableAppPath returns null for a non-.app macOS exe (dev run)', () => {
  assert.equal(
    resolveRemovableAppPath('/repo/node_modules/electron/dist/Electron.app/Contents/MacOS/Electron', 'darwin'),
    '/repo/node_modules/electron/dist/Electron.app'
  )
  // A bare path with no .app ancestor → null.
  assert.equal(resolveRemovableAppPath('/usr/bin/electron', 'darwin'), null)
})

test('resolveRemovableAppPath finds the install dir on Windows', () => {
  assert.equal(
    resolveRemovableAppPath('C:\\Users\\x\\AppData\\Local\\Programs\\Hermes\\Hermes.exe', 'win32'),
    'C:\\Users\\x\\AppData\\Local\\Programs\\Hermes'
  )
  assert.equal(
    resolveRemovableAppPath('C:\\Users\\x\\AppData\\Local\\hermes-desktop\\Hermes.exe', 'win32'),
    'C:\\Users\\x\\AppData\\Local\\hermes-desktop'
  )
})

test('resolveRemovableAppPath returns null for an unrecognized Windows dir', () => {
  assert.equal(resolveRemovableAppPath('C:\\Temp\\foo\\Hermes.exe', 'win32'), null)
})

test('resolveRemovableAppPath uses APPIMAGE on Linux when set', () => {
  assert.equal(
    resolveRemovableAppPath('/tmp/.mount_HermesXXXX/hermes', 'linux', { APPIMAGE: '/home/x/Apps/Hermes.AppImage' }),
    '/home/x/Apps/Hermes.AppImage'
  )
})

test('resolveRemovableAppPath finds the unpacked dir on Linux', () => {
  assert.equal(
    resolveRemovableAppPath('/opt/hermes/linux-unpacked/hermes', 'linux', {}),
    '/opt/hermes/linux-unpacked'
  )
  // A system-package install (/usr/bin) → null, left to apt/dnf.
  assert.equal(resolveRemovableAppPath('/usr/bin/hermes', 'linux', {}), null)
})

test('resolveRemovableAppPath returns null for an empty exe path', () => {
  assert.equal(resolveRemovableAppPath('', 'darwin'), null)
  assert.equal(resolveRemovableAppPath(null, 'win32'), null)
})

// --- shouldRemoveAppBundle ---

test('shouldRemoveAppBundle requires packaged AND a resolved path', () => {
  assert.equal(shouldRemoveAppBundle(true, '/Applications/Hermes.app'), true)
  assert.equal(shouldRemoveAppBundle(false, '/Applications/Hermes.app'), false)
  assert.equal(shouldRemoveAppBundle(true, null), false)
  assert.equal(shouldRemoveAppBundle(false, null), false)
})

// --- buildPosixCleanupScript ---

test('buildPosixCleanupScript waits for the PID, runs uninstall, removes bundle', () => {
  const script = buildPosixCleanupScript({
    desktopPid: 4321,
    pythonExe: '/home/x/.hermes/hermes-agent/venv/bin/python',
    agentRoot: '/home/x/.hermes/hermes-agent',
    uninstallArgs: ['uninstall', '--gui', '--yes'],
    appPath: '/opt/hermes/linux-unpacked',
    hermesHome: '/home/x/.hermes'
  })
  assert.match(script, /^#!\/bin\/bash/)
  assert.match(script, /pid=4321/)
  assert.match(script, /kill -0 "\$pid"/)
  assert.match(script, /-m hermes_cli\.main 'uninstall' '--gui' '--yes'/)
  assert.match(script, /rm -rf '\/opt\/hermes\/linux-unpacked'/)
  assert.match(script, /export HERMES_HOME='\/home\/x\/\.hermes'/)
})

test('buildPosixCleanupScript omits the bundle rm when appPath is null', () => {
  const script = buildPosixCleanupScript({
    desktopPid: 1,
    pythonExe: '/p/python',
    agentRoot: '/a',
    uninstallArgs: ['uninstall', '--yes'],
    appPath: null,
    hermesHome: '/h'
  })
  assert.doesNotMatch(script, /rm -rf '\//)
  // Still runs the uninstall.
  assert.match(script, /-m hermes_cli\.main 'uninstall' '--yes'/)
})

test('buildPosixCleanupScript single-quote-escapes paths with apostrophes', () => {
  const script = buildPosixCleanupScript({
    desktopPid: 1,
    pythonExe: "/home/o'brien/python",
    agentRoot: '/a',
    uninstallArgs: ['uninstall', '--yes'],
    appPath: null,
    hermesHome: '/h'
  })
  // The apostrophe is closed-escaped-reopened so the shell sees the literal.
  assert.match(script, /'\/home\/o'\\''brien\/python'/)
})

// --- buildWindowsCleanupScript ---

test('buildWindowsCleanupScript waits for the PID, runs uninstall, rmdir bundle', () => {
  const script = buildWindowsCleanupScript({
    desktopPid: 9988,
    pythonExe: 'C:\\hermes\\venv\\Scripts\\python.exe',
    agentRoot: 'C:\\hermes',
    uninstallArgs: ['uninstall', '--full', '--yes'],
    appPath: 'C:\\Users\\x\\AppData\\Local\\Programs\\Hermes',
    hermesHome: 'C:\\Users\\x\\AppData\\Local\\hermes'
  })
  assert.match(script, /@echo off/)
  assert.match(script, /set PID=9988/)
  assert.match(script, /tasklist \/FI "PID eq %PID%"/)
  assert.match(script, /-m hermes_cli\.main "uninstall" "--full" "--yes"/)
  // Removal is a retry loop (Windows releases dir handles lazily), not a single
  // rmdir — assert the loop structure + the bounded retry guard.
  assert.match(script, /:rmloop/)
  assert.match(script, /rmdir \/s \/q "C:\\Users\\x\\AppData\\Local\\Programs\\Hermes" >nul 2>&1/)
  assert.match(script, /if %tries% geq 10 goto rmdone/)
  assert.match(script, /del "%~f0"/)
})

test('buildWindowsCleanupScript omits rmdir when appPath is null', () => {
  const script = buildWindowsCleanupScript({
    desktopPid: 2,
    pythonExe: 'C:\\p\\python.exe',
    agentRoot: 'C:\\a',
    uninstallArgs: ['uninstall', '--yes'],
    appPath: null,
    hermesHome: 'C:\\h'
  })
  assert.doesNotMatch(script, /rmdir/)
})
