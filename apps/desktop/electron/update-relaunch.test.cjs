/**
 * Tests for electron/update-relaunch.cjs — the pure decision + script helpers
 * behind the Linux in-app update relaunch (#45205).
 *
 * Run with: node --test electron/update-relaunch.test.cjs
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * What this locks (review acceptance criteria for PR #45205):
 *   1. The execPath split: only a binary under release/<plat>-unpacked may
 *      relaunch/claim a GUI update; AppImage/.deb/.rpm/dev/unresolved paths land
 *      on the guiSkew terminal state and do NOT claim the GUI was updated.
 *   2. Launch context is replayed on re-exec (args filtered of Electron
 *      internals; HERMES_HOME / HERMES_DESKTOP_* env + cwd preserved) and is
 *      safely shell-quoted.
 *   3. The sandbox preflight: chrome-sandbox must be root-owned + setuid to be
 *      launchable; otherwise the decision degrades to a manual terminal state
 *      (keep a working window) unless a non-interactive fallback applies.
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { execFileSync } = require('node:child_process')

const {
  unpackedDirName,
  resolveUnpackedRelease,
  decideRelaunchOutcome,
  sandboxPreflight,
  sandboxFallbackFromEnv,
  collectRelaunchArgs,
  collectRelaunchEnv,
  buildRelaunchScript,
  shellQuote
} = require('./update-relaunch.cjs')

const ROOT = '/home/u/.hermes/hermes-agent'
const UNPACKED = path.join(ROOT, 'apps', 'desktop', 'release', 'linux-unpacked')

// ---------------------------------------------------------------------------
// 1) The execPath split — the heart of the GUI/backend skew guard.
// ---------------------------------------------------------------------------

test('unpackedDirName maps platform to the electron-builder dir', () => {
  assert.equal(unpackedDirName('linux'), 'linux-unpacked')
  assert.equal(unpackedDirName('win32'), 'win-unpacked')
})

test('resolveUnpackedRelease returns the dir for a binary UNDER release/<plat>-unpacked', () => {
  const exec = path.join(UNPACKED, 'hermes')
  assert.equal(resolveUnpackedRelease(exec, ROOT, 'linux'), UNPACKED)
  // The unpacked dir itself also counts.
  assert.equal(resolveUnpackedRelease(UNPACKED, ROOT, 'linux'), UNPACKED)
})

test('resolveUnpackedRelease is null for AppImage / .deb / .rpm / dev / unresolved paths', () => {
  // AppImage mount
  assert.equal(resolveUnpackedRelease('/tmp/.mount_Hermes12345/AppRun', ROOT, 'linux'), null)
  // .deb / .rpm system install
  assert.equal(resolveUnpackedRelease('/usr/lib/hermes/hermes', ROOT, 'linux'), null)
  assert.equal(resolveUnpackedRelease('/opt/Hermes/hermes', ROOT, 'linux'), null)
  // dev electron
  assert.equal(resolveUnpackedRelease('/home/u/.hermes/hermes-agent/node_modules/electron/dist/electron', ROOT, 'linux'), null)
  // empty / missing
  assert.equal(resolveUnpackedRelease('', ROOT, 'linux'), null)
  assert.equal(resolveUnpackedRelease(path.join(UNPACKED, 'hermes'), '', 'linux'), null)
})

test('resolveUnpackedRelease is not fooled by a sibling prefix dir', () => {
  // `.../release/linux-unpacked-evil` must NOT match `.../release/linux-unpacked`.
  const sneaky = path.join(ROOT, 'apps', 'desktop', 'release', 'linux-unpacked-evil', 'hermes')
  assert.equal(resolveUnpackedRelease(sneaky, ROOT, 'linux'), null)
})

test('decideRelaunchOutcome: only under-unpacked + sandbox-ok relaunches', () => {
  assert.equal(decideRelaunchOutcome({ underUnpacked: true, sandboxOk: true }), 'relaunch')
  // Under unpacked but sandbox not launchable → manual (keep a working window).
  assert.equal(decideRelaunchOutcome({ underUnpacked: true, sandboxOk: false }), 'manual')
  // Not under unpacked → guiSkew regardless of sandbox flag.
  assert.equal(decideRelaunchOutcome({ underUnpacked: false, sandboxOk: true }), 'guiSkew')
  assert.equal(decideRelaunchOutcome({ underUnpacked: false, sandboxOk: false }), 'guiSkew')
})

// ---------------------------------------------------------------------------
// 3) Sandbox preflight
// ---------------------------------------------------------------------------

const fakeStat = (uid, mode) => () => ({ uid, mode })
const throwStat = () => {
  throw Object.assign(new Error('ENOENT'), { code: 'ENOENT' })
}

test('sandboxPreflight: root-owned + setuid is launchable', () => {
  const r = sandboxPreflight(UNPACKED, fakeStat(0, 0o4755))
  assert.equal(r.ok, true)
  assert.equal(r.reason, 'launchable')
})

test('sandboxPreflight: not root → not launchable', () => {
  const r = sandboxPreflight(UNPACKED, fakeStat(1000, 0o4755))
  assert.equal(r.ok, false)
  assert.equal(r.reason, 'not-root')
})

test('sandboxPreflight: missing setuid bit → not launchable', () => {
  const r = sandboxPreflight(UNPACKED, fakeStat(0, 0o755))
  assert.equal(r.ok, false)
  assert.equal(r.reason, 'not-setuid')
})

test('sandboxPreflight: neither root nor setuid (the fresh-rebuild trap)', () => {
  const r = sandboxPreflight(UNPACKED, fakeStat(1000, 0o755))
  assert.equal(r.ok, false)
  assert.equal(r.reason, 'not-root-not-setuid')
})

test('sandboxPreflight: no chrome-sandbox helper present → ok (build does not use SUID sandbox)', () => {
  const r = sandboxPreflight(UNPACKED, throwStat)
  assert.equal(r.ok, true)
  assert.equal(r.reason, 'no-sandbox-helper')
})

test('sandboxFallbackFromEnv: ELECTRON_DISABLE_SANDBOX / --no-sandbox make a broken sandbox safe', () => {
  assert.equal(sandboxFallbackFromEnv({ ELECTRON_DISABLE_SANDBOX: '1' }, []), true)
  assert.equal(sandboxFallbackFromEnv({ ELECTRON_DISABLE_SANDBOX: 'true' }, []), true)
  assert.equal(sandboxFallbackFromEnv({}, ['--no-sandbox']), true)
  assert.equal(sandboxFallbackFromEnv({}, ['--foo']), false)
  assert.equal(sandboxFallbackFromEnv({}, []), false)
  assert.equal(sandboxFallbackFromEnv(null, null), false)
})

// ---------------------------------------------------------------------------
// 2) Launch-context preservation
// ---------------------------------------------------------------------------

test('collectRelaunchArgs drops Electron internals, keeps user/launcher args', () => {
  const argv = [
    '--type=renderer',
    '--user-data-dir=/tmp/x',
    '--enable-features=Foo',
    '--field-trial-handle=123',
    '--no-sandbox', // sandbox opt-out — KEEP (user/env intent + relaunch fallback)
    '--lang=en-US',
    'hermes://open/agent/42', // deep link — keep
    '--profile=work', // app flag — keep
    '--remote-debugging-port=9222' // internal — drop
  ]
  assert.deepEqual(collectRelaunchArgs(argv), ['--no-sandbox', 'hermes://open/agent/42', '--profile=work'])
  assert.deepEqual(collectRelaunchArgs(undefined), [])
})

test('collectRelaunchEnv preserves HERMES_HOME + HERMES_DESKTOP_* + sandbox opt-out only', () => {
  const env = {
    HERMES_HOME: '/home/u/.hermes',
    HERMES_DESKTOP_REMOTE_URL: 'http://box:9119',
    HERMES_DESKTOP_REMOTE_TOKEN: 'secret',
    HERMES_DESKTOP_HERMES_ROOT: '/home/u/dev/hermes',
    ELECTRON_DISABLE_SANDBOX: '1', // sandbox opt-out — preserved
    PATH: '/usr/bin', // not preserved
    HOME: '/home/u', // not preserved
    UNRELATED: 'x'
  }
  assert.deepEqual(collectRelaunchEnv(env), {
    HERMES_HOME: '/home/u/.hermes',
    HERMES_DESKTOP_REMOTE_URL: 'http://box:9119',
    HERMES_DESKTOP_REMOTE_TOKEN: 'secret',
    HERMES_DESKTOP_HERMES_ROOT: '/home/u/dev/hermes',
    ELECTRON_DISABLE_SANDBOX: '1'
  })
  assert.deepEqual(collectRelaunchEnv(null), {})
})

// ---------------------------------------------------------------------------
// Generated watcher script: safe quoting + valid bash syntax.
// ---------------------------------------------------------------------------

test('shellQuote neutralizes single quotes and metacharacters', () => {
  assert.equal(shellQuote(`a'b`), `'a'\\''b'`)
  assert.equal(shellQuote('$(rm -rf /)'), `'$(rm -rf /)'`)
})

test('buildRelaunchScript embeds pid/exec/args/env/cwd and is valid bash', () => {
  const script = buildRelaunchScript({
    pid: 4242,
    execPath: '/home/u/.hermes/hermes-agent/apps/desktop/release/linux-unpacked/Hermes',
    args: ['hermes://open/agent/42', "--note=it's fine"],
    env: { HERMES_HOME: '/home/u/.hermes', HERMES_DESKTOP_REMOTE_URL: 'http://box:9119' },
    cwd: '/home/u/work dir'
  })

  // Structural assertions.
  assert.match(script, /^#!\/bin\/bash/)
  assert.match(script, /APP_PID=4242/)
  assert.match(script, /kill -9 "\$APP_PID"/)
  assert.match(script, /rm -f -- "\$0"/)
  // env exports + cwd restore + args replay are present and quoted.
  assert.match(script, /export HERMES_HOME='\/home\/u\/\.hermes'/)
  assert.match(script, /export HERMES_DESKTOP_REMOTE_URL='http:\/\/box:9119'/)
  assert.match(script, /cd '\/home\/u\/work dir'/)
  assert.match(script, /exec '.*\/linux-unpacked\/Hermes' 'hermes:\/\/open\/agent\/42' '--note=it'\\''s fine'/)

  // It must be syntactically valid bash (`bash -n`). Write to a temp file and lint.
  const tmp = path.join(os.tmpdir(), `hermes-relaunch-test-${Date.now()}.sh`)
  fs.writeFileSync(tmp, script)
  try {
    execFileSync('bash', ['-n', tmp], { stdio: 'pipe' })
  } finally {
    fs.rmSync(tmp, { force: true })
  }
})

test('buildRelaunchScript with no args/env still lints clean', () => {
  const script = buildRelaunchScript({
    pid: 1,
    execPath: '/opt/Hermes/Hermes',
    args: [],
    env: {},
    cwd: ''
  })
  const tmp = path.join(os.tmpdir(), `hermes-relaunch-test2-${Date.now()}.sh`)
  fs.writeFileSync(tmp, script)
  try {
    execFileSync('bash', ['-n', tmp], { stdio: 'pipe' })
  } finally {
    fs.rmSync(tmp, { force: true })
  }
  // exec line has no trailing args.
  assert.match(script, /exec '\/opt\/Hermes\/Hermes'\n/)
})
