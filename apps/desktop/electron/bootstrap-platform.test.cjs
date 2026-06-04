const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')

const {
  bundledRuntimeImportCheck,
  detectRemoteDisplay,
  isWindowsBinaryPathInWsl,
  isWslEnvironment
} = require('./bootstrap-platform.cjs')

test('isWslEnvironment detects WSL2 env vars on linux', () => {
  assert.equal(isWslEnvironment({ WSL_DISTRO_NAME: 'Ubuntu' }, 'linux'), true)
  assert.equal(isWslEnvironment({ WSL_INTEROP: '/run/WSL/123_interop' }, 'linux'), true)
  assert.equal(isWslEnvironment({}, 'linux', '6.6.87.2-microsoft-standard-WSL2'), true)
  assert.equal(isWslEnvironment({}, 'linux', '6.6.87-generic'), false)
  assert.equal(isWslEnvironment({ WSL_DISTRO_NAME: 'Ubuntu' }, 'darwin'), false)
})

test('isWindowsBinaryPathInWsl blocks Windows binary types on WSL', () => {
  assert.equal(isWindowsBinaryPathInWsl('/mnt/c/Tools/hermes.exe', { isWsl: true }), true)
  assert.equal(isWindowsBinaryPathInWsl('/mnt/c/Tools/hermes.cmd', { isWsl: true }), true)
  assert.equal(isWindowsBinaryPathInWsl('/mnt/c/Tools/hermes.bat', { isWsl: true }), true)
  assert.equal(isWindowsBinaryPathInWsl('/mnt/c/Tools/install.ps1', { isWsl: true }), true)
  assert.equal(isWindowsBinaryPathInWsl('/usr/local/bin/hermes', { isWsl: true }), false)
  assert.equal(isWindowsBinaryPathInWsl('/mnt/c/Tools/hermes.exe', { isWsl: false }), false)
})

test('bundledRuntimeImportCheck selects platform-specific import checks', () => {
  assert.equal(bundledRuntimeImportCheck('win32'), 'import fastapi, uvicorn, winpty')
  assert.equal(bundledRuntimeImportCheck('darwin'), 'import fastapi, uvicorn, ptyprocess')
  assert.equal(bundledRuntimeImportCheck('linux'), 'import fastapi, uvicorn, ptyprocess')
})

test('detectRemoteDisplay keeps GPU on for local sessions', () => {
  // Plain local X11, Wayland, native Windows, native macOS — no remote signal.
  assert.equal(detectRemoteDisplay({ env: { DISPLAY: ':0' }, platform: 'linux' }), null)
  assert.equal(detectRemoteDisplay({ env: { WAYLAND_DISPLAY: 'wayland-0' }, platform: 'linux' }), null)
  assert.equal(detectRemoteDisplay({ env: { SESSIONNAME: 'Console' }, platform: 'win32' }), null)
  assert.equal(detectRemoteDisplay({ env: {}, platform: 'darwin' }), null)
})

test('detectRemoteDisplay does not treat WSLg as remote', () => {
  // WSLg renders locally via vGPU and doesn't show the flicker, so a WSL
  // session with a local DISPLAY keeps hardware acceleration on.
  assert.equal(detectRemoteDisplay({ env: { WSL_DISTRO_NAME: 'Ubuntu', DISPLAY: ':0' }, platform: 'linux' }), null)
  assert.equal(detectRemoteDisplay({ env: { WSL_INTEROP: '/run/WSL/1_interop', DISPLAY: ':0' }, platform: 'linux' }), null)
})

test('detectRemoteDisplay flags SSH sessions on any platform', () => {
  assert.equal(detectRemoteDisplay({ env: { SSH_CONNECTION: '1.2.3.4 5 6.7.8.9 22' }, platform: 'linux' }), 'ssh-session')
  assert.equal(detectRemoteDisplay({ env: { SSH_CLIENT: '1.2.3.4 5 22' }, platform: 'darwin' }), 'ssh-session')
  assert.equal(detectRemoteDisplay({ env: { SSH_TTY: '/dev/pts/0' }, platform: 'win32' }), 'ssh-session')
})

test('detectRemoteDisplay flags forwarded X11 displays but not local ones', () => {
  assert.match(String(detectRemoteDisplay({ env: { DISPLAY: 'localhost:10.0' }, platform: 'linux' })), /x11-forwarding/)
  assert.match(String(detectRemoteDisplay({ env: { DISPLAY: '192.168.1.5:0' }, platform: 'linux' })), /x11-forwarding/)
  assert.equal(detectRemoteDisplay({ env: { DISPLAY: ':1' }, platform: 'linux' }), null)
})

test('detectRemoteDisplay flags RDP sessions', () => {
  assert.match(String(detectRemoteDisplay({ env: { SESSIONNAME: 'RDP-Tcp#7' }, platform: 'win32' })), /^rdp/)
})

test('detectRemoteDisplay honors the HERMES_DESKTOP_DISABLE_GPU override both ways', () => {
  // Force-on even on a local display.
  assert.match(
    String(detectRemoteDisplay({ env: { HERMES_DESKTOP_DISABLE_GPU: '1', DISPLAY: ':0' }, platform: 'linux' })),
    /override/
  )
  // Force-off even over SSH (escape hatch when a remote display has working accel).
  assert.equal(
    detectRemoteDisplay({
      env: { HERMES_DESKTOP_DISABLE_GPU: 'false', SSH_CONNECTION: '1.2.3.4 5 6.7.8.9 22' },
      platform: 'linux'
    }),
    null
  )
})

test('packaged electron entrypoints do not require unpackaged npm modules', () => {
  const electronDir = __dirname
  const entrypoints = ['main.cjs', 'preload.cjs', 'bootstrap-platform.cjs']
  // - electron: provided by the electron runtime, always resolvable in packaged builds.
  // - node-pty: hoisted by workspace dedup AND shipped via extraResources to
  //   resources/native-deps/node-pty (see scripts/stage-native-deps.cjs). main.cjs
  //   has a try/catch fallback at line ~38 that resolves the staged copy when the
  //   bare require fails in the packaged asar, so the bare require itself is by
  //   design rather than an oversight.
  const allowedBareRequires = new Set(['electron', 'node-pty'])
  const requirePattern = /require\(['"]([^'"]+)['"]\)/g

  for (const entrypoint of entrypoints) {
    const source = fs.readFileSync(path.join(electronDir, entrypoint), 'utf8')
    const bareRequires = Array.from(source.matchAll(requirePattern))
      .map(match => match[1])
      .filter(specifier => !specifier.startsWith('node:'))
      .filter(specifier => !specifier.startsWith('.'))
      .filter(specifier => !allowedBareRequires.has(specifier))

    assert.deepEqual(bareRequires, [], `${entrypoint} has unpackaged runtime requires`)
  }
})
