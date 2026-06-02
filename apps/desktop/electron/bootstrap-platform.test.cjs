const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')

const { bundledRuntimeImportCheck, isWindowsBinaryPathInWsl, isWslEnvironment } = require('./bootstrap-platform.cjs')

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
