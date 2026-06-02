const assert = require('node:assert/strict')
const path = require('node:path')
const test = require('node:test')

const { isPathInside, resolvePosixGuiDeploy } = require('./update-deploy.cjs')

const ROOT = path.resolve('/home/dev/hermes-agent')
const RELEASE_DIR = path.join(ROOT, 'apps', 'desktop', 'release')

test('isPathInside: a nested path is inside', () => {
  assert.equal(isPathInside(path.join(RELEASE_DIR, 'linux-unpacked', 'hermes'), RELEASE_DIR), true)
})

test('isPathInside: an identical path counts as inside', () => {
  assert.equal(isPathInside(RELEASE_DIR, RELEASE_DIR), true)
})

test('isPathInside: a sibling/system path is outside', () => {
  assert.equal(isPathInside(path.resolve('/opt/Hermes/hermes'), RELEASE_DIR), false)
})

test('isPathInside: missing arguments are never inside', () => {
  assert.equal(isPathInside('', RELEASE_DIR), false)
  assert.equal(isPathInside(path.join(RELEASE_DIR, 'x'), ''), false)
})

test('macOS packaged install swaps the bundle in place', () => {
  const plan = resolvePosixGuiDeploy({
    platform: 'darwin',
    execPath: '/Applications/Hermes.app/Contents/MacOS/Hermes',
    releaseDir: RELEASE_DIR,
    macBundleSrc: path.join(RELEASE_DIR, 'mac-arm64', 'Hermes.app'),
    macBundleDst: '/Applications/Hermes.app'
  })
  assert.deepEqual(plan, {
    kind: 'swap-mac',
    src: path.join(RELEASE_DIR, 'mac-arm64', 'Hermes.app'),
    dst: '/Applications/Hermes.app'
  })
})

test('macOS dev run (no bundle to swap) restarts in place', () => {
  const plan = resolvePosixGuiDeploy({
    platform: 'darwin',
    execPath: path.join(ROOT, 'node_modules', 'electron', 'dist', 'Electron'),
    releaseDir: RELEASE_DIR,
    macBundleSrc: null,
    macBundleDst: null
  })
  assert.equal(plan.kind, 'restart')
})

test('Linux CLI install (running from release/) restarts in place', () => {
  const plan = resolvePosixGuiDeploy({
    platform: 'linux',
    execPath: path.join(RELEASE_DIR, 'linux-unpacked', 'hermes'),
    releaseDir: RELEASE_DIR,
    macBundleSrc: null,
    macBundleDst: null
  })
  assert.equal(plan.kind, 'restart')
})

test('Linux packaged AppImage needs a manual GUI reinstall', () => {
  const plan = resolvePosixGuiDeploy({
    platform: 'linux',
    execPath: '/tmp/.mount_HermesAbc123/hermes',
    releaseDir: RELEASE_DIR,
    macBundleSrc: null,
    macBundleDst: null
  })
  assert.equal(plan.kind, 'manual-gui')
})

test('Linux packaged deb/rpm install needs a manual GUI reinstall', () => {
  const plan = resolvePosixGuiDeploy({
    platform: 'linux',
    execPath: '/opt/Hermes/hermes',
    releaseDir: RELEASE_DIR,
    macBundleSrc: null,
    macBundleDst: null
  })
  assert.equal(plan.kind, 'manual-gui')
})
