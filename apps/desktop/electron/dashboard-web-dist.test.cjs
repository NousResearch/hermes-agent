const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')

const { resolveDashboardWebDist } = require('./dashboard-web-dist.cjs')

function touchIndex(dir) {
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(path.join(dir, 'index.html'), '<!doctype html>')
}

test('desktop-spawned dashboard resolves hermes_cli/web_dist, not Desktop renderer dist', t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-dashboard-web-dist-'))
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))

  const activeHermesRoot = path.join(root, 'hermes-agent')
  const desktopDist = path.join(activeHermesRoot, 'apps', 'desktop', 'release', 'win-unpacked', 'resources', 'app.asar.unpacked', 'dist')
  const dashboardDist = path.join(activeHermesRoot, 'hermes_cli', 'web_dist')

  touchIndex(desktopDist)
  touchIndex(dashboardDist)

  assert.equal(
    resolveDashboardWebDist({
      activeHermesRoot,
      appRoot: path.join(activeHermesRoot, 'apps', 'desktop'),
      env: {}
    }),
    dashboardDist
  )
})

test('explicit dashboard dist override wins when it exists', t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-dashboard-web-dist-'))
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))

  const override = path.join(root, 'custom-dashboard-dist')
  touchIndex(override)

  assert.equal(
    resolveDashboardWebDist({
      activeHermesRoot: path.join(root, 'hermes-agent'),
      env: { HERMES_DESKTOP_DASHBOARD_WEB_DIST: override }
    }),
    override
  )
})

test('missing dashboard bundle falls back to canonical path for a clear child error', t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-dashboard-web-dist-'))
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))

  const activeHermesRoot = path.join(root, 'hermes-agent')

  assert.equal(
    resolveDashboardWebDist({ activeHermesRoot, env: {} }),
    path.join(activeHermesRoot, 'hermes_cli', 'web_dist')
  )
})
