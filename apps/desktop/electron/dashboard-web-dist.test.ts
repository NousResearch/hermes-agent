import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

import { resolveDashboardWebDist } from './dashboard-web-dist'

function writeIndexHtml(dir: string) {
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(path.join(dir, 'index.html'), '<!doctype html><title>test</title>')
}

test('resolveDashboardWebDist prefers hermes_cli/web_dist over renderer dist', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-dashboard-dist-'))
  const cliDist = path.join(root, 'hermes_cli', 'web_dist')
  writeIndexHtml(cliDist)

  const resolved = resolveDashboardWebDist({
    hermesRoot: root,
    activeHermesRoot: path.join(root, 'unused-active'),
    isPackaged: true
  })

  assert.equal(resolved, cliDist)
})

test('resolveDashboardWebDist honours HERMES_DESKTOP_DASHBOARD_WEB_DIST override', () => {
  const overrideRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-dashboard-override-'))
  const overrideDist = path.join(overrideRoot, 'custom-web-dist')
  writeIndexHtml(overrideDist)

  const resolved = resolveDashboardWebDist({
    hermesRoot: null,
    activeHermesRoot: path.join(overrideRoot, 'missing'),
    dashboardOverride: overrideDist,
    isPackaged: true
  })

  assert.equal(resolved, path.resolve(overrideDist))
})

test('resolveDashboardWebDist falls back to active install root when backend has no root', () => {
  const active = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-dashboard-active-'))
  const cliDist = path.join(active, 'hermes_cli', 'web_dist')
  writeIndexHtml(cliDist)

  const resolved = resolveDashboardWebDist({
    hermesRoot: null,
    activeHermesRoot: active,
    isPackaged: true
  })

  assert.equal(resolved, cliDist)
})
