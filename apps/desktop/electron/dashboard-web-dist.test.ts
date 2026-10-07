import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { resolveDashboardWebDist } from './dashboard-web-dist'

function touchIndex(dir) {
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(path.join(dir, 'index.html'), '<!doctype html>')
}

function withTempRoot(fn) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'rabbit-dashboard-web-dist-'))

  try {
    return fn(root)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
}

test('desktop-spawned dashboard resolves rabbit_cli/web_dist, not Desktop renderer dist', () => {
  withTempRoot(root => {
    const activeRabbitRoot = path.join(root, 'rabbit-agent')

    const desktopDist = path.join(
      activeRabbitRoot,
      'apps',
      'desktop',
      'release',
      'win-unpacked',
      'resources',
      'app.asar.unpacked',
      'dist'
    )

    const dashboardDist = path.join(activeRabbitRoot, 'rabbit_cli', 'web_dist')
    touchIndex(desktopDist)
    touchIndex(dashboardDist)
    assert.equal(
      resolveDashboardWebDist({
        activeRabbitRoot,
        appRoot: path.join(activeRabbitRoot, 'apps', 'desktop'),
        env: {}
      }),
      dashboardDist
    )
  })
})

test('explicit dashboard dist override wins when it exists', () => {
  withTempRoot(root => {
    const override = path.join(root, 'custom-dashboard-dist')
    touchIndex(override)
    assert.equal(
      resolveDashboardWebDist({
        activeRabbitRoot: path.join(root, 'rabbit-agent'),
        env: { RABBIT_DESKTOP_DASHBOARD_WEB_DIST: override }
      }),
      override
    )
  })
})

test('missing dashboard bundle falls back to canonical path for a clear child error', () => {
  withTempRoot(root => {
    const activeRabbitRoot = path.join(root, 'rabbit-agent')
    assert.equal(
      resolveDashboardWebDist({ activeRabbitRoot, env: {} }),
      path.join(activeRabbitRoot, 'rabbit_cli', 'web_dist')
    )
  })
})

test('appRoot candidate is used when it has a dashboard bundle and activeRabbitRoot does not', () => {
  withTempRoot(root => {
    const activeRabbitRoot = path.join(root, 'active-install')
    const sourceRoot = path.join(root, 'source-checkout')
    const appRoot = path.join(sourceRoot, 'apps', 'desktop')
    const appRootDashboard = path.resolve(appRoot, '..', '..', 'rabbit_cli', 'web_dist')
    fs.mkdirSync(activeRabbitRoot, { recursive: true })
    touchIndex(appRootDashboard)
    assert.equal(
      resolveDashboardWebDist({
        activeRabbitRoot,
        appRoot,
        env: {}
      }),
      appRootDashboard
    )
  })
})
