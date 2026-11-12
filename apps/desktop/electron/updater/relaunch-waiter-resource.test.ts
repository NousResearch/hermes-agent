// The sealed payload's repo snapshot has no scripts/, so the relaunch waiter
// must be shipped by the packaging config at the path main resolves.

import assert from 'node:assert/strict'
import fs from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'

import { expect, test } from 'vitest'

import { RELAUNCH_WAITER_SCRIPT, relaunchWaiterScript } from './relaunch-waiter'

const require: NodeJS.Require = createRequire(import.meta.url)
const desktop: string = path.resolve(import.meta.dirname, '..', '..')

test('the windows packaging config ships the relaunch waiter at the resource root', () => {
  const config = require('../../electron-builder.config.cjs')
  const extra: { from: string; to: string }[] = config.win.extraResources ?? []
  const shipped = extra.filter(item => item.to === RELAUNCH_WAITER_SCRIPT)

  assert.equal(shipped.length, 1, `win.extraResources must ship ${RELAUNCH_WAITER_SCRIPT}`)
  assert.ok(fs.existsSync(path.join(desktop, shipped[0].from)), `${shipped[0].from} must exist`)
})

test('the waiter resolves at the resources root', () => {
  const resources = path.join('app', 'resources')

  expect(relaunchWaiterScript(resources)).toBe(path.join(resources, RELAUNCH_WAITER_SCRIPT))
})
