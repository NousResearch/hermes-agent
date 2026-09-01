/**
 * Reaping an SSH-kind pool entry that owns a spawned remote `serve --isolated`
 * must kill that remote process. The idle reaper used to drop the local
 * descriptor via stopPoolBackend and leave the detached remote serve at pid 1;
 * the next dial of that profile then spawned a fresh serve alongside the
 * orphan.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

import { createPoolStopper, type PoolStopEntry } from './pool-stop'

const NOW = 1_000_000
const IDLE_MS = 5 * 60_000

test('reaping idle ssh-kind entries fires the remote-kill hook; remote-client descriptors are dropped without it', async () => {
  const pool = new Map<string, PoolStopEntry>()
  const killed: string[] = []

  pool.set('conn:ssh::atlas', {
    lastActiveAt: NOW - 500_000,
    ownsRemoteServe: true,
    process: null
  })
  // Remote-client descriptor (#75398): reaped like before, but never remote-killed.
  pool.set('conn:cloud::default', {
    lastActiveAt: NOW - 999_000,
    process: null
  })
  pool.set('conn:ssh::clio', {
    lastActiveAt: NOW - 1_000,
    ownsRemoteServe: true,
    process: null
  })

  const stopper = createPoolStopper({
    pool,
    stopChild: () => undefined,
    stopOwnedRemote: async key => {
      killed.push(key)
    },
    waitForExit: async () => undefined
  })

  // The idle reaper's selection: everything idle past the threshold.
  const idle = [...pool.entries()]
    .filter(([, entry]) => NOW - (entry.lastActiveAt || 0) > IDLE_MS)
    .map(([key]) => key)

  assert.deepEqual(idle, ['conn:ssh::atlas', 'conn:cloud::default'])

  for (const key of idle) {
    await stopper.stopAndReclaim(key)
  }

  assert.deepEqual(killed, ['conn:ssh::atlas'])
  assert.equal(pool.has('conn:ssh::atlas'), false)
  assert.equal(pool.has('conn:cloud::default'), false)
  assert.equal(pool.has('conn:ssh::clio'), true)
})

test('stopAndReclaim kills the owned remote serve', async () => {
  const pool = new Map<string, PoolStopEntry>()
  const killed: string[] = []

  pool.set('conn:ssh::atlas', { ownsRemoteServe: true, process: null })

  const stopper = createPoolStopper({
    pool,
    stopChild: () => undefined,
    stopOwnedRemote: async key => {
      killed.push(key)
    },
    waitForExit: async () => undefined
  })

  await stopper.stopAndReclaim('conn:ssh::atlas')

  assert.deepEqual(killed, ['conn:ssh::atlas'])
  assert.equal(pool.has('conn:ssh::atlas'), false)
})

test('plain stop of an ssh-kind entry drops the descriptor without killing the remote serve', async () => {
  const pool = new Map<string, PoolStopEntry>()
  const killed: string[] = []

  pool.set('conn:ssh::atlas', { ownsRemoteServe: true, process: null })

  const stopper = createPoolStopper({
    pool,
    stopChild: () => undefined,
    stopOwnedRemote: async key => {
      killed.push(key)
    },
    waitForExit: async () => undefined
  })

  await stopper.stop('conn:ssh::atlas')

  assert.equal(pool.has('conn:ssh::atlas'), false)
  assert.deepEqual(killed, [])
})

const here = path.dirname(fileURLToPath(import.meta.url))
const mainSource = fs.readFileSync(path.join(here, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')

test('main.ts idle reaper reclaims owned remote serves; LRU eviction is unchanged', () => {
  const evictStart = mainSource.indexOf('function evictLruPoolBackends(')
  const evictBody = mainSource.slice(evictStart, evictStart + 500)
  const reaperStart = mainSource.indexOf('function startPoolIdleReaper(')
  const reaperBody = mainSource.slice(reaperStart, reaperStart + 900)

  assert.match(reaperBody, /stopPoolBackendReclaiming\(/)
  assert.doesNotMatch(evictBody, /stopPoolBackendReclaiming\(/)
  assert.match(mainSource, /stopOwnedRemote:\s*async key =>/)
  assert.match(mainSource, /ownsRemoteServe: source\.kind === 'ssh'/)
})
