import assert from 'node:assert/strict'

import { test } from 'vitest'

import { recoverRendererAfterCrash } from './renderer-crash-recovery'

test('disposes only the crashed renderer terminals before scheduling reload', () => {
  const calls: string[] = []

  const terminalSessions = new Map([
    ['matching-1', { webContentsId: 7 }],
    ['other-renderer', { webContentsId: 8 }],
    ['matching-2', { webContentsId: 7 }]
  ])

  const reloadTimes = recoverRendererAfterCrash({
    disposeTerminalSession: id => calls.push(`dispose:${id}`),
    maxReloads: 2,
    now: 1_000,
    onDisposeError: (id, error) => calls.push(`error:${id}:${error}`),
    onReload: () => calls.push('reload'),
    onSuppress: count => calls.push(`suppress:${count}`),
    reloadTimes: [],
    reloadWindowMs: 500,
    rendererWebContentsId: 7,
    terminalSessions
  })

  assert.deepEqual(calls, ['dispose:matching-1', 'dispose:matching-2', 'reload'])
  assert.deepEqual(reloadTimes, [1_000])
})

test('disposes the crashed renderer terminals before reload-loop suppression', () => {
  const calls: string[] = []

  const terminalSessions = new Map([
    ['matching', { webContentsId: 7 }],
    ['other-renderer', { webContentsId: 8 }]
  ])

  const reloadTimes = recoverRendererAfterCrash({
    disposeTerminalSession: id => calls.push(`dispose:${id}`),
    maxReloads: 2,
    now: 1_000,
    onDisposeError: (id, error) => calls.push(`error:${id}:${error}`),
    onReload: () => calls.push('reload'),
    onSuppress: count => calls.push(`suppress:${count}`),
    reloadTimes: [700, 900],
    reloadWindowMs: 500,
    rendererWebContentsId: 7,
    terminalSessions
  })

  assert.deepEqual(calls, ['dispose:matching', 'suppress:2'])
  assert.deepEqual(reloadTimes, [700, 900])
})

test('expired reload attempts do not suppress recovery', () => {
  const calls: string[] = []

  const reloadTimes = recoverRendererAfterCrash({
    disposeTerminalSession: id => calls.push(`dispose:${id}`),
    maxReloads: 2,
    now: 1_000,
    onDisposeError: (id, error) => calls.push(`error:${id}:${error}`),
    onReload: () => calls.push('reload'),
    onSuppress: count => calls.push(`suppress:${count}`),
    reloadTimes: [100, 700],
    reloadWindowMs: 500,
    rendererWebContentsId: 7,
    terminalSessions: new Map([['matching', { webContentsId: 7 }]])
  })

  assert.deepEqual(calls, ['dispose:matching', 'reload'])
  assert.deepEqual(reloadTimes, [700, 1_000])
})

test('continues cleanup and reload when one terminal disposer throws', () => {
  const calls: string[] = []

  const terminalSessions = new Map([
    ['broken', { webContentsId: 7 }],
    ['healthy', { webContentsId: 7 }]
  ])

  recoverRendererAfterCrash({
    disposeTerminalSession: id => {
      calls.push(`dispose:${id}`)

      if (id === 'broken') {
        throw new Error('kill failed')
      }
    },
    maxReloads: 2,
    now: 1_000,
    onDisposeError: (id, error) => calls.push(`error:${id}:${error instanceof Error ? error.message : error}`),
    onReload: () => calls.push('reload'),
    onSuppress: count => calls.push(`suppress:${count}`),
    reloadTimes: [],
    reloadWindowMs: 500,
    rendererWebContentsId: 7,
    terminalSessions
  })

  assert.deepEqual(calls, [
    'dispose:broken',
    'error:broken:kill failed',
    'dispose:healthy',
    'reload'
  ])
})

test('continues cleanup and reload when dispose error reporting throws', () => {
  const calls: string[] = []

  const terminalSessions = new Map([
    ['broken', { webContentsId: 7 }],
    ['healthy', { webContentsId: 7 }]
  ])

  recoverRendererAfterCrash({
    disposeTerminalSession: id => {
      calls.push(`dispose:${id}`)

      if (id === 'broken') {
        throw new Error('kill failed')
      }
    },
    maxReloads: 2,
    now: 1_000,
    onDisposeError: id => {
      calls.push(`error:${id}`)
      throw new Error('logger failed')
    },
    onReload: () => calls.push('reload'),
    onSuppress: count => calls.push(`suppress:${count}`),
    reloadTimes: [],
    reloadWindowMs: 500,
    rendererWebContentsId: 7,
    terminalSessions
  })

  assert.deepEqual(calls, ['dispose:broken', 'error:broken', 'dispose:healthy', 'reload'])
})
