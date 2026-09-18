import { createRequire } from 'node:module'

import { expect, test } from 'vitest'

const { prepareWindowForInput } = createRequire(import.meta.url)(
  '../../../tests/install/e2e-assets/window-input.cjs',
)

test('does not finish when IPC reports 100% before the window factor settles', async () => {
  let observations = 0
  const previous = (globalThis as any).hermesDesktop
  ;(globalThis as any).hermesDesktop = { zoom: {
    setPercent: () => undefined,
    get: async () => ({ percent: 100 }),
  } }
  const appWindow = { evaluate: async (fn: any) => fn({ webContents: {
    getZoomFactor: () => ++observations === 1 ? 0.9 : 1,
  } }) }
  const page = {
    evaluate: async (fn: any) => fn(),
    waitForTimeout: async () => undefined,
  }
  try {
    await prepareWindowForInput({ browserWindow: async () => appWindow }, page)
    expect(observations).toBeGreaterThan(1)
  } finally {
    ;(globalThis as any).hermesDesktop = previous
  }
})

test('reapplies zoom when startup overwrites the first request', async () => {
  let requests = 0
  let factor = 0.9

  const previous = (globalThis as any).hermesDesktop

  ;(globalThis as any).hermesDesktop = { zoom: {
    setPercent: () => { requests++;

 if (requests > 1) {factor = 1} },
    get: async () => ({ percent: factor * 100 }),
  } }
  const window = { evaluate: async (fn: any) => fn({ webContents: { getZoomFactor: () => factor } }) }

  const page = {
    evaluate: async (fn: any) => fn(),
    waitForTimeout: async () => {
      if (requests === 1) {throw new Error('startup overwrote zoom and the driver never reapplied it')}
    },
  }

  try {
    await prepareWindowForInput({ browserWindow: async () => window }, page)
    expect(requests).toBeGreaterThan(1)
  } finally {
    ;(globalThis as any).hermesDesktop = previous
  }
})

test('awaits the zoom response instead of accepting a truthy Promise', async () => {
  let reads = 0
  let factor = 0.9

  const zoom = {
    setPercent: () => undefined,
    get: async () => {
      reads++

      if (reads > 1) {factor = 1}

      return { percent: factor * 100 }
    },
  }

  const previous = (globalThis as any).hermesDesktop

  ;(globalThis as any).hermesDesktop = { zoom }
  const window = { evaluate: async (fn: any) => fn({ webContents: { getZoomFactor: () => factor } }) }

  const page = {
    evaluate: async (fn: any) => fn(),
    // Playwright 1.58 accepts the predicate's Promise before it resolves.
    waitForFunction: async (fn: any) => { await fn() },
    waitForTimeout: async () => undefined,
  }

  try {
    await prepareWindowForInput({ browserWindow: async () => window }, page)
    expect(reads).toBeGreaterThan(1)
    expect(factor).toBe(1)
  } finally {
    ;(globalThis as any).hermesDesktop = previous
  }
})
