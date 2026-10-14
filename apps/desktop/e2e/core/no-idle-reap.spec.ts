/**
 * Support f502bc6f: a "This device" profile's local `hermes serve` child was SIGTERM'd by the pool
 * idle reaper (60 min with nothing streamed), leaving "This device · Backend offline". A local
 * child hosts its profile's cron jobs and bot chats with nobody watching, so it is never retired
 * for being idle. The idle windows are pinned to their 60s floor so two reaper ticks fit the test.
 */

import * as path from 'node:path'

import { expect, test } from '@playwright/test'

import {
  backendProcesses,
  coreAppEnv,
  createCoreSandbox,
  launchCoreApp,
  waitForInteractive,
  writeProviderHome
} from './harness'
import { startScriptedProvider } from './provider'

const IDLE_MS = 60_000

test('a local profile backend is never idle-reaped', async () => {
  test.setTimeout(300_000)
  const provider = await startScriptedProvider()
  const box = createCoreSandbox('no-idle-reap')
  writeProviderHome(box.hermesHome, provider.url)
  writeProviderHome(path.join(box.hermesHome, 'profiles', 'reviewer'), provider.url)

  const { app, page } = await launchCoreApp(
    coreAppEnv(box, {
      HERMES_DESKTOP_POOL_IDLE_MS: String(IDLE_MS),
      HERMES_DESKTOP_POOL_PINNED_IDLE_MS: String(IDLE_MS),
      // One process per profile: the pooled child the reaper used to kill.
      HERMES_DESKTOP_ISOLATED_BACKEND: '1'
    })
  )

  try {
    await waitForInteractive(app, page)
    const before = new Set(backendProcesses(box).map(p => p.pid))

    await page.evaluate(() =>
      (window as any).hermesDesktop.getConnectionFor({
        connectionId: 'local',
        profile: 'reviewer',
        priority: 'foreground'
      })
    )

    const child = backendProcesses(box).find(p => !before.has(p.pid))
    expect(child, 'a pooled backend for reviewer').toBeDefined()

    // Two reaper ticks past both idle windows, with nothing touching or streaming.
    await page.waitForTimeout(IDLE_MS + 2 * 60_000 + 5_000)

    expect(
      backendProcesses(box).map(p => p.pid),
      'reviewer backend still running'
    ).toContain(child!.pid)
  } finally {
    await app.close().catch(() => undefined)
    await provider.close()
  }
})
