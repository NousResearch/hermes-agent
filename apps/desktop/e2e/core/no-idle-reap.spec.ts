/**
 * Support f502bc6f: a "This device" profile's local backend was SIGTERM'd by the Desktop pool idle
 * reaper (60 min with nothing streamed), leaving "This device · Backend offline". A local profile's
 * backend hosts its cron jobs and bot chats with nobody watching, so it is never torn down for being
 * idle while the app runs.
 *
 * Desktop no longer owns per-profile backend children: every local profile attaches to the one host
 * gateway (`hermes gateway ensure`). The invariant here: an attached, idle secondary profile keeps
 * the same gateway process(es) and the same endpoint past the retired reaper's 60s idle floor plus
 * two of its 60s ticks, and re-dialing it attaches instead of spawning.
 */

import * as path from 'node:path'

import { expect, type Page, test } from '@playwright/test'

import {
  backendProcesses,
  coreAppEnv,
  createCoreSandbox,
  launchCoreApp,
  waitForInteractive,
  writeProviderHome
} from './harness'
import { startScriptedProvider } from './provider'

// The retired pool's idle floor (60s) plus two reaper ticks (60s each) and slack.
const OLD_IDLE_WINDOW_MS = 60_000 + 2 * 60_000 + 5_000

function dialReviewer(page: Page) {
  return page.evaluate(async () => {
    const conn = await (window as any).hermesDesktop.getConnectionFor({ connectionId: 'local', profile: 'reviewer' })

    return { baseUrl: String(conn?.baseUrl ?? ''), mode: String(conn?.mode ?? ''), profile: String(conn?.profile ?? '') }
  })
}

const gatewayPids = (box: ReturnType<typeof createCoreSandbox>) =>
  backendProcesses(box)
    .map(p => p.pid)
    .sort((a, b) => a - b)

test('an idle local profile stays attached to its gateway', async () => {
  test.setTimeout(360_000)
  const provider = await startScriptedProvider()
  const box = createCoreSandbox('no-idle-reap')
  writeProviderHome(box.hermesHome, provider.url)
  writeProviderHome(path.join(box.hermesHome, 'profiles', 'reviewer'), provider.url)

  const { app, page } = await launchCoreApp(coreAppEnv(box))

  try {
    await waitForInteractive(app, page)

    const attached = await dialReviewer(page)
    expect(attached.mode, 'reviewer dials the local gateway').toBe('local')
    expect(attached.baseUrl, 'reviewer has a gateway endpoint').toMatch(/^http:\/\/127\.0\.0\.1:\d+/)

    const before = gatewayPids(box)
    expect(before.length, 'a local gateway is running').toBeGreaterThan(0)

    // Nothing touches or streams on reviewer for longer than any idle window the old pool used.
    await page.waitForTimeout(OLD_IDLE_WINDOW_MS)

    expect(gatewayPids(box), 'the same gateway process(es) still run').toEqual(before)

    const redial = await dialReviewer(page)
    expect(redial, 'reviewer is still attached to the same endpoint').toEqual(attached)
    expect(gatewayPids(box), 're-dial attaches, never spawns').toEqual(before)
  } finally {
    await app.close().catch(() => undefined)
    await provider.close()
  }
})
