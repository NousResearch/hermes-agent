/**
 * Idle contract: an app that sits open doing nothing moves nothing, and a
 * hidden window moves nothing even while work runs (tracker #127647,
 * apps/desktop/AGENTS.md "Idle costs nothing").
 *
 * Real Electron + real `hermes serve`; only the LLM is faked. The assertions
 * count motion, not CPU %, so they do not depend on runner speed. Motion is
 * the running infinite CSS animations (`document.getAnimations()`) plus
 * `requestAnimationFrame` callbacks per second, the JS loops (dither canvas,
 * pet, decode text), counted by wrapping the global `requestAnimationFrame`:
 *
 *  1. idle, window shown: nothing moves.
 *  2. a turn is running: something moves (proves the probe sees real
 *     motion, so step 1 is not vacuously green).
 *  3. same turn, window hidden: nothing moves. Stream throttling is off while
 *     a turn streams, so only the app's own pause stops it (#51927, #53902).
 *  4. the turn finishes: nothing moves. The stale-busy class (#53902, #91450)
 *     is an indicator that kept running after its work ended.
 *
 * Style recalcs per second at idle and while hidden are attached as
 * annotations for comparison across runs; they are reported, not gated.
 */

import { type ElectronApplication, expect, type Page, test } from '@playwright/test'
import type { BrowserWindow as ElectronWindow } from 'electron'

import { coreAppEnv, createCoreSandbox, launchCoreApp, send, waitForInteractive, writeProviderHome } from './harness'
import { gate, startScriptedProvider } from './provider'

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-${nonce}`
const A = (n: number) => `A${n}-${nonce}`

interface Motion {
  animations: string[]
  framesPerSecond: number
}

const STILL: Motion = { animations: [], framesPerSecond: 0 }
const moves = (m: Motion) => m.animations.length > 0 || m.framesPerSecond > 0

/** Count every `requestAnimationFrame` call from now on; a loop reschedules through the global each frame. */
function installFrameCounter(page: Page): Promise<void> {
  return page.evaluate(() => {
    const w = window as unknown as { __idleFrames?: number }

    if (w.__idleFrames !== undefined) {
      return
    }

    w.__idleFrames = 0
    const original = window.requestAnimationFrame.bind(window)

    window.requestAnimationFrame = callback => {
      w.__idleFrames = (w.__idleFrames ?? 0) + 1

      return original(callback)
    }
  })
}

/** Running infinite CSS animations (name + target) and rAF callbacks per second over `ms`. */
async function motion(page: Page, ms = 2_000): Promise<Motion> {
  const frames = () => page.evaluate(() => (window as unknown as { __idleFrames?: number }).__idleFrames ?? 0)
  const before = await frames()
  await page.waitForTimeout(ms)
  const framesPerSecond = Math.round(((await frames()) - before) / (ms / 1000))

  const animations = await page.evaluate(() =>
    document
      .getAnimations()
      .filter(a => a.playState === 'running' && a.effect?.getComputedTiming().iterations === Infinity)
      .map(a => {
        const effect = a.effect as KeyframeEffect | null
        const el = effect?.target as Element | null
        const cls = el && typeof el.className === 'string' ? el.className.trim().split(/\s+/).slice(0, 2).join('.') : ''

        return `${(a as CSSAnimation).animationName ?? 'script'} on ${el?.tagName.toLowerCase() ?? '?'}${cls ? `.${cls}` : ''}${effect?.pseudoElement ?? ''}`
      })
  )

  return { animations, framesPerSecond }
}

/** Style recalcs per second over `ms`, from the renderer's own counters. */
async function recalcsPerSecond(page: Page, ms = 5_000): Promise<number> {
  const cdp = await page.context().newCDPSession(page)

  try {
    await cdp.send('Performance.enable')

    const read = async () =>
      (await cdp.send('Performance.getMetrics')).metrics.find(m => m.name === 'RecalcStyleCount')?.value ?? 0

    const before = await read()
    await page.waitForTimeout(ms)

    return Math.round(((await read()) - before) / (ms / 1000))
  } finally {
    await cdp.detach()
  }
}

/** Run `action` on the BrowserWindow that hosts `page` (the app also owns overlay windows). */
function onPageWindow(app: ElectronApplication, page: Page, action: 'hide' | 'show' | 'size') {
  return app.evaluate(
    ({ BrowserWindow }, { url, action }) => {
      const win = BrowserWindow.getAllWindows().find(
        (w: ElectronWindow) => !w.isDestroyed() && w.webContents.getURL() === url
      )

      if (!win) {
        throw new Error(`no window hosts ${url}`)
      }

      if (action === 'hide') {
        win.hide()
      } else if (action === 'show') {
        win.show()
      } else {
        // Desktop-sized, so the sidebar and its rows lay out as users see them.
        win.setSize(1400, 900)
      }
    },
    { url: page.url(), action }
  )
}

test('idle windows move nothing, hidden windows pause, and motion stops with the work', async () => {
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('idle')
  writeProviderHome(sandbox.hermesHome, provider.url)
  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))

  try {
    await waitForInteractive(app, page)
    await onPageWindow(app, page, 'size')
    await installFrameCounter(page)

    await test.step('idle, shown: nothing moves', async () => {
      // Boot-time motion (the connecting glyph) settles first.
      await expect.poll(() => motion(page), { timeout: 30_000, message: 'nothing moves at idle' }).toEqual(STILL)
      test.info().annotations.push({ type: 'idle recalcs/s', description: String(await recalcsPerSecond(page)) })
      // Still nothing after a sustained window: a clock or loop that starts
      // motion later shows up here.
      expect(await motion(page)).toEqual(STILL)
    })

    const hold = gate()

    await test.step('a turn is running: something moves', async () => {
      provider.script(U(1), [{ text: [`${A(1)} `, 'working ', 'on it'], holdAfterFirstChunk: hold }])
      await send(page, `${U(1)} start`, 'Enter')
      await provider.streamStarted(U(1))
      await expect
        .poll(async () => moves(await motion(page)), {
          timeout: 30_000,
          message: 'a running turn shows motion (otherwise the idle step proves nothing)'
        })
        .toBe(true)
    })

    await test.step('same turn, window hidden: nothing moves', async () => {
      await app.evaluate(({ BrowserWindow }) => {
        const g = globalThis as unknown as { __idleHideSeen?: boolean }
        g.__idleHideSeen = false

        for (const w of BrowserWindow.getAllWindows()) {
          w.once('hide', () => (g.__idleHideSeen = true))
        }
      })
      await onPageWindow(app, page, 'hide')

      const hideSeen = await expect
        .poll(() => app.evaluate(() => (globalThis as unknown as { __idleHideSeen?: boolean }).__idleHideSeen), {
          timeout: 5_000
        })
        .toBe(true)
        .then(
          () => true,
          () => false
        )

      if (!hideSeen) {
        // Observed on macOS for a test-launched window: hide() leaves it
        // invisible but Electron emits no `hide`, so the app's own path never
        // runs and there is nothing to judge. The CI lane (Linux/xvfb) gets
        // the event; there this branch is not taken.
        test.info().annotations.push({
          type: 'hidden step not judged',
          description: `${process.platform}: Electron emitted no hide event`
        })
        expect(process.platform, 'only macOS is known to drop the programmatic hide event').toBe('darwin')
      } else {
        await expect
          .poll(() => page.evaluate(() => document.documentElement.hasAttribute('data-renderer-animations-paused')), {
            timeout: 15_000,
            message: 'the app marks the hidden window paused'
          })
          .toBe(true)
        await expect.poll(() => motion(page), { timeout: 15_000, message: 'nothing moves while hidden' }).toEqual(STILL)
        test.info().annotations.push({
          type: 'hidden recalcs/s (turn running)',
          description: String(await recalcsPerSecond(page))
        })
      }

      await onPageWindow(app, page, 'show')
      await expect
        .poll(async () => moves(await motion(page)), { timeout: 15_000, message: 'motion resumes when shown' })
        .toBe(true)
    })

    await test.step('the turn finishes: nothing moves', async () => {
      hold.open()
      await expect
        .poll(() => provider.completions.some(c => c.marker === U(1) && c.finished), {
          timeout: 60_000,
          message: 'provider finished the turn'
        })
        .toBe(true)
      await expect
        .poll(() => motion(page), { timeout: 30_000, message: 'nothing moves once the turn ends' })
        .toEqual(STILL)
    })
  } finally {
    await app.close().catch(() => undefined)
    await provider.close()
  }
})
