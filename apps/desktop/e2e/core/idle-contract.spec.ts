/**
 * Idle contract: an app that sits open doing nothing moves nothing, and a
 * hidden window moves nothing even while work runs (tracker #127647,
 * apps/desktop/AGENTS.md "Idle costs nothing").
 *
 * Real Electron + real `hermes serve`; only the LLM is faked. The assertions
 * count motion, not CPU %, so they do not depend on runner speed. Motion is:
 *  - running infinite CSS animations (`document.getAnimations()`);
 *  - `requestAnimationFrame` calls per second, from a wrapper installed
 *    before any app module loads (so libraries that capture rAF at import,
 *    like motion's frame loop, are counted too);
 *  - `Element.animate()` calls, which catch timer-driven replays of finite
 *    Web Animations (StatusPulse's shared beat).
 * Not counted: DOM updates driven by setInterval/setTimeout alone (for
 * example DecodeText). Idle polls legitimately use timers, so a timer count
 * cannot be gated.
 *
 *  1. idle, window shown: nothing moves, confirmed over a sustained window.
 *  2. a turn is running: something moves (proves the probe sees real
 *     motion, so step 1 is not vacuously green).
 *  3. same turn, window hidden: nothing moves. Stream throttling is off while
 *     a turn streams, so only the app's own pause stops it (#51927, #53902).
 *  4. the turn finishes: nothing moves, confirmed over a sustained window.
 *     The stale-busy class (#53902, #91450) is an indicator that kept
 *     running after its work ended.
 *
 * Style recalcs per second over the sustained windows are attached as
 * annotations for comparison across runs; they are reported, not gated.
 */

import { type CDPSession, type ElectronApplication, expect, type Page, test } from '@playwright/test'
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

// Mirrors src/lib/renderer-loop-pause.ts (e2e does not import renderer modules).
const RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE = 'data-renderer-animations-paused'

// A live loop runs at the display rate (tens of calls per second); one-off
// frames (a layout measure after a store update) stay far below this.
const LOOP_FRAMES_PER_SECOND = 5

interface Probe {
  frames: number
  animateCalls: number
  animateTargets: string[]
  rawRaf: (cb: FrameRequestCallback) => number
}

interface Motion {
  animations: string[]
  framesPerSecond: number
  animateCalls: number
  animateTargets: string[]
  recalcsPerSecond: number
}

/** Runs in every new document before app code: counts rAF and Element.animate() calls. */
function installProbe() {
  const probe = {
    frames: 0,
    animateCalls: 0,
    animateTargets: [] as string[],
    rawRaf: window.requestAnimationFrame.bind(window)
  }

  ;(window as unknown as { __idleProbe: typeof probe }).__idleProbe = probe

  window.requestAnimationFrame = callback => {
    probe.frames += 1

    return probe.rawRaf(callback)
  }

  const animate = Element.prototype.animate

  Element.prototype.animate = function (this: Element, ...args: Parameters<Element['animate']>) {
    probe.animateCalls += 1
    // The last few callers, so a failure names what animated.
    const cls = typeof this.className === 'string' ? this.className.trim().split(/\s+/).slice(0, 3).join('.') : ''
    const keys = Array.isArray(args[0]) ? Object.keys(args[0][0] ?? {}).join(',') : Object.keys(args[0] ?? {}).join(',')
    probe.animateTargets = [
      ...probe.animateTargets.slice(-4),
      `${this.tagName.toLowerCase()}${cls ? `.${cls}` : ''} [${keys}]`
    ]

    return animate.apply(this, args)
  }
}

function readProbe(page: Page) {
  return page.evaluate(() => {
    const probe = (window as unknown as { __idleProbe?: Probe }).__idleProbe

    if (!probe) {
      // A reload without the init script would read as "still"; fail instead.
      throw new Error('idle probe missing: the renderer reloaded without it')
    }

    return { frames: probe.frames, animateCalls: probe.animateCalls, animateTargets: probe.animateTargets }
  })
}

async function recalcCount(cdp: CDPSession): Promise<number> {
  const { metrics } = await cdp.send('Performance.getMetrics')

  return metrics.find(m => m.name === 'RecalcStyleCount')?.value ?? 0
}

/** Everything that moved during `ms`, plus the running infinite CSS animations at the end. */
async function motion(page: Page, cdp: CDPSession, ms = 2_000): Promise<Motion> {
  const [p0, r0] = await Promise.all([readProbe(page), recalcCount(cdp)])
  await page.waitForTimeout(ms)
  const [p1, r1] = await Promise.all([readProbe(page), recalcCount(cdp)])

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

  const seconds = ms / 1000

  return {
    animations,
    framesPerSecond: Math.round(((p1.frames - p0.frames) / seconds) * 10) / 10,
    animateCalls: p1.animateCalls - p0.animateCalls,
    animateTargets: p1.animateCalls > p0.animateCalls ? p1.animateTargets : [],
    recalcsPerSecond: Math.round((r1 - r0) / seconds)
  }
}

/** 'still', or the motion that was seen (as the assertion's diff). */
function verdict(m: Motion, { judgeFrames = true } = {}): string {
  const moving =
    m.animations.length > 0 || m.animateCalls > 0 || (judgeFrames && m.framesPerSecond >= LOOP_FRAMES_PER_SECOND)

  return moving ? JSON.stringify(m) : 'still'
}

/** Act on the BrowserWindow that hosts `page` (the app also owns overlay windows). */
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
        const g = globalThis as unknown as { __idleHideSeen?: boolean }
        g.__idleHideSeen = false
        win.once('hide', () => (g.__idleHideSeen = true))
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

/** Whether the page still gets animation frames at all (a test-owned loop on the unwrapped rAF). */
function framesServiced(page: Page, ms = 1_000): Promise<boolean> {
  return page.evaluate(
    duration =>
      new Promise<boolean>(resolve => {
        const raf = (window as unknown as { __idleProbe: Probe }).__idleProbe.rawRaf
        let count = 0

        const tick = () => {
          count += 1

          if (count < 3) {
            raf(tick)
          }
        }

        raf(tick)
        setTimeout(() => resolve(count >= 3), duration)
      }),
    ms
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
    // Install the probe ahead of every app module, then load them again.
    await page.context().addInitScript(installProbe)
    await page.reload()
    await waitForInteractive(app, page)

    const cdp = await page.context().newCDPSession(page)
    await cdp.send('Performance.enable')

    const settleStill = async (message: string) => {
      await expect.poll(async () => verdict(await motion(page, cdp)), { timeout: 30_000, message }).toBe('still')
      // Confirm over a sustained window: a loop that restarts on a timer,
      // or a one-off quiet sample, does not pass.
      const sustained = await motion(page, cdp, 5_000)
      expect(verdict(sustained), `${message} (sustained 5s)`).toBe('still')

      return sustained
    }

    await test.step('idle, shown: nothing moves', async () => {
      const sustained = await settleStill('nothing moves at idle')
      test.info().annotations.push({ type: 'idle recalcs/s', description: String(sustained.recalcsPerSecond) })
    })

    const hold = gate()

    await test.step('a turn is running: something moves', async () => {
      provider.script(U(1), [{ text: [`${A(1)} `, 'working ', 'on it'], holdAfterFirstChunk: hold }])
      await send(page, `${U(1)} start`, 'Enter')
      await provider.streamStarted(U(1))
      await expect
        .poll(async () => verdict(await motion(page, cdp)), {
          timeout: 30_000,
          message: 'a running turn shows motion (otherwise the idle step proves nothing)'
        })
        .not.toBe('still')
    })

    await test.step('same turn, window hidden: nothing moves', async () => {
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
          .poll(
            () =>
              page.evaluate(attr => document.documentElement.hasAttribute(attr), RENDERER_ANIMATIONS_PAUSED_ATTRIBUTE),
            { timeout: 15_000, message: 'the app marks the hidden window paused' }
          )
          .toBe(true)

        // If the hidden page gets no frames at all, a loop that ignores the
        // pause stalls instead of counting, so frames cannot be judged; CSS
        // animations and Element.animate() replays still can.
        const judgeFrames = await framesServiced(page)

        if (!judgeFrames) {
          test
            .info()
            .annotations.push({ type: 'hidden frames not judged', description: 'no animation frames while hidden' })
        }

        await expect
          .poll(async () => verdict(await motion(page, cdp), { judgeFrames }), {
            timeout: 15_000,
            message: 'nothing moves while hidden'
          })
          .toBe('still')
        const sustained = await motion(page, cdp, 5_000)
        expect(verdict(sustained, { judgeFrames }), 'nothing moves while hidden (sustained 5s)').toBe('still')
        test.info().annotations.push({
          type: 'hidden recalcs/s (turn running)',
          description: String(sustained.recalcsPerSecond)
        })
      }

      await onPageWindow(app, page, 'show')
      await expect
        .poll(async () => verdict(await motion(page, cdp)), { timeout: 15_000, message: 'motion resumes when shown' })
        .not.toBe('still')
    })

    await test.step('the turn finishes: nothing moves', async () => {
      hold.open()
      await expect
        .poll(() => provider.completions.some(c => c.marker === U(1) && c.finished), {
          timeout: 60_000,
          message: 'provider finished the turn'
        })
        .toBe(true)
      await settleStill('nothing moves once the turn ends')
    })

    await cdp.detach()
  } finally {
    await app.close().catch(() => undefined)
    await provider.close()
  }
})
