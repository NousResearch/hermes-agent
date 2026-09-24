/**
 * C5 core: boot handshake and backend process lifecycle.
 *
 * Real Electron + real `hermes serve`; only the LLM is faked. Process facts
 * come from a /proc census of every process carrying this sandbox's
 * HERMES_HOME (so an orphan reparented to init is still counted), sampled
 * continuously in the background so a transient extra spawn is seen too.
 *
 *  1. boot: composer interactive, exactly one backend, first turn completes
 *     and passes the transcript oracle.
 *  2. kill -9 the backend: the supervisor respawns EXACTLY one replacement
 *     (no crash-loop, no double spawn), and the app serves a new turn.
 *  3. quit while a turn is streaming and a tool subprocess is running: zero
 *     sandbox processes remain — no backend, no tool child, no Electron helper.
 *  4. relaunch the same HERMES_HOME repeatedly: each boot has exactly one
 *     backend, each quit leaves zero processes, and the transcript persisted
 *     by the first launch cold-hydrates exactly once every time.
 */

import * as fs from 'node:fs'

import { expect, test } from '@playwright/test'

import {
  backendProcesses,
  coreAppEnv,
  createCoreSandbox,
  currentSessionId,
  launchCoreApp,
  type ProcInfo,
  recordWebSockets,
  sandboxProcesses,
  send,
  waitForInteractive,
  writeProviderHome
} from './harness'
import { assertTranscriptOracle, installDuplicateSampler, type OracleTarget } from './oracle'
import { gate, startScriptedProvider } from './provider'

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-${nonce}`
const A = (n: number) => `A${n}-${nonce}`
const TOOL_TAG = `core-orphan-${nonce}`

/** Every live process whose command line carries `tag` (tool children may scrub HERMES_HOME). */
function taggedProcesses(tag: string): ProcInfo[] {
  const out: ProcInfo[] = []

  for (const entry of fs.readdirSync('/proc')) {
    const pid = Number(entry)

    if (!Number.isInteger(pid) || pid === process.pid) {
      continue
    }

    try {
      const cmdline = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8').split('\0').join(' ').trim()

      if (cmdline.includes(tag)) {
        out.push({ pid, ppid: 0, cmdline })
      }
    } catch {
      /* exited mid-scan */
    }
  }

  return out
}

test('boot handshake, supervised respawn, and zero orphans on quit', async () => {
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('boot')
  writeProviderHome(sandbox.hermesHome, provider.url)
  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))
  const ws = recordWebSockets(page)
  let closed = false

  // Background census: every backend pid ever observed, with its parent,
  // command line and lifetime so an unexpected extra pid explains itself.
  const seen = new Map<number, { ppid: number; cmdline: string; first: number; last: number; samples: number }>()
  const t0 = Date.now()

  const census = setInterval(() => {
    const now = Date.now() - t0

    for (const proc of backendProcesses(sandbox)) {
      const entry = seen.get(proc.pid)

      if (entry) {
        entry.last = now
        entry.samples++
      } else {
        seen.set(proc.pid, { ppid: proc.ppid, cmdline: proc.cmdline.slice(0, 200), first: now, last: now, samples: 1 })
      }
    }
  }, 100)

  const describeSeen = () =>
    JSON.stringify([...seen].map(([pid, e]) => ({ pid, ...e, electronPid: app.process().pid })))

  const finished = (marker: string, step = 0) =>
    expect
      .poll(() => provider.completions.some(c => c.marker === marker && c.step === step && c.finished), {
        timeout: 120_000,
        message: `provider finished ${marker} step ${step}`
      })
      .toBe(true)

  try {
    const session: OracleTarget = { sessionId: '', expectUserMarkers: [] }

    await test.step('boot: interactive composer, one backend, first turn', async () => {
      await waitForInteractive(app, page)
      await installDuplicateSampler(page)
      await expect.poll(() => backendProcesses(sandbox).length, { message: 'exactly one backend after boot' }).toBe(1)
      provider.script(U(1), [{ text: [`${A(1)} `, 'booted ', 'and ', 'answered'] }])
      await send(page, `${U(1)} first`, 'Enter', ws)
      await finished(U(1))
      await expect.poll(() => currentSessionId(page)).not.toBe('')
      session.sessionId = await currentSessionId(page)
      session.expectUserMarkers.push(U(1))
      await assertTranscriptOracle(page, ws, provider, session, 'boot first turn')
      expect(seen.size, `backend pids seen during boot: ${describeSeen()}`).toBe(1)
    })

    await test.step('kill -9 backend: exactly one supervised respawn, app serves again', async () => {
      const [victim] = backendProcesses(sandbox)
      expect(victim).toBeTruthy()
      process.kill(victim!.pid, 'SIGKILL')
      await expect
        .poll(
          () =>
            backendProcesses(sandbox)
              .map(p => p.pid)
              .filter(pid => pid !== victim!.pid).length,
          {
            timeout: 120_000,
            message: 'a replacement backend is spawned'
          }
        )
        .toBe(1)
      await waitForInteractive(app, page)
      provider.script(U(2), [{ text: [`${A(2)} `, 'after ', 'respawn'] }])
      await send(page, `${U(2)} still there`, 'Enter', ws)
      await finished(U(2))
      session.expectUserMarkers.push(U(2))
      await assertTranscriptOracle(page, ws, provider, session, 'after backend respawn')
      const alive = backendProcesses(sandbox)
      expect(alive.length, `live backends after recovery: ${JSON.stringify(alive)}`).toBe(1)
      // Initial + exactly one replacement, ever — a crash loop or a racing
      // second spawn would add pids here even if they died again.
      expect(seen.size, `backend pids ever seen: ${describeSeen()}`).toBe(2)
    })

    await test.step('quit mid-turn with a running tool child: zero processes remain', async () => {
      const hold = gate()
      provider.script(U(3), [
        {
          text: [`${A(3)} `, 'starting ', 'tool'],
          toolCalls: [{ name: 'terminal', args: { command: `exec -a ${TOOL_TAG} sleep 3600` } }]
        },
        { text: [`${A(3)}b `, 'never ', 'reached'], holdAfterFirstChunk: hold }
      ])
      await send(page, `${U(3)} long tool`, 'Enter', ws)
      await expect
        .poll(() => taggedProcesses(TOOL_TAG).length, { timeout: 120_000, message: 'tool child running' })
        .toBeGreaterThan(0)
      expect(sandboxProcesses(sandbox).length).toBeGreaterThan(0)

      await app.close()
      closed = true
      clearInterval(census)
      await expect
        .poll(
          () =>
            [...sandboxProcesses(sandbox), ...taggedProcesses(TOOL_TAG)].map(
              p => `${p.pid} ${p.cmdline.slice(0, 120)}`
            ),
          {
            timeout: 60_000,
            message: 'no sandbox process (backend, tool child, Electron helper) survives quit'
          }
        )
        .toEqual([])
      hold.open()
    })
  } finally {
    clearInterval(census)

    if (!closed) {
      await app.close().catch(() => undefined)
    }

    // Never leave a test-made process behind even on failure (only our own tag / sandbox).
    for (const proc of [...sandboxProcesses(sandbox), ...taggedProcesses(TOOL_TAG)]) {
      try {
        process.kill(proc.pid, 'SIGKILL')
      } catch {
        /* already gone */
      }
    }

    await provider.close()
    sandbox.cleanup()
  }
})

test('relaunching the same home: one backend per boot, zero after each quit, transcript intact', async () => {
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('relaunch')
  writeProviderHome(sandbox.hermesHome, provider.url)
  const session: OracleTarget = { sessionId: '', expectUserMarkers: [U(1)] }
  let live: Awaited<ReturnType<typeof launchCoreApp>> | null = null

  try {
    for (let launch = 1; launch <= 3; launch++) {
      await test.step(`launch ${launch}`, async () => {
        live = await launchCoreApp(coreAppEnv(sandbox))
        const { app, page } = live
        const ws = recordWebSockets(page)
        await waitForInteractive(app, page)
        await installDuplicateSampler(page)
        await expect
          .poll(() => backendProcesses(sandbox).length, { message: `one backend on launch ${launch}` })
          .toBe(1)

        if (launch === 1) {
          provider.script(U(1), [{ text: [`${A(1)} `, 'persisted ', 'across ', 'launches'] }])
          await send(page, `${U(1)} remember me`, 'Enter', ws)
          await expect
            .poll(() => provider.completions.some(c => c.marker === U(1) && c.finished), { timeout: 120_000 })
            .toBe(true)
          await expect.poll(() => currentSessionId(page)).not.toBe('')
          session.sessionId = await currentSessionId(page)
        } else {
          await page.evaluate(id => {
            window.location.hash = `#/${encodeURIComponent(id)}`
          }, session.sessionId)
          await expect(
            page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
          ).toContainText(A(1), {
            timeout: 60_000
          })
        }

        await assertTranscriptOracle(page, ws, provider, session, `launch ${launch}`)
        await app.close()
        live = null
        await expect
          .poll(() => sandboxProcesses(sandbox).map(p => `${p.pid} ${p.cmdline.slice(0, 120)}`), {
            timeout: 60_000,
            message: `no sandbox process survives quit #${launch}`
          })
          .toEqual([])
      })
    }
  } finally {
    if (live) {
      await (live as Awaited<ReturnType<typeof launchCoreApp>>).app.close().catch(() => undefined)
    }

    for (const proc of sandboxProcesses(sandbox)) {
      try {
        process.kill(proc.pid, 'SIGKILL')
      } catch {
        /* already gone */
      }
    }

    await provider.close()
    sandbox.cleanup()
  }
})
