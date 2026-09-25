/**
 * C20 core: blocking interactive prompts round-trip through the real chain.
 *
 * Real Electron + real `hermes serve`, approvals in manual mode; only the LLM
 * is faked. For each prompt kind the invariant is end to end, not "a card
 * rendered":
 *  - clarify: exactly one card; the choice the user clicks is exactly what
 *    the model receives in the tool result on its next request; the turn
 *    completes; transcript oracle holds.
 *  - approval (run once): exactly one approval card; the gated command has
 *    NOT run before the click and HAS run after it (observable side effect);
 *    the turn completes; transcript oracle holds.
 *  - approval (deny): the command never runs; the turn still completes (no
 *    wedged busy state); transcript oracle holds.
 */

import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, type Page, test } from '@playwright/test'

import {
  coreAppEnv,
  createCoreSandbox,
  currentSessionId,
  launchCoreApp,
  recordWebSockets,
  send,
  waitForInteractive,
  writeProviderHome
} from './harness'
import { assertTranscriptOracle, installDuplicateSampler, type OracleTarget } from './oracle'
import { type RecordedCompletion, startScriptedProvider } from './provider'

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-${nonce}`
const A = (n: number) => `A${n}-${nonce}`
const AI = (n: number) => `A${n}i-${nonce}`

function viewport(page: Page) {
  return page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
}

/** Text of the tool results the model received in `completion`'s request. */
function toolResults(completion: RecordedCompletion | undefined): string {
  const messages: any[] = completion?.body?.messages ?? []

  return messages
    .filter(m => m?.role === 'tool')
    .map(m => (typeof m.content === 'string' ? m.content : JSON.stringify(m.content)))
    .join('\n')
}

test('clarify and approval prompts round-trip exactly once', async () => {
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('prompts')
  writeProviderHome(sandbox.hermesHome, provider.url, '', 'manual')
  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))
  const ws = recordWebSockets(page)
  const session: OracleTarget = { sessionId: '', expectUserMarkers: [] }

  const finished = (marker: string, step = 0) =>
    expect
      .poll(() => provider.completions.some(c => c.marker === marker && c.step === step && c.finished), {
        timeout: 120_000,
        message: `provider finished ${marker} step ${step}`
      })
      .toBe(true)

  const completion = (marker: string, step: number) =>
    provider.completions.find(c => c.marker === marker && c.step === step)

  try {
    await waitForInteractive(app, page)
    await installDuplicateSampler(page)

    await test.step('clarify: the clicked choice is exactly what the model receives', async () => {
      const question = `Q1-${nonce} which drink`
      const pick = `mate-${nonce}`
      const other = `chai-${nonce}`
      provider.script(U(1), [
        {
          text: [`${AI(1)} `, 'asking'],
          toolCalls: [{ name: 'clarify', args: { questions: [{ question, choices: [other, pick] }] } }]
        },
        { text: [`${A(1)} `, 'noted ', 'your ', 'choice'] }
      ])
      await send(page, `${U(1)} ask me`, 'Enter', ws)
      // The open (unanswered) clarify form: single-question or batch shape.
      const openForms = page.locator('form[data-clarify-choices], form[data-clarify-batch]').filter({ visible: true })
      await expect(openForms).toHaveCount(1, { timeout: 120_000 })
      await expect(viewport(page).getByText(question)).toHaveCount(1)
      await openForms.getByRole('button', { name: new RegExp(pick) }).click()
      const submit = openForms.locator('button[type="submit"]')
      await expect(submit).toBeEnabled()
      await submit.click()
      await finished(U(1), 1)
      const received = toolResults(completion(U(1), 1))
      expect(received, 'model received the clicked choice').toContain(pick)
      expect(received, 'model did not receive the other choice as the answer').not.toMatch(
        new RegExp(`answer[^\\n]*${other}`, 'i')
      )
      await expect(
        page.locator('form[data-clarify-choices], form[data-clarify-batch]').filter({ visible: true })
      ).toHaveCount(0)
      await expect(page.locator('[data-clarify-settled]').filter({ visible: true })).toHaveCount(1)
      await expect.poll(() => currentSessionId(page)).not.toBe('')
      session.sessionId = await currentSessionId(page)
      session.expectUserMarkers.push(U(1))
      await assertTranscriptOracle(page, ws, provider, session, 'clarify round trip')
      await expect(viewport(page).getByText(question)).toHaveCount(1)
    })

    await test.step('approval (run once): the command runs only after the click', async () => {
      const victim = path.join(sandbox.root, `victim-run-${nonce}`)
      fs.mkdirSync(victim)
      provider.script(U(2), [
        {
          text: [`${AI(2)} `, 'needs ', 'approval'],
          toolCalls: [{ name: 'terminal', args: { command: `rm -rf ${victim}` } }]
        },
        { text: [`${A(2)} `, 'deleted ', 'it'] }
      ])
      await send(page, `${U(2)} delete run dir`, 'Enter', ws)
      const run = page.locator('[data-approval-run]').filter({ visible: true })
      await expect(run).toHaveCount(1, { timeout: 120_000 })
      expect(fs.existsSync(victim), 'gated command must not run before approval').toBe(true)
      await run.click()
      await finished(U(2), 1)
      expect(fs.existsSync(victim), 'approved command ran').toBe(false)
      await expect(page.locator('[data-approval-run]').filter({ visible: true })).toHaveCount(0)
      session.expectUserMarkers.push(U(2))
      await assertTranscriptOracle(page, ws, provider, session, 'approval run once')
    })

    await test.step('approval (deny): the command never runs, the turn still completes', async () => {
      const victim = path.join(sandbox.root, `victim-deny-${nonce}`)
      fs.mkdirSync(victim)
      provider.script(U(3), [
        {
          text: [`${AI(3)} `, 'needs ', 'approval'],
          toolCalls: [{ name: 'terminal', args: { command: `rm -rf ${victim}` } }]
        },
        { text: [`${A(3)} `, 'left ', 'it ', 'alone'] }
      ])
      await send(page, `${U(3)} delete deny dir`, 'Enter', ws)
      const deny = page.locator('[data-approval-deny]').filter({ visible: true })
      await expect(deny).toHaveCount(1, { timeout: 120_000 })
      await deny.click()
      await finished(U(3), 1)
      expect(fs.existsSync(victim), 'denied command never ran').toBe(true)
      expect(toolResults(completion(U(3), 1)), 'the denial reached the model as the tool result').not.toBe('')
      await expect(page.locator('[data-approval-deny]').filter({ visible: true })).toHaveCount(0)
      session.expectUserMarkers.push(U(3))
      await assertTranscriptOracle(page, ws, provider, session, 'approval deny')
    })
  } finally {
    await app.close().catch(() => undefined)
    await provider.close()
    sandbox.cleanup()
  }
})
