import { randomUUID } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import { type ConsoleMessage, expect, type Locator, type Page } from '@playwright/test'
import { z } from 'zod'

import { validateMockUrl } from './mock-provider-config.ts'
import { MOCK_REPLY } from './mock-server.ts'

export type ChatPhase = 'old' | 'new' | 'installed'

export interface DesktopChatSmokeOptions {
  mockUrl: string
  phase: ChatPhase
  outDir: string
  expectCommit?: string
  provenanceCommit?: string
  observePrompts?: () => Promise<readonly string[]>
}

export interface ChatIdentity {
  appVersion: string
  commit: string | null
  hermesRoot: string
  platform: string
  /** The resolved Hermes home (newer desktops report it; absent on older ones). */
  hermesHome?: string
}

interface SmokeWindow extends Window {
  hermesDesktop?: { getVersion: () => Promise<ChatIdentity> }
}

export interface TranscriptMessage {
  id: string
  /** Historical user roots omit the message ID; label their content identity honestly. */
  idSource?: 'legacy-user-text'
  role: string
  text: string
  streaming: boolean
  error: boolean
}

export interface DesktopChatReceipt {
  status: 'passed'
  phase: ChatPhase
  expectedCommit: string | null
  identity: ChatIdentity
  provenanceCommit: string | null
  prompt: string
  requestWitness: { index: number; prompt: string; before: number }
  transcript: { beforeIds: string[]; user: TranscriptMessage; assistant: TranscriptMessage }
  turnComplete: true
  screenshot: string
}

const identitySchema = z.object({
  appVersion: z.string(), commit: z.string().nullable().optional(),
  hermesRoot: z.string(), platform: z.string(), hermesHome: z.string().optional(),
})

export async function readChatIdentity(page: Page): Promise<ChatIdentity> {
  const identity = identitySchema.parse(await page.evaluate(async (): Promise<ChatIdentity> => {
    // SAFETY: the driver uses a desktop page; the returned IPC data is parsed outside the renderer.
    const bridge = (window as SmokeWindow).hermesDesktop

    if (!bridge) {
      throw new Error('Desktop version bridge is unavailable')
    }

    return bridge.getVersion()
  }))

  return { ...identity, commit: identity.commit ?? null }
}

/** Old desktops do not report a commit; their caller must verify the installed runtime. */
export function assertChatCommit(identity: ChatIdentity, expected: string, provenanceCommit?: string): void {
  if (identity.commit !== null && identity.commit !== expected) {
    throw new Error(`Running commit ${identity.commit} does not equal expected ${expected}`)
  }

  if (provenanceCommit !== undefined && provenanceCommit !== expected) {
    throw new Error(`Verified installation commit ${provenanceCommit} does not equal expected ${expected}`)
  }

  if (identity.commit === null && provenanceCommit !== expected) {
    throw new Error('Desktop does not report a commit; verified installation provenance is required')
  }
}

export async function readMockPrompts(mockUrl: string): Promise<string[]> {
  const response = await fetch(`${validateMockUrl(mockUrl)}/__e2e__/prompts`, {
    signal: AbortSignal.timeout(5000), redirect: 'error',
  })

  if (!response.ok) {
    throw new Error(`Mock request witness returned HTTP ${response.status}`)
  }

  return z.object({ receivedPrompts: z.array(z.string()) }).parse(await response.json()).receivedPrompts
}

export async function waitForChatReady(page: Page, timeoutMs = 120_000): Promise<Locator> {
  // The visible editor is a contentEditable div. assistant-ui also renders an
  // aria-hidden, sr-only <textarea> that carries the composer binding: it is
  // "editable" but clipped out of the viewport, so a bare `textarea` selector
  // latches onto it, sails through toBeEditable, and then can never be
  // hit-tested (225 trial-click retries, then "element is outside of the
  // viewport"). Require the editor; keep a textarea fallback only for a real,
  // non-hidden input.
  const root = page.locator('[data-slot="composer-root"]')

  const composer = root
    .locator('[contenteditable="true"]:visible, textarea:not([aria-hidden="true"]):not(.sr-only):visible')
    .first()

  try {
    await composer.waitFor({ state: 'visible', timeout: timeoutMs })
    await expect(composer).toBeEditable({ timeout: timeoutMs })
    // Trial input checks hit testing, not merely a non-zero box behind the boot overlay.
    await composer.click({ trial: true, timeout: timeoutMs })
  } catch (error) {
    throw new Error(`${(error as Error).message} -- composer not interactable `
      + `(composer-root=${await root.count()}, contenteditable=${await root.locator('[contenteditable]').count()}): `
      + await composerDiagnostics(root))
  }

  return composer
}

/** What the composer actually contains, for a failure that explains itself. */
async function composerDiagnostics(root: Locator): Promise<string> {
  try {
    if (await root.count() === 0) {
      return '(no [data-slot="composer-root"] in the DOM)'
    }

    return (await root.first().evaluate((el: Element): string => el.outerHTML.slice(0, 1500)))
  } catch (error) {
    return `(diagnostics unavailable: ${(error as Error).message})`
  }
}

/** The composer's text, whether the app rendered it as a contentEditable or a real textarea. */
export async function composerText(composer: Locator): Promise<string> {
  return composer.evaluate((node: HTMLElement): string =>
    node instanceof HTMLTextAreaElement ? node.value : node.textContent ?? '')
}

export function transcriptMessages(viewports: Element[]): TranscriptMessage[] {
  return viewports.flatMap((viewport: Element): TranscriptMessage[] =>
    [...viewport.querySelectorAll('[data-message-id][data-role], [data-slot="aui_user-message-root"][data-role="user"]')]
      .map((node: Element): TranscriptMessage => {
        const id = node.getAttribute('data-message-id') ?? ''
        const role = node.getAttribute('data-role') ?? ''
        const text = node.textContent ?? ''
        // v2026.6.19's sticky user wrapper drops MessagePrimitive.Root's ID.
        // Its full text is stable across remounts, unlike a DOM index/handle.
        // The new nonce and ordered, fresh assistant ID remain mandatory.
        const legacyUser = !id && role === 'user' && node.getAttribute('data-slot') === 'aui_user-message-root'

        const message: TranscriptMessage = {
          id: legacyUser ? `legacy-user-text:${text}` : id,
          role, text,
          streaming: node.getAttribute('data-streaming') === 'true' || Boolean(node.querySelector('[data-message-streaming="true"]')),
          error: Boolean(node.querySelector('[role="alert"]')),
        }

        if (legacyUser) { message.idSource = 'legacy-user-text' }

        return message
      }))
}

async function readTranscript(page: Page): Promise<TranscriptMessage[]> {
  return page.locator('[data-slot="aui_thread-viewport"]:visible').evaluateAll(transcriptMessages)
}

/** Match an ordered new pair, never a reply carried over from an earlier checkpoint. */
export function newCompletedPair(
  messages: TranscriptMessage[], beforeIds: readonly string[], prompt: string,
): { user: TranscriptMessage; assistant: TranscriptMessage } | null {
  const userIndex = messages.findIndex((message: TranscriptMessage): boolean =>
    message.role === 'user' && message.text.includes(prompt) && !beforeIds.includes(message.id))

  const user = messages[userIndex]
  const assistant = messages[userIndex + 1]

  if (!user || !user.id || !assistant?.id || assistant.role !== 'assistant'
      || beforeIds.includes(assistant.id) || !assistant.text.includes(MOCK_REPLY)
      || assistant.streaming || assistant.error) {
    return null
  }

  return { user, assistant }
}

/** Evidence before assertion: what the renderer held when a checkpoint failed. */
async function rendererEvidence(page: Page, consoleLines: readonly string[]): Promise<string> {
  const state = await page.evaluate((): Record<string, unknown> => {
    const viewport = document.querySelector('[data-slot="aui_thread-viewport"]')
    const composer = document.querySelector('[data-slot="composer-root"]')

    return {
      url: location.href,
      threadMessageCount: document.querySelectorAll('[data-message-id][data-role]').length,
      threadText: viewport?.textContent?.slice(0, 2000) ?? null,
      composerHtml: composer?.outerHTML?.slice(0, 2000) ?? null,
      alerts: [...document.querySelectorAll('[role="alert"]')].map((node: Element): string => node.textContent?.slice(0, 400) ?? ''),
    }
  }).catch((error: Error): Record<string, unknown> => ({ error: String(error) }))

  return ['--- renderer state ---', JSON.stringify(state, null, 2), '--- renderer console ---', ...consoleLines].join('\n')
}

/** Lifecycle belongs to the caller, so this also runs inside the OLD update window. */
export async function runDesktopChatSmoke(page: Page, options: DesktopChatSmokeOptions): Promise<DesktopChatReceipt> {
  const { phase, outDir, expectCommit } = options
  fs.mkdirSync(outDir, { recursive: true })
  const receiptPath = path.join(outDir, `desktop-chat-${phase}.json`)
  const evidencePath = path.join(outDir, `desktop-chat-${phase}-renderer.log`)
  const screenshot = path.join(outDir, `desktop-chat-${phase}.png`)
  const prompt = `Hello, can you hear me? Desktop smoke ${phase} ${randomUUID()}`
  const observe = options.observePrompts ?? ((): Promise<string[]> => readMockPrompts(options.mockUrl))
  // A send the app swallows and a send the app never made look identical from
  // the mock's side; the renderer's own console is the only witness to which.
  const consoleLines: string[] = []

  const onConsole = (message: ConsoleMessage): void => {
    if (consoleLines.length >= 200) {
      return
    }

    const type = message.type()

    if (type === 'debug') {
      return
    }

    consoleLines.push(`[${type}] ${message.text().slice(0, 500)}`)
  }

  page.on('console', onConsole)

  try {
    const composer = await waitForChatReady(page)
    const identity = await readChatIdentity(page)

    if (expectCommit) { assertChatCommit(identity, expectCommit, options.provenanceCommit) }
    const beforeIds = (await readTranscript(page)).map((message: TranscriptMessage): string => message.id)
    const before = (await observe()).length
    await composer.click()
    // The app persists its composer draft across launches, so a checkpoint that
    // types on top of a restored draft can submit the PREVIOUS checkpoint's text
    // (proved: the update window's turn carried the root checkpoint's prompt) and
    // the witness never matches. Clear it deliberately and refuse to type until
    // the composer is provably empty.
    await composer.press('ControlOrMeta+A')
    await composer.press('Delete')
    await expect.poll(async (): Promise<string> => composerText(composer), {
      timeout: 15_000, message: 'The composer must be empty before the checkpoint types (a restored draft must not survive)',
    }).toBe('')
    await composer.pressSequentially(prompt)
    // The composer clears on submit whether or not a turn was ever started, so
    // the clear-poll below passes vacuously when the editor refused the input.
    // Prove the typing landed before trusting anything downstream of Enter.
    await expect.poll(async (): Promise<string> => composerText(composer), {
      timeout: 15_000, message: 'The composer must hold the typed prompt before Enter (did the editor accept input?)',
    }).toContain(prompt)
    await composer.press('Enter')
    await expect.poll(async (): Promise<string> => composerText(composer),
    { timeout: 90_000, message: 'The submitted draft must clear before the idle control proves completion' }).toBe('')
    let witnessIndex = -1
    let receivedPrompt = ''
    await expect.poll(async (): Promise<boolean> => {
      const prompts = await observe()
      witnessIndex = prompts.findIndex((text: string, index: number): boolean => index >= before && text.includes(prompt))
      receivedPrompt = prompts[witnessIndex] ?? ''

      return witnessIndex >= before
    }, { timeout: 90_000, message: 'The mock must receive this checkpoint prompt after the send' }).toBe(true)
    await expect.poll(async (): Promise<boolean> => newCompletedPair(await readTranscript(page), beforeIds, prompt) !== null,
      { timeout: 90_000, message: 'A new completed assistant reply must follow the new user message' }).toBe(true)
    // An empty idle composer offers voice chat; older desktops retain Send.
    // Stop and the streaming marker must settle even when all reply text arrived.
    await expect(page.locator('[data-slot="composer-root"] button[aria-label="Start voice conversation"]:visible, [data-slot="composer-root"] button[type="submit"][aria-label="Send"]:visible')).toBeVisible({ timeout: 90_000 })
    const pair = newCompletedPair(await readTranscript(page), beforeIds, prompt)

    if (!pair) {
      throw new Error('New transcript pair disappeared before the turn completed')
    }

    await page.screenshot({ path: screenshot })

    const receipt: DesktopChatReceipt = {
      status: 'passed', phase, expectedCommit: expectCommit ?? null, identity, provenanceCommit: options.provenanceCommit ?? null, prompt,
      requestWitness: { index: witnessIndex, prompt: receivedPrompt, before },
      transcript: { beforeIds, ...pair }, turnComplete: true, screenshot,
    }

    fs.writeFileSync(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`)

    return receipt
  } catch (error) {
    await page.screenshot({ path: screenshot }).catch((): void => {})
    fs.writeFileSync(evidencePath, `${await rendererEvidence(page, consoleLines)}\n`)
    fs.writeFileSync(receiptPath, `${JSON.stringify({ status: 'failed', phase, expectedCommit: expectCommit ?? null, prompt, error: String(error) }, null, 2)}\n`)
    throw error
  } finally {
    page.off('console', onConsole)
  }
}