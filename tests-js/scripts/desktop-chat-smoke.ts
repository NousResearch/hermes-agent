import { randomUUID } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import { expect, type Locator, type Page } from '@playwright/test'
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
}

interface SmokeWindow extends Window {
  hermesDesktop?: { getVersion: () => Promise<ChatIdentity> }
}

export interface TranscriptMessage {
  id: string
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
  hermesRoot: z.string(), platform: z.string(),
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
  const composer = page.locator('[data-slot="composer-root"] [contenteditable="true"], [data-slot="composer-root"] textarea').filter({ visible: true }).first()
  await composer.waitFor({ state: 'visible', timeout: timeoutMs })
  await expect(composer).toBeEditable({ timeout: timeoutMs })
  // Trial input checks hit testing, not merely a non-zero box behind the boot overlay.
  await composer.click({ trial: true, timeout: timeoutMs })
  return composer
}

async function readTranscript(page: Page): Promise<TranscriptMessage[]> {
  return page.locator('[data-slot="aui_thread-viewport"]:visible [data-message-id][data-role]').evaluateAll(
    (nodes: Element[]): TranscriptMessage[] => nodes.map((node: Element): TranscriptMessage => ({
      id: node.getAttribute('data-message-id') ?? '',
      role: node.getAttribute('data-role') ?? '',
      text: node.textContent ?? '',
      streaming: node.getAttribute('data-streaming') === 'true' || Boolean(node.querySelector('[data-message-streaming="true"]')),
      error: Boolean(node.querySelector('[role="alert"]')),
    })),
  )
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

/** Lifecycle belongs to the caller, so this also runs inside the OLD update window. */
export async function runDesktopChatSmoke(page: Page, options: DesktopChatSmokeOptions): Promise<DesktopChatReceipt> {
  const { phase, outDir, expectCommit } = options
  fs.mkdirSync(outDir, { recursive: true })
  const receiptPath = path.join(outDir, `desktop-chat-${phase}.json`)
  const screenshot = path.join(outDir, `desktop-chat-${phase}.png`)
  const prompt = `Hello, can you hear me? Desktop smoke ${phase} ${randomUUID()}`
  const observe = options.observePrompts ?? ((): Promise<string[]> => readMockPrompts(options.mockUrl))
  try {
    const composer = await waitForChatReady(page)
    const identity = await readChatIdentity(page)
    if (expectCommit) { assertChatCommit(identity, expectCommit, options.provenanceCommit) }
    const beforeIds = (await readTranscript(page)).map((message: TranscriptMessage): string => message.id)
    const before = (await observe()).length
    await composer.click()
    await composer.press('ControlOrMeta+A')
    await composer.pressSequentially(prompt)
    await composer.press('Enter')
    await expect.poll(async (): Promise<string> => composer.evaluate((node: HTMLElement): string =>
      node instanceof HTMLTextAreaElement ? node.value : node.textContent ?? ''),
    { timeout: 15_000, message: 'The submitted draft must clear before the idle control proves completion' }).toBe('')
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
    fs.writeFileSync(receiptPath, `${JSON.stringify({ status: 'failed', phase, expectedCommit: expectCommit ?? null, prompt, error: String(error) }, null, 2)}\n`)
    throw error
  }
}