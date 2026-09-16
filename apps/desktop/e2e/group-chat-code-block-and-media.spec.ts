import fs from 'node:fs'
import path from 'node:path'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

// A bot's group-chat reply was rendered through raw Streamdown, whose stock
// code block lays its header and body out as inline siblings: the code body
// sat shifted right by the header's width and the wrapper's overflow-hidden
// clipped the tail with no scrollbar (#91878). The same raw path never ran the
// Desktop's `MEDIA:` transform, so a bot's `MEDIA:/path/file.png` showed as
// a plain path instead of an inline image/player (#93728). Group replies now
// go through the same message renderer as the 1:1 chat.

const SHOT_DIR = '/tmp/batchbots/panes-layout-cron-tile/shots'
// One unbroken 600+ char token: it cannot wrap, so it MUST overflow the
// message column — the probe asserts that overflow exists before asserting
// nothing clips it (a line that fits proves nothing about #91878).
const LONG_LINE = `const veryLongIdentifierNameForTheGroupChatCodeBlockRepro_${'x'.repeat(600)} = 1`
// 1x1 PNG.
const PNG_BASE64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='

let fixture: MockBackendFixture | null = null
let mediaPath = ''

function codeReply(): string {
  return ['Here is the draft:', '', '```ts', LONG_LINE, 'export {}', '```', '', `MEDIA:${mediaPath}`].join('\n')
}

async function openBots(page: MockBackendFixture['page']): Promise<void> {
  const tab = page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()

  await tab.click()
  await expect(page.getByRole('button', { name: 'New bot or group chat' })).toBeVisible()
}

async function createAgent(page: MockBackendFixture['page'], name: string, title: string): Promise<void> {
  await page.getByRole('button', { name: 'New bot or group chat' }).click()
  await page.getByRole('menuitem', { name: 'New Bot' }).click()

  const dialog = page.getByRole('dialog', { name: 'New Bot' })
  await dialog.getByPlaceholder('inbox-triage').fill(name)
  await dialog.getByPlaceholder('Inbox Triage').fill(title)
  await dialog.getByRole('button', { name: 'Create Bot' }).click()
  await expect(dialog).toBeHidden({ timeout: 30_000 })
  await expect(page.getByRole('button', { name: new RegExp(`^${title}\\b`) }).first()).toBeVisible({ timeout: 30_000 })
}

async function createRoom(page: MockBackendFixture['page']) {
  await openBots(page)
  await createAgent(page, 'writer', 'Writer')
  await createAgent(page, 'editor', 'Editor')

  await page.getByRole('button', { name: 'New bot or group chat' }).click()
  await page.getByRole('menuitem', { name: 'New Group Chat' }).click()

  const dialog = page.getByRole('dialog', { name: 'New Group Chat' })

  for (const title of ['Writer', 'Editor']) {
    await dialog.getByText(title, { exact: true }).locator('xpath=ancestor::label').getByRole('checkbox').click()
  }

  await dialog.getByRole('textbox', { name: 'Group name' }).fill('Writer, Editor')
  await dialog.getByRole('button', { name: 'Create Group (2)' }).click()

  const composer = page.getByRole('textbox', { name: 'Message Writer, Editor' }).filter({ visible: true })
  await expect(composer).toBeVisible({ timeout: 20_000 })

  return composer
}

test.beforeEach(async () => {
  fixture = await setupMockBackend({
    mockServer: {
      replyForPrompt: prompt => (prompt.includes('CODE_BLOCK_REPRO') ? codeReply() : 'ok')
    }
  })
  mediaPath = path.join(fixture.sandbox.hermesHome, 'group-media.png')
  fs.writeFileSync(mediaPath, Buffer.from(PNG_BASE64, 'base64'))
  fs.mkdirSync(SHOT_DIR, { recursive: true })
  await waitForAppReady(fixture, 120_000)
})

test.afterEach(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('a group reply renders its code block inside the message and its MEDIA: line as inline media', async () => {
  test.setTimeout(300_000)
  const page = fixture!.page
  const composer = await createRoom(page)

  await composer.fill('@writer CODE_BLOCK_REPRO')
  await composer.press('Enter')

  // A bot created moments ago runs its intro turn in the background; when it
  // lands, the roster fronts that bot's chat tab and yanks the center away
  // from the room. Re-select the room and read the reply from a room body.
  const roomTab = page.getByRole('tab', { name: /Writer, Editor Close/ })

  const code = page
    .locator('[data-selectable-text="true"]')
    .getByText(/veryLongIdentifierNameForTheGroupChatCodeBlockRepro/)
    .filter({ visible: true })
    .first()

  await expect(async () => {
    if ((await roomTab.getAttribute('aria-selected')) !== 'true') {
      await roomTab.click()
    }

    await expect(code).toBeVisible({ timeout: 5_000 })
  }).toPass({ timeout: 90_000 })
  await expect(page.getByRole('button', { name: 'Stop', exact: true })).toHaveCount(0, { timeout: 60_000 })
  await page.screenshot({ path: path.join(SHOT_DIR, 'group-chat-code-media.png') })

  const geometry = await code.evaluate(el => {
    const body = el.closest<HTMLElement>('[data-selectable-text="true"]')!
    const bodyRect = body.getBoundingClientRect()
    const pre = el.closest('pre') ?? el
    const preRect = pre.getBoundingClientRect()
    // Every box between the code text and the message body: a box whose
    // content is wider than itself must scroll, never clip — that is the
    // "cut off with no horizontal scrollbar" the report describes.
    const chain: Array<{ tag: string; overflowX: string; scrollWidth: number; clientWidth: number }> = []
    let node: HTMLElement | null = el as HTMLElement

    while (node && node !== body.parentElement) {
      chain.push({
        tag: node.tagName.toLowerCase(),
        overflowX: getComputedStyle(node).overflowX,
        scrollWidth: node.scrollWidth,
        clientWidth: node.clientWidth
      })
      node = node.parentElement
    }

    const clippedWithoutScroll = chain.filter(
      box => box.scrollWidth > box.clientWidth + 1 && !['auto', 'scroll'].includes(box.overflowX)
    )

    const scrollingPre = chain.find(
      box => box.tag === 'pre' && box.scrollWidth > box.clientWidth + 1 && ['auto', 'scroll'].includes(box.overflowX)
    )

    return {
      body: { left: bodyRect.left, right: bodyRect.right },
      pre: { left: preRect.left, right: preRect.right, width: preRect.width },
      chain,
      clippedWithoutScroll,
      scrollingPre,
      rawMedia: body.textContent?.includes('MEDIA:') ?? false,
      inlineMedia: Boolean(body.querySelector('img, audio, video'))
    }
  })

  console.log('GROUP CODE BLOCK GEOMETRY', JSON.stringify(geometry))
  // The fixture really overflows and the code block scrolls to show it...
  expect.soft(geometry.scrollingPre).toBeDefined()
  // ...while staying inside the message column...
  expect.soft(geometry.pre.right).toBeLessThanOrEqual(geometry.body.right + 1)
  // ...and no box between the code and the message clips text it cannot scroll.
  expect.soft(geometry.clippedWithoutScroll).toEqual([])
  // MEDIA: renders as media, never as the raw directive.
  expect.soft(geometry.rawMedia).toBe(false)
  expect.soft(geometry.inlineMedia).toBe(true)
})
