import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'
import type { Locator, Page } from '@playwright/test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

let fixture: MockBackendFixture | undefined
let home: string

test.beforeAll(async () => {
  // Profile creation is HOME-anchored, not HERMES_HOME-anchored.
  home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-group-resize-'))
  const priorHome = process.env.HOME
  process.env.HOME = home
  try {
    fixture = await setupMockBackend()
  } finally {
    if (priorHome === undefined) delete process.env.HOME
    else process.env.HOME = priorHome
  }
  await fixture.app.evaluate(({ BrowserWindow }) => {
    for (const window of BrowserWindow.getAllWindows()) window.showInactive()
  })
  await waitForAppReady(fixture, 60_000)
})

test.afterAll(async () => {
  try {
    await fixture?.cleanup()
  } finally {
    if (home) fs.rmSync(home, { recursive: true, force: true })
  }
})

async function dragCorner(page: Page, input: Locator, dy: number) {
  await input.scrollIntoViewIfNeeded()
  await expect
    .poll(
      () =>
        input.evaluate(el => {
          const box = el.getBoundingClientRect()
          return document.elementFromPoint(box.right - 3, box.bottom - 3) === el
        }),
      { timeout: 20_000 }
    )
    .toBe(true)
  const box = (await input.boundingBox())!
  const x = box.x + box.width - 3
  const y = box.y + box.height - 3
  await page.mouse.move(x, y)
  await page.mouse.down()
  await page.mouse.move(x + 30, y + dy, { steps: 12 })
  await page.mouse.up()
}

async function verifyResize(page: Page, input: Locator) {
  const initial = (await input.boundingBox())!
  await dragCorner(page, input, 100)
  await expect.poll(async () => (await input.boundingBox())!.height).toBeGreaterThan(initial.height + 30)
  const grown = (await input.boundingBox())!
  expect(grown.width).toBeCloseTo(initial.width, 0)

  await input.fill('A multiline draft')
  await input.press('Shift+Enter')
  await input.pressSequentially('Still editing')
  await expect(input).toHaveValue('A multiline draft\nStill editing')
  expect((await input.boundingBox())!.height).toBeCloseTo(grown.height, 0)

  await input.fill(Array.from({ length: 80 }, (_, i) => `Line ${i}`).join('\n'))
  expect(await input.evaluate(el => el.scrollHeight > el.clientHeight)).toBe(true)
  await input.evaluate(el => {
    el.scrollTop = 30
  })
  expect(await input.evaluate(el => el.scrollTop)).toBeGreaterThan(0)

  await dragCorner(page, input, -60)
  expect((await input.boundingBox())!.height).toBeLessThan(grown.height)
  await dragCorner(page, input, -1000)
  const minimum = await input.evaluate(el => parseFloat(getComputedStyle(el).minHeight))
  expect(minimum).toBe(36)
  expect((await input.boundingBox())!.height).toBeCloseTo(minimum, 0)
  await dragCorner(page, input, 1000)
  const maximum = await input.evaluate(el => parseFloat(getComputedStyle(el).maxHeight))
  expect(maximum).toBeCloseTo(await page.evaluate(() => innerHeight * 0.4), 0)
  expect((await input.boundingBox())!.height).toBeCloseTo(maximum, 0)
  // Scrollbars may narrow the transcript; width must follow its layout,
  // never an inline width written by the native drag.
  expect(await input.evaluate(el => el.style.width)).toBe('')
  expect(await input.evaluate(el => el.getBoundingClientRect().width)).toBeCloseTo(
    await input.evaluate(el => el.parentElement!.getBoundingClientRect().width),
    0
  )
  console.log('resize verified', {
    label: await input.getAttribute('aria-label'),
    initial: initial.height,
    grown: grown.height,
    minimum,
    maximum
  })
  await input.fill('')
}

test('new-thread and reply composers resize vertically without resetting drafts or width', async () => {
  test.setTimeout(240_000)
  const page = fixture!.page
  await page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()
    .click()
  for (const name of ['alpha', 'builder']) {
    await page.getByRole('button', { name: 'New bot or group chat' }).click()
    await page.getByRole('menuitem', { name: /^New bot$/i }).click()
    const dialog = page.getByRole('dialog', { name: /^New bot$/i })
    await dialog.getByPlaceholder('inbox-triage').fill(name)
    await dialog.getByPlaceholder('Inbox Triage').fill(name)
    await dialog.getByRole('button', { name: 'Create Bot', exact: true }).click()
    await expect(dialog).toBeHidden({ timeout: 30_000 })
  }
  await page.getByRole('button', { name: 'New bot or group chat' }).click()
  await page.getByRole('menuitem', { name: /^New group chat$/i }).click()
  const dialog = page.getByRole('dialog', { name: /^New group chat$/i })
  for (const name of ['alpha', 'builder']) {
    await dialog.getByText(name, { exact: true }).first().locator('xpath=ancestor::label').getByRole('checkbox').click()
  }
  await dialog.getByRole('textbox', { name: 'Group name' }).fill('Resize probe')
  await dialog.getByRole('button', { name: 'Create Group (2)' }).click()
  const main = page.getByRole('textbox', { name: 'Message Resize probe', exact: true })
  await expect(main).toBeVisible()
  await verifyResize(page, main)
  await main.fill('Hello')
  await main.press('Enter')
  await expect(main).toHaveValue('')
  await page.getByRole('button', { name: 'Reply in thread', exact: true }).first().click()
  const reply = page.getByRole('textbox', { name: 'Reply in thread', exact: true })
  await verifyResize(page, reply)
  await fixture!.app.evaluate(({ BrowserWindow }) => {
    for (const window of BrowserWindow.getAllWindows()) window.setContentSize(1024, 600)
  })
  for (const input of [main, reply]) {
    await expect
      .poll(() => input.evaluate(el => el.getBoundingClientRect().height))
      .toBeCloseTo(await page.evaluate(() => innerHeight * 0.4), 0)
    await input.fill('@alp')
    await input.press('Tab')
    await expect(input).toHaveValue('@alpha ')
    await input.dispatchEvent('keydown', { key: 'Enter', isComposing: true })
    await input.dispatchEvent('keydown', { key: 'Enter', keyCode: 229 })
    await expect(input).toHaveValue('@alpha ')
  }
  await page.screenshot({ path: test.info().outputPath('resizable-group-composers.png') })
})
