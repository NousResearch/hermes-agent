import { expect, test } from './test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'

let fixture: MockBackendFixture | null = null

async function openBots(page: MockBackendFixture['page']): Promise<void> {
  const tab = page.getByRole('button', { name: 'Bots', exact: true }).or(page.getByRole('tab', { name: 'Bots', exact: true })).first()
  await tab.click()
  await expect(page.getByRole('button', { name: 'New agent or group chat' })).toBeVisible()
}

async function createAgent(page: MockBackendFixture['page'], name: string, title: string): Promise<void> {
  await page.getByRole('button', { name: 'New agent or group chat' }).click()
  await page.getByRole('menuitem', { name: 'New Agent' }).click()

  const dialog = page.getByRole('dialog', { name: 'New Agent' })
  await dialog.getByPlaceholder('inbox-triage').fill(name)
  await dialog.getByPlaceholder('Inbox Triage').fill(title)
  await dialog.getByRole('button', { name: 'Create Agent' }).click()
  await expect(dialog).toBeHidden({ timeout: 30_000 })
  await expect(page.getByRole('button', { name: new RegExp(`^${title}\\b`) }).first()).toBeVisible({ timeout: 30_000 })
}

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('local bot replaces an open group main workspace', async () => {
  test.setTimeout(180_000)
  const page = fixture!.page

  await openBots(page)
  await createAgent(page, 'programmer', 'Programmer')
  await createAgent(page, 'reviewer', 'Reviewer')

  await page.getByRole('button', { name: 'New agent or group chat' }).click()
  await page.getByRole('menuitem', { name: 'New Group Chat' }).click()

  const dialog = page.getByRole('dialog', { name: 'New Group Chat' })
  for (const title of ['Programmer', 'Reviewer']) {
    await dialog.getByText(title, { exact: true }).locator('xpath=ancestor::label').getByRole('checkbox').click()
  }
  await dialog.getByRole('textbox', { name: 'Group name' }).fill('Programmer, Reviewer')
  await dialog.getByRole('button', { name: 'Create Group (2)' }).click()

  const groupTitle = page.getByText('Programmer, Reviewer — group chat', { exact: true }).filter({ visible: true })
  const groupComposer = page.getByRole('textbox', { name: 'Message Programmer, Reviewer' }).filter({ visible: true })
  await expect(groupTitle).toBeVisible({ timeout: 20_000 })
  await expect(groupComposer).toBeVisible()

  const programmer = page.getByRole('button', { name: /^Programmer\b/ }).filter({ visible: true }).first()
  await programmer.click()

  await expect(groupTitle).toHaveCount(0, { timeout: 20_000 })
  await expect(groupComposer).toHaveCount(0)
  await expect(page.getByRole('tab', { name: /New session/ }).filter({ visible: true })).toBeVisible()
  await expect(page.locator('[data-slot="composer-root"] [contenteditable="true"]').filter({ visible: true }).first()).toBeVisible()
  await expect
    .poll(() =>
      page
        .getByText('Programmer', { exact: true })
        .filter({ visible: true })
        .evaluateAll(nodes => nodes.some(node => node.getBoundingClientRect().left > window.innerWidth * 0.65)),
    )
    .toBe(true)
})
