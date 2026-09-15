import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

let fixture: MockBackendFixture | null = null

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

test.beforeAll(async () => {
  fixture = await setupMockBackend()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('group mentions cover expanded posts in light and dark mode', async () => {
  test.setTimeout(240_000)
  const page = fixture!.page
  await openBots(page)
  await createAgent(page, 'programmer', 'Programmer')
  await createAgent(page, 'reviewer', 'Reviewer')
  await page.getByRole('button', { name: 'New bot or group chat' }).click()
  await page.getByRole('menuitem', { name: 'New Group Chat' }).click()
  const dialog = page.getByRole('dialog', { name: 'New Group Chat' })
  for (const title of ['Programmer', 'Reviewer']) {
    await dialog.getByText(title, { exact: true }).locator('xpath=ancestor::label').getByRole('checkbox').click()
  }
  await dialog.getByRole('textbox', { name: 'Group name' }).fill('Mention surface')
  await dialog.getByRole('button', { name: 'Create Group (2)' }).click()
  const composer = page.getByRole('textbox', { name: 'Message Mention surface', exact: true }).filter({ visible: true })
  await expect(composer).toBeVisible()
  await composer.fill(
    Array.from({ length: 32 }, (_, i) => `Post line ${i + 1}: content behind the mention menu.`).join('\n\n')
  )
  await composer.press('Enter')
  await expect(page.getByRole('button', { name: 'Collapse thread', exact: true })).toBeVisible()

  for (const mode of ['light', 'dark']) {
    // The real theme switch updates the semantic surface tokens, not just a
    // class on the menu. The shortcut is registered by the desktop shell.
    const dark = await page.locator('html').evaluate(el => el.classList.contains('dark'))
    if (dark !== (mode === 'dark')) {
      await composer.press('Tab')
      await page.keyboard.press('Shift+X')
    }
    await expect(page.locator('html')).toHaveClass(mode === 'dark' ? /dark/ : /^(?!.*\bdark\b)/)
    await composer.fill('')
    await composer.fill('@')
    const option = page.getByRole('button', { name: /^@everyone / }).filter({ visible: true })
    await expect(option).toBeVisible()
    const painted = await option.evaluate(el => {
      const menu = el.parentElement!
      const style = getComputedStyle(menu)
      const canvas = document.createElement('canvas')
      canvas.width = canvas.height = 1
      const ctx = canvas.getContext('2d')!
      ctx.fillStyle = style.backgroundColor
      ctx.fillRect(0, 0, 1, 1)
      const rect = el.getBoundingClientRect()
      return {
        alpha: ctx.getImageData(0, 0, 1, 1).data[3],
        receivesPointer: el.contains(document.elementFromPoint(rect.left + 8, rect.top + rect.height / 2))
      }
    })
    expect(painted.alpha, `${mode}: transcript must not show through the menu`).toBe(255)
    expect(painted.receivesPointer).toBe(true)
    await page.screenshot({ path: test.info().outputPath(`mentions-${mode}.png`) })
    await composer.press('ArrowDown')
    await composer.press('Enter')
    await expect(composer).toHaveValue('@all ')
    await composer.fill('@')
    await composer.press('Escape')
    await expect(option).toBeHidden()
  }
})
