import { expect, test } from './test'
import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { MOCK_REPLY } from '../../../tests-js/scripts/mock-server'

let fixture: MockBackendFixture
const HELD_PROMPT = 'Keep running while I inspect the board.'

test.beforeAll(async () => {
  fixture = await setupMockBackend({
    modelContextLength: 128_000,
    extraConfig:
      'platform_toolsets:\n  cli: [terminal]\nagent:\n  coding_context: off\nsecurity:\n  allow_lazy_installs: false',
    mockServer: { holdFirstStreamForPrompt: HELD_PROMPT }
  })
  await waitForAppReady(fixture, 120_000)
  // Enable the bundled UI in the isolated Electron profile, as Settings does.
  await fixture.page.evaluate(() => {
    localStorage.setItem('hermes.desktop.pluginDecisions.v2', JSON.stringify({ kanban: true }))
  })
  await fixture.page.reload()
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
})

test('returns to the same chat and gives nested controls the first Escape', async ({}, testInfo) => {
  const page = fixture.page
  const closeButton = page.getByRole('banner').getByRole('button', { name: 'Close', exact: true })
  const composer = page.locator('[contenteditable="true"]').first()
  await composer.click()
  await composer.pressSequentially('Keep this conversation while inspecting Kanban.')
  await page.keyboard.press('Enter')
  await expect(page.getByText(MOCK_REPLY, { exact: true })).toBeVisible({ timeout: 60_000 })
  const previousUrl = page.url()

  const openBoard = async () => {
    await page.getByRole('button', { name: 'Kanban', exact: true }).click()
    await expect(page.getByRole('heading', { name: 'Kanban', exact: true })).toBeVisible()
    await expect(page).toHaveURL(/#\/kanban$/)
  }

  await openBoard()
  await closeButton.click()
  await expect(page).toHaveURL(previousUrl)
  await expect(page.getByText(MOCK_REPLY, { exact: true })).toBeVisible()

  await openBoard()
  await page.keyboard.press('Escape')
  await expect(page).toHaveURL(previousUrl)

  await page.keyboard.press('ControlOrMeta+k')
  await page.getByRole('combobox').fill('Kanban: Open board')
  await expect(page.getByRole('option', { name: 'Kanban: Open board' })).toBeVisible()
  await page.keyboard.press('ArrowDown')
  await page.keyboard.press('Enter')
  await expect(page.getByRole('dialog')).toHaveCount(0)
  await expect(page).toHaveURL(/#\/kanban$/)
  await page.keyboard.press('Escape')
  await expect(page).toHaveURL(previousUrl)

  await openBoard()
  await page.getByRole('button', { name: 'Open settings', exact: true }).click()
  await expect(page).toHaveURL(/#\/settings/)
  const settings = page.locator('[data-overlay-surface]')
  await settings.getByRole('button', { name: 'Close settings', exact: true }).click()
  await expect(page).toHaveURL(/#\/kanban$/)
  await closeButton.click()
  await expect(page).toHaveURL(previousUrl)

  await openBoard()
  await page.getByRole('button', { name: 'New task', exact: true }).first().click()
  await expect(page.getByRole('dialog')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('dialog')).toHaveCount(0)
  await expect(page).toHaveURL(/#\/kanban$/)

  const search = page.getByRole('textbox', { name: 'Filter cards…' })
  await search.click()
  await page.keyboard.press('Escape')
  await expect(page).toHaveURL(/#\/kanban$/)
  await page.getByRole('heading', { name: 'Kanban', exact: true }).click()
  await page.keyboard.press('Escape')
  await expect(page).toHaveURL(previousUrl)

  await openBoard()
  await page.getByRole('button', { name: 'New task', exact: true }).first().click()
  const newTask = page.getByRole('dialog')
  await newTask.getByPlaceholder('Rough idea — a specifier will flesh it out').fill('A parked task')
  await newTask.getByRole('combobox').nth(1).click()
  await page.getByRole('option', { name: "unassigned (parked — won't run)", exact: true }).click()
  await newTask.getByRole('button', { name: 'Create task', exact: true }).click()
  await expect(newTask).toHaveCount(0)
  const card = page.getByText('A parked task', { exact: true })
  await card.click({ modifiers: ['ControlOrMeta'] })
  await expect(page.getByRole('button', { name: 'Clear selection (Esc)' })).toBeVisible()
  await card.click()
  await expect(page.getByRole('heading', { name: 'A parked task' })).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('heading', { name: 'A parked task' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Clear selection (Esc)' })).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('button', { name: 'Clear selection (Esc)' })).toHaveCount(0)
  await expect(page).toHaveURL(/#\/kanban$/)
  await card.click({ button: 'right' })
  await expect(page.getByRole('menu')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('menu')).toHaveCount(0)
  await expect(page).toHaveURL(/#\/kanban$/)
  await card.click({ button: 'right' })
  await page.getByRole('menuitem', { name: 'Move to Ready', exact: true }).click()
  await page.getByRole('heading', { name: 'Kanban', exact: true }).click()
  await page.keyboard.press('Escape')
  await expect(page).toHaveURL(previousUrl)

  // The live counter is a third entry point into the same route.
  await page.getByRole('contentinfo').getByRole('button', { name: '1', exact: true }).click()
  await expect(page).toHaveURL(/#\/kanban$/)
  await closeButton.click()
  await expect(page).toHaveURL(previousUrl)

  // A route tile shares the board component but owns a different close action.
  const openSplitBoard = async () => {
    await page.getByRole('button', { name: 'Kanban', exact: true }).click({ button: 'right' })
    await page.getByRole('menuitem', { name: 'Open in split' }).hover()
    await page.getByRole('menuitem', { name: 'Right', exact: true }).click()
    await expect(page.getByRole('heading', { name: 'Kanban', exact: true })).toBeVisible()
    await expect(page).toHaveURL(previousUrl)
  }

  await openSplitBoard()
  await composer.click()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('heading', { name: 'Kanban', exact: true })).toBeVisible()
  await closeButton.click()
  await expect(page.getByRole('heading', { name: 'Kanban', exact: true })).toHaveCount(0)
  await expect(page.getByText(MOCK_REPLY, { exact: true })).toBeVisible()
  await openSplitBoard()
  // The sidebar menu can restore focus to its trigger. Target this pane
  // explicitly; Escape in the chat above must leave the board alone.
  await page.getByRole('heading', { name: 'Kanban', exact: true }).click()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('heading', { name: 'Kanban', exact: true })).toHaveCount(0)
  await expect(page).toHaveURL(previousUrl)

  // Closing the board must not also stop the conversation behind it.
  await composer.click()
  await composer.pressSequentially(HELD_PROMPT)
  await page.keyboard.press('Enter')
  await fixture.mock.waitForHeldStream()
  await openBoard()
  await page.keyboard.press('Escape')
  await expect(page).toHaveURL(previousUrl)
  await expect(page.locator('[data-slot="composer-root"] button[aria-label="Stop"]')).toBeVisible()
  fixture.mock.releaseHeldStream()
  await expect(page.getByText(MOCK_REPLY, { exact: true })).toHaveCount(2)

  await openBoard()
  await page.screenshot({ path: testInfo.outputPath('kanban-close.png') })
  await closeButton.click()
  await page.screenshot({ path: testInfo.outputPath('returned-conversation.png') })
})
