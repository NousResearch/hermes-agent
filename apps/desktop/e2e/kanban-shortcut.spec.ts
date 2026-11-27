import fs from 'node:fs'
import path from 'node:path'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { buildAppEnv, createSandbox, launchDesktop, waitForAppReady } from './fixtures'
import { expect, test } from './test'

test('Show Kanban shortcut opens the remembered board without changing operational defaults', async () => {
  test.setTimeout(180_000)
  const mock = await startMockServer()
  const sandbox = createSandbox('kanban-shortcut')
  writeMockProviderConfig(sandbox.hermesHome, mock.url, undefined, 'kanban:\n  dispatch_in_gateway: false\n')
  writeEnvFile(sandbox.hermesHome, 'e2e-mock-key', mock.url)
  const { app, page } = await launchDesktop(buildAppEnv(sandbox))

  const fixture = {
    app,
    page,
    sandbox,
    mock,
    mockUrl: mock.url,
    cleanup: async () => {
      await app.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    }
  }

  const current = path.join(sandbox.hermesHome, 'kanban', 'current')
  const board = path.join(sandbox.hermesHome, 'kanban', 'boards', 'navigation-test')
  fs.mkdirSync(board, { recursive: true })
  fs.writeFileSync(path.join(board, 'board.json'), JSON.stringify({ name: 'Navigation Test', archived: false }))
  fs.writeFileSync(current, 'default\n')
  const config = fs.readFileSync(path.join(sandbox.hermesHome, 'config.yaml'), 'utf8')

  try {
    await waitForAppReady(fixture, 120_000)
    // Synthetic test setup only: opt in to the built-in plugin and seed its
    // remembered selection. All assertions below exercise the built app.
    await page.evaluate(() => {
      localStorage.setItem('hermes.desktop.pluginDecisions.v2', JSON.stringify({ kanban: true }))
      localStorage.setItem('hermes.plugin.kanban.boardSlug', JSON.stringify('navigation-test'))
    })
    await page.reload()
    await waitForAppReady(fixture, 120_000)
    await page.keyboard.press('Control+Alt+k')
    await expect(page.getByRole('button', { name: 'Board: Navigation Test', exact: true })).toBeVisible()
    await expect(page.getByRole('dialog')).toHaveCount(0)
    await page.screenshot({ path: test.info().outputPath('kanban-shortcut-linux.png') })

    await page.keyboard.press('Control+,')
    await expect(page.getByRole('button', { name: 'Board: Navigation Test', exact: true })).toBeHidden()
    await page.keyboard.press('Control+Alt+k')
    await expect(page.getByRole('button', { name: 'Board: Navigation Test', exact: true })).toBeVisible()
    await expect(page.getByRole('dialog')).toHaveCount(0)
    // Capture a replacement binding through native Settings, not storage.
    await page.keyboard.press('Control+/')
    await page.getByRole('textbox', { name: 'Search shortcuts…' }).fill('Show Kanban')
    await page.getByRole('button', { name: 'Rebind', exact: true }).click()
    await page.keyboard.press('Control+Alt+j')
    await expect(page.getByRole('button', { name: 'Reset to default', exact: true })).toBeVisible()
    await page.keyboard.press('Control+Alt+k')
    await expect(page.getByRole('button', { name: 'Board: Navigation Test', exact: true })).toBeHidden()
    await page.keyboard.press('Control+Alt+j')
    await expect(page.getByRole('button', { name: 'Board: Navigation Test', exact: true })).toBeVisible()
    await page.reload()
    await expect(page.getByRole('button', { name: 'Board: Navigation Test', exact: true })).toBeVisible({
      timeout: 60_000
    })
    await page.keyboard.press('Control+,')
    await expect(page.getByRole('button', { name: 'Board: Navigation Test', exact: true })).toBeHidden()
    await page.keyboard.press('Control+Alt+j')
    await expect(page.getByRole('button', { name: 'Board: Navigation Test', exact: true })).toBeVisible()
    await expect(page.getByRole('dialog')).toHaveCount(0)
    await page.screenshot({ path: test.info().outputPath('kanban-rebound-linux.png') })
    expect(fs.readFileSync(current, 'utf8')).toBe('default\n')
    expect(fs.readFileSync(path.join(sandbox.hermesHome, 'config.yaml'), 'utf8')).toBe(config)
  } finally {
    await fixture.cleanup()
  }
})
