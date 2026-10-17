import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import type { BrowserWindow as ElectronWindow } from 'electron'
import yaml from 'js-yaml'

import { setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

// eslint-disable-next-line no-empty-pattern
test('compact subagent menu matches parent rows and persists independent settings', async ({}, info) => {
  test.setTimeout(180_000)
  const fixture = await setupMockBackend()
  const { page } = fixture

  const readConfig = () =>
    yaml.load(readFileSync(join(fixture.sandbox.hermesHome, 'config.yaml'), 'utf8')) as {
      model: Record<string, unknown>
      delegation?: Record<string, unknown>
    }

  try {
    await waitForAppReady(fixture, 120_000)
    const parent = readConfig().model
    const model = page.getByTestId('workhorse-model-pill')
    const menu = page.getByTestId('workhorse-model-menu')
    const effort = page.getByTestId('workhorse-reasoning-menu')
    await expect(model).toHaveAccessibleName('Subagents')
    await expect(model).toHaveText('')
    const buttonBounds = await model.boundingBox()
    expect(buttonBounds!.width).toBeLessThanOrEqual(36)
    expect(buttonBounds!.width).toBe(buttonBounds!.height)

    await page.getByRole('button', { name: /mock-model/ }).click()
    await expect(
      page
        .getByRole('menuitem')
        .filter({ hasText: /mock[- ]model/i })
        .first()
    ).toBeVisible()

    const parentRows = await page
      .getByRole('menuitem')
      .filter({ hasText: /mock[- ]model/i })
      .locator('span.truncate')
      .allTextContents()

    expect(parentRows.length).toBeGreaterThan(0)
    const parentBounds = await page.getByRole('menu').boundingBox()
    await page.keyboard.press('Escape')

    await model.click()
    await expect(menu).toBeVisible()
    await expect(page.getByRole('dialog')).toHaveCount(0)
    expect(
      await menu
        .getByRole('menuitem')
        .filter({ hasText: /mock[- ]model/i })
        .locator('span.truncate')
        .allTextContents()
    ).toEqual(parentRows)
    expect((await menu.boundingBox())!.width).toBe(parentBounds!.width)
    await page.screenshot({ path: info.outputPath('subagent-menu.png') })
    await menu
      .getByRole('menuitem')
      .filter({ hasText: /mock[- ]model/i })
      .first()
      .click()
    await expect.poll(() => readConfig().delegation?.model).toBe('mock-model')
    expect(readConfig().model).toEqual(parent)

    await model.click()
    await effort.hover()
    await page.getByRole('menuitemradio', { name: 'High', exact: true }).click()
    await expect.poll(() => readConfig().delegation?.reasoning_effort).toBe('high')
    expect(readConfig().model).toEqual(parent)
    await page.keyboard.press('Escape')
    await page.keyboard.press('Escape')
    await expect(menu).toBeHidden()
    await page.screenshot({ path: info.outputPath('subagent-toolbar.png') })

    await model.click()
    await page.getByRole('menuitem', { name: 'Inherit parent model and route', exact: true }).click()
    await expect.poll(() => readConfig().delegation?.model).toBe('')
    expect(readConfig().delegation?.reasoning_effort).toBe('high')
    await model.click()
    await effort.hover()
    await page.getByRole('menuitem', { name: 'Inherit parent effort', exact: true }).click()
    await expect.poll(() => readConfig().delegation?.reasoning_effort).toBe('')
    expect(readConfig().model).toEqual(parent)

    await fixture.app.evaluate(({ BrowserWindow }) => {
      BrowserWindow.getAllWindows()
        .find((window: ElectronWindow) => window.getBounds().width > 500)
        ?.setSize(760, 800)
    })
    await expect(model).toBeVisible()
    await expect(model).toHaveText('')
    await model.click()
    await expect(menu).toBeVisible()
    const bounds = await menu.boundingBox()
    const viewport = await page.evaluate(() => ({ width: innerWidth, height: innerHeight }))
    expect(bounds!.x).toBeGreaterThanOrEqual(0)
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(viewport.width)
    expect(bounds!.y).toBeGreaterThanOrEqual(0)
    expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(viewport.height)
    await page.screenshot({ path: info.outputPath('subagent-menu-narrow.png') })
  } finally {
    await fixture.cleanup()
  }
})
