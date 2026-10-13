import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import yaml from 'js-yaml'

import { setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

// eslint-disable-next-line no-empty-pattern
test('subagent model and effort persist independently of the parent and can inherit again', async ({}, info) => {
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
    const effort = page.getByTestId('workhorse-reasoning-pill')
    await expect(model).toContainText('Subagents')
    await model.click()
    const dialog = page.getByRole('dialog')
    await expect(dialog).toContainText('Inherit parent model')
    await dialog.getByText('mock-model', { exact: true }).last().click()
    await expect.poll(() => readConfig().delegation?.model).toBe('mock-model')
    expect(readConfig().model).toEqual(parent)

    await expect(dialog).toBeHidden()
    await effort.click()
    await page.getByRole('menuitemradio', { name: 'High', exact: true }).click()
    await expect.poll(() => readConfig().delegation?.reasoning_effort).toBe('high')
    expect(readConfig().model).toEqual(parent)
    await page.keyboard.press('Escape')
    await page.screenshot({ path: info.outputPath('subagent-controls.png') })

    await model.click()
    await page.getByRole('button', { name: 'Inherit parent model and route', exact: true }).click()
    await expect.poll(() => readConfig().delegation?.model).toBe('')
    expect(readConfig().delegation?.reasoning_effort).toBe('high')
    await effort.click()
    await page.getByRole('menuitem', { name: 'Inherit parent effort', exact: true }).click()
    await expect.poll(() => readConfig().delegation?.reasoning_effort).toBe('')
    expect(readConfig().model).toEqual(parent)
  } finally {
    await fixture.cleanup()
  }
})
