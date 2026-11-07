import { execFileSync } from 'node:child_process'
import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

import { expect, test } from '@playwright/test'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import { MOCK_REPLY, startMockServer } from '../../../tests-js/scripts/mock-server'

import { buildAppEnv, createSandbox, launchDesktop, waitForAppReady } from './fixtures'

test('project targeting on the built desktop', async () => {
  test.setTimeout(150_000)
  const sandbox = createSandbox('project-targeting')
  const mock = await startMockServer()
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)
  const roaming = join(sandbox.root, 'AppData', 'Roaming')
  const local = join(sandbox.root, 'AppData', 'Local')
  mkdirSync(roaming, { recursive: true })
  mkdirSync(local, { recursive: true })

  const { app, page } = await launchDesktop(
    buildAppEnv(sandbox, { USERPROFILE: sandbox.root, APPDATA: roaming, LOCALAPPDATA: local })
  )

  try {
    await waitForAppReady({ app, page } as any, 100_000)
    await app.evaluate(({ ipcMain }) => {
      const handlers = (ipcMain as any)._invokeHandlers

      const start = handlers.get('hermes:terminal:start')

      ;(globalThis as any).__startedTerminals = []
      ipcMain.removeHandler('hermes:terminal:start')
      ipcMain.handle('hermes:terminal:start', async (event, payload) => {
        const result = await start(event, payload)

        ;(globalThis as any).__startedTerminals.push(result)

        return result
      })
    })
    const sessions: { name: string; repo: string; hash: string }[] = []

    for (const name of ['project-alpha', 'project-beta']) {
      const repo = join(sandbox.root, name)
      mkdirSync(repo)
      execFileSync('git', ['init', '-b', 'main', repo])
      writeFileSync(join(repo, `${name}.txt`), `${name}\n`)
      execFileSync('git', ['-C', repo, 'add', '.'])
      execFileSync('git', [
        '-C',
        repo,
        '-c',
        'user.name=Test',
        '-c',
        'user.email=test@example.invalid',
        '-c',
        'commit.gpgsign=false',
        'commit',
        '-m',
        'Test fixture'
      ])
      writeFileSync(join(repo, `${name}.txt`), `${name}\nchanged\n`)
      await app.evaluate(({ dialog }, repo) => {
        dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [repo] })
      }, repo)
      await page.getByRole('button', { name: 'New project', exact: true }).click()
      const dialog = page.getByRole('dialog')
      await dialog.getByPlaceholder('e.g. Skunkworks').fill(name)
      await dialog.getByRole('button', { name: 'Add folder', exact: true }).click()
      await dialog.getByRole('button', { name: 'Create', exact: true }).click()
      await expect(dialog).toBeHidden()

      if (sessions.length) {
        await page
          .getByRole('button', { name: /New session/ })
          .first()
          .click()
        await expect.poll(() => page.evaluate(() => location.hash)).toBe('#/')
      }

      const composer = page.locator('[contenteditable="true"]').filter({ visible: true }).first()
      await composer.fill(`Please acknowledge ${name}`)
      await composer.press('Enter')
      await expect(page.getByText(MOCK_REPLY, { exact: true }).first()).toBeVisible({ timeout: 60_000 })
      await expect.poll(() => page.evaluate(() => location.hash)).not.toBe('#/')
      sessions.push({ name, repo, hash: await page.evaluate(() => location.hash) })
      await page.getByRole('button', { name: 'All projects', exact: true }).click()
    }

    for (const { name, repo, hash } of sessions) {
      await page.evaluate(hash => {
        location.hash = hash
      }, hash)
      await expect(page.getByText(`Please acknowledge ${name}`, { exact: true }).first()).toBeVisible()
      await page.keyboard.press('Control+g')
      await expect(page.getByTitle(repo, { exact: true }).first()).toBeVisible()
      await expect(page.getByText(`${name}.txt`, { exact: true }).first()).toBeVisible()
      await page.keyboard.press('Control+Shift+Backquote')
      await expect.poll(async () => app.evaluate(() => (globalThis as any).__startedTerminals.at(-1)?.cwd)).toBe(repo)
      await page.keyboard.press('Control+g')
    }

    writeFileSync(test.info().outputPath('ui.txt'), await page.locator('body').innerText())
    writeFileSync(test.info().outputPath('ui-dom.html'), await page.content())
    await page.screenshot({ path: test.info().outputPath('desktop.png') })
  } finally {
    await app.close().catch(() => undefined)
    await mock.close()
    sandbox.cleanup()
  }
})
