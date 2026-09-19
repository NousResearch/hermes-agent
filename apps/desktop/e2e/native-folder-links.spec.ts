import { execFileSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as path from 'node:path'

import { setupMockBackend, waitForAppReady } from './fixtures'
import { allowErrorBanners, expect, test } from './test'

// Opt-in: this test really opens Explorer. It does not replace shell.openPath
// with a spy, and closes only the window showing its own temporary directory.
// Run on Windows after building:
// HERMES_E2E_NATIVE_FILE_MANAGER=1 npm exec playwright test e2e/native-folder-links.spec.ts
const nativeExplorer = process.platform === 'win32' && process.env.HERMES_E2E_NATIVE_FILE_MANAGER === '1'

function explorerFolders(): string[] {
  const json = execFileSync(
    'powershell.exe',
    [
      '-NoProfile',
      '-NonInteractive',
      '-Command',
      '[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); $shell = New-Object -ComObject Shell.Application; ConvertTo-Json -Compress -InputObject @($shell.Windows() | ForEach-Object { try { $_.Document.Folder.Self.Path } catch {} })'
    ],
    { encoding: 'utf8', windowsHide: true }
  ).trim()

  return JSON.parse(json || '[]') as string[]
}

function closeExplorerFolder(folder: string): void {
  execFileSync(
    'powershell.exe',
    [
      '-NoProfile',
      '-NonInteractive',
      '-Command',
      '$shell = New-Object -ComObject Shell.Application; @($shell.Windows()) | ForEach-Object { try { if ($_.Document.Folder.Self.Path -eq $env:HERMES_E2E_FOLDER) { $_.Quit() } } catch {} }'
    ],
    { env: { ...process.env, HERMES_E2E_FOLDER: folder }, windowsHide: true }
  )
}

test('an explicit transcript folder link opens Explorer without a preview pane', async ({}, testInfo) => {
  test.skip(!nativeExplorer, 'Opt in on Windows with HERMES_E2E_NATIVE_FILE_MANAGER=1; opens real Explorer')
  test.setTimeout(180_000)
  let folder = ''
  const fixture = await setupMockBackend({
    mockServer: {
      replyForPrompt: () => {
        const encoded = encodeURIComponent(folder).replace(
          /[!'()*]/g,
          char => `%${char.charCodeAt(0).toString(16).toUpperCase()}`
        )
        return `[Open test folder](#folder/${encoded})`
      }
    }
  })

  try {
    folder = path.join(fixture.sandbox.root, 'Folder Проверка #100% (ready)')
    fs.mkdirSync(folder)
    await waitForAppReady(fixture, 120_000)
    const { page } = fixture
    const composer = page.locator('[data-slot="composer-rich-input"]').first()
    await composer.fill('Show the native folder link for this test.')
    await composer.press('Enter')

    const link = page.getByRole('button', { name: 'Open test folder', exact: true })
    await expect(link).toBeVisible({ timeout: 60_000 })
    const urlBefore = page.url()
    const windowCountBefore = fixture.app.windows().length
    const previewTabsBefore = await page.locator('[role="tab"]').count()
    expect(explorerFolders()).not.toContain(folder)
    await link.click()

    await expect.poll(() => explorerFolders(), { timeout: 20_000 }).toContain(folder)
    expect(page.url()).toBe(urlBefore)
    expect(fixture.app.windows()).toHaveLength(windowCountBefore)
    expect(await page.locator('[role="tab"]').count()).toBe(previewTabsBefore)
    await expect(page.getByRole('button', { name: 'Open preview', exact: true })).toHaveCount(0)
    await page.screenshot({ path: testInfo.outputPath('native-folder-link.png') })
  } finally {
    if (folder) closeExplorerFolder(folder)
    await fixture.cleanup()
  }
})

test('a native folder error uses the configured language without creating the missing directory', async () => {
  test.skip(!nativeExplorer, 'Opt in on Windows with HERMES_E2E_NATIVE_FILE_MANAGER=1')
  test.setTimeout(180_000)
  allowErrorBanners()
  let missing = ''
  const fixture = await setupMockBackend({
    extraDisplayConfig: '  language: ru',
    mockServer: {
      replyForPrompt: () => `[Открыть папку](#folder/${encodeURIComponent(missing)})`
    }
  })

  try {
    missing = path.join(fixture.sandbox.root, 'missing-directory')
    await waitForAppReady(fixture, 120_000)
    const { page } = fixture
    const composer = page.locator('[data-slot="composer-rich-input"]').first()
    await composer.fill('Покажи ссылку на отсутствующую папку для проверки ошибки.')
    await composer.press('Enter')
    const link = page.getByRole('button', { name: 'Открыть папку', exact: true })
    await expect(link).toBeVisible({ timeout: 60_000 })
    await link.click()
    await expect(page.getByRole('alert')).toContainText('Не удалось открыть папку')
    expect(fs.existsSync(missing)).toBe(false)
  } finally {
    await fixture.cleanup()
  }
})
