import { resolveElectronBinary } from '../electron-binary'
import { _electron, expect, test } from '@playwright/test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

test('native browser selection → persisted board → changed source → chat draft', async () => {
  const userData = await mkdtemp(join(tmpdir(), 'hermes-lens-'))
  const app = await _electron.launch({
    executablePath: resolveElectronBinary([process.cwd(), resolve('../..')]),
    args: [resolve('e2e/lens/main.cjs'), '--user-data-dir=' + userData]
  })
  try {
    const page = await app.firstWindow()
    page.on('pageerror', error => console.error(error.message))
    await expect(page.getByRole('button', { name: 'Hermes Lens', exact: true })).toBeVisible({ timeout: 60000 })
    await expect
      .poll(() =>
        page.evaluate(() => {
          const guest = document.querySelector('webview') as HTMLElement & {
            executeJavaScript: (code: string) => Promise<unknown>
          }
          return guest.executeJavaScript('document.readyState')
        })
      )
      .toBe('complete')
    await page.evaluate(() => {
      const guest = document.querySelector('webview') as HTMLElement & {
        executeJavaScript: (code: string) => Promise<unknown>
      }
      return guest.executeJavaScript(
        "var r=document.createRange();r.selectNodeContents(document.querySelector('#offer'));getSelection().removeAllRanges();getSelection().addRange(r)"
      )
    })
    await page.getByRole('button', { name: 'Hermes Lens', exact: true }).click()
    await page.getByRole('button', { name: 'Pin selected block' }).click()
    await expect(page.getByRole('article')).toContainText('Monthly price: $240')
    await expect(page.getByRole('article')).not.toContainText('Another offer')
    await page.getByRole('textbox', { name: 'Your note' }).fill('Prefer waterfront')
    await page.getByRole('heading', { name: 'Hermes Lens' }).click()
    await page.reload()
    await page.getByRole('button', { name: 'Hermes Lens', exact: true }).click()
    await expect(page.getByRole('textbox', { name: 'Your note' })).toHaveValue('Prefer waterfront')
    await page.evaluate(() => {
      const guest = document.querySelector('webview') as HTMLElement & {
        executeJavaScript: (code: string) => Promise<unknown>
      }
      return guest.executeJavaScript("localStorage.setItem('updated','1')")
    })
    await page.getByRole('button', { name: 'Refresh', exact: true }).click()
    await expect(page.getByRole('article')).toContainText('Monthly price: $210')
    await page.getByText('Changed · show previous capture').click()
    await expect(page.getByRole('article')).toContainText('Monthly price: $240')
    await page.getByRole('checkbox').check()
    await page.getByRole('textbox', { name: 'What should Hermes investigate?' }).fill('Compare value and amenities')
    await page.getByRole('button', { name: 'Ask Hermes', exact: true }).click()
    await expect(page.getByRole('textbox', { name: 'Chat draft' })).toHaveValue(/Compare value and amenities/)
    await expect(page.getByRole('textbox', { name: 'Chat draft' })).toHaveValue(/Prefer waterfront/)
    await page.keyboard.press('Escape')
    await expect(page.getByRole('heading', { name: 'Hermes Lens' })).toBeHidden()
    await expect(page.locator('webview')).toBeVisible()
    await page.getByRole('button', { name: 'Hermes Lens', exact: true }).click()
    await page.screenshot({ path: 'test-results/hermes-lens.png' })
  } finally {
    await app.close()
    await rm(userData, { recursive: true, force: true })
  }
})
