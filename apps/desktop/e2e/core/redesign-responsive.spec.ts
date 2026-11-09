import { appendFileSync } from 'node:fs'

import { chromium, expect, test } from '@playwright/test'

import { composer, createCoreSandbox, freePort, webappLaunchLink, writeProviderHome } from './harness'
import { startScriptedProvider } from './provider'

test('redesigned web app stays usable on touch screens and restores the classic interface', async () => {
  test.setTimeout(180_000)
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('redesign-responsive')
  writeProviderHome(sandbox.hermesHome, provider.url)
  appendFileSync(sandbox.hermesHome + '/config.yaml', `\nwebapp:\n  enabled: true\n  port: ${await freePort()}\n`)
  const browser = await chromium.launch({ executablePath: process.env.HERMES_E2E_CHROMIUM })

  try {
    const context = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true })
    const page = await context.newPage()
    page.setDefaultTimeout(12_000)
    await page.goto(webappLaunchLink(sandbox))
    await expect(composer(page)).toBeVisible({ timeout: 120_000 })
    const go = async (route: string) => page.evaluate(route => { location.hash = route }, route)

    for (const viewport of [{ width: 320, height: 740 }, { width: 390, height: 844 }, { width: 768, height: 1024 }, { width: 844, height: 390 }]) {
      await page.setViewportSize(viewport)
      await go('/')
      await expect(composer(page)).toBeVisible()
      await expect(page.locator('[data-portal-redesign="true"]')).toBeVisible()
      const nav = page.getByRole('navigation', { name: 'Hermes', exact: true })
      await nav.getByRole('button', { name: 'Capabilities', exact: true }).tap()
      await expect(page).toHaveURL(/capabilities/)
      await nav.getByRole('button', { name: 'Chats', exact: true }).tap()
      await expect(composer(page)).toBeVisible()
      const sidebar = page.locator('.portal-sidebar')

      if (!await sidebar.isVisible()) {
        await page.getByRole('button', { name: 'Show sidebar', exact: true }).tap()
      }

      await expect(sidebar).toBeVisible()
      const bounds = await sidebar.boundingBox()
      expect(bounds!.x).toBeGreaterThanOrEqual(0)
      expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(viewport.width)
      await page.getByLabel('Window controls', { exact: true }).getByRole('button', { name: 'Hide sidebar', exact: true }).tap()
      await expect(sidebar).not.toBeVisible()
      await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
      await go('/settings?tab=config:appearance&page=window-layout')
      const toggle = page.getByRole('switch', { name: 'Redesigned interface', exact: true })
      await expect(toggle).toBeVisible()
      await expect(toggle).toBeChecked()
      await toggle.tap()
      await expect(toggle).not.toBeChecked()
      await toggle.tap()
      await expect(toggle).toBeChecked()
      await go('/')
      await expect(composer(page)).toBeVisible()
      const consent = page.locator('[data-slot="composer-status-stack"]')
      const consentBounds = await consent.boundingBox()

      for (const name of ['Share with Nous', 'No thanks', 'Details']) {
        const choice = consent.getByRole('button', { name, exact: true })
        await expect(choice).toBeVisible()
        const bounds = await choice.boundingBox()
        expect(bounds!.x).toBeGreaterThanOrEqual(consentBounds!.x)
        expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(consentBounds!.x + consentBounds!.width)
      }

      await page.screenshot({ path: test.info().outputPath(`redesign-${viewport.width}.png`) })
    }

    await page.setViewportSize({ width: 1280, height: 860 })
    await go('/settings?tab=config:appearance&page=window-layout')
    await page.getByRole('button', { name: 'Simple', exact: true }).tap()
    await go('/')
    await expect(page.getByRole('button', { name: 'Terminal', exact: true })).not.toBeVisible()
    await go('/settings?tab=config:appearance&page=window-layout')
    await page.locator('[id="setting-field-appearance.interface-mode"]').getByRole('button', { name: 'Advanced', exact: true }).tap()
    await go('/settings?tab=config:safety&page=approvals')
    await expect(page.getByRole('radio', { name: /^Smart/ })).toBeVisible()
    await page.locator('[data-slot="portal-safety-settings"] summary').tap()
    await expect(page.locator('[data-slot="portal-safety-settings"] details')).toHaveAttribute('open', '')
    await page.screenshot({ path: test.info().outputPath('redesign-safety.png') })
    await go('/')

    if (!await page.locator('.portal-sidebar').isVisible()) {
      await page.getByRole('button', { name: 'Show sidebar', exact: true }).tap()
    }

    await page.getByRole('button', { name: 'New group chat', exact: true }).tap()
    await expect(page.getByRole('dialog')).toBeVisible()
    await page.getByRole('dialog').getByRole('button', { name: 'Cancel', exact: true }).tap()
    await expect(page.getByRole('dialog')).not.toBeVisible()
    await page.getByRole('button', { name: 'Terminal', exact: true }).tap()
    const terminal = page.locator('[data-persistent-terminal][aria-hidden="false"] .xterm-helper-textarea').first()
    await expect(terminal).toBeVisible()
    const tools = page.getByRole('navigation', { name: 'Right sidebar panels' })
    const terminalPanel = page.locator('[data-persistent-terminal]')
    await expect.poll(async () => (await terminalPanel.boundingBox())?.height ?? 0).toBeGreaterThan(600)
    await page.screenshot({ path: test.info().outputPath('redesign-terminal.png') })

    for (const name of ['Files', 'Review', 'Preview']) {
      await tools.getByRole('button', { name, exact: true }).tap()
      await expect(tools.getByRole('button', { name, exact: true })).toHaveAttribute('aria-pressed', 'true', { timeout: 10_000 })
      await expect(tools.getByRole('button', { name: 'Terminal', exact: true })).toHaveAttribute('aria-pressed', 'false', { timeout: 10_000 })
      await expect(terminalPanel).toHaveAttribute('aria-hidden', 'true')
      await tools.getByRole('button', { name: 'Terminal', exact: true }).tap()
      await expect(terminal).toBeVisible()
      await expect(tools.getByRole('button', { name, exact: true })).toHaveAttribute('aria-pressed', 'false', { timeout: 10_000 })
      await expect.poll(async () => (await terminalPanel.boundingBox())?.height ?? 0).toBeGreaterThan(600)
      await expect.poll(async () => (await terminalPanel.boundingBox())?.width ?? 0).toBeGreaterThan(300)
    }

    await page.getByRole('button', { name: 'Hide terminal', exact: true }).tap()
    await expect(page.locator('[data-persistent-terminal]')).toHaveAttribute('aria-hidden', 'true', { timeout: 5000 })
    await expect(page.locator('[data-persistent-terminal]')).toHaveAttribute('inert', '')
    await expect(page.locator('[data-persistent-terminal]')).toHaveCSS('opacity', '0')
    await expect(terminal).not.toBeVisible()
    await expect(page.locator('[data-persistent-terminal] .xterm-helper-textarea')).toHaveCount(1)
    await page.getByRole('button', { name: 'Terminal', exact: true }).tap()
    await expect(terminal).toBeVisible()
    await page.getByRole('button', { name: 'Hide terminal', exact: true }).tap()
    await page.screenshot({ path: test.info().outputPath('redesign-desktop.png') })
    await go('/settings?tab=config:appearance&page=window-layout')
    await page.getByRole('switch', { name: 'Redesigned interface', exact: true }).tap()
    await go('/')
    await expect(composer(page)).toBeVisible()
    await page.screenshot({ path: test.info().outputPath('classic-desktop.png') })
  } finally {
    await browser.close()
    sandbox.cleanup()
    await provider.close()
  }
})
