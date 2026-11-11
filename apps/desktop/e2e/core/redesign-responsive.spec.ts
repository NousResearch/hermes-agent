import { appendFileSync } from 'node:fs'

import { chromium, expect, type Page, test } from '@playwright/test'

import { composer, createCoreSandbox, freePort, send, webappLaunchLink, writeProviderHome } from './harness'
import { startScriptedProvider } from './provider'

test('redesigned web app stays usable on touch screens and restores the classic interface', async () => {
  test.setTimeout(180_000)
  const provider = await startScriptedProvider()
  provider.script('U1-preview', [{ text: ['The history preview is ready.'] }])
  const sandbox = createCoreSandbox('redesign-responsive')
  writeProviderHome(sandbox.hermesHome, provider.url)
  appendFileSync(sandbox.hermesHome + '/config.yaml', `\nwebapp:\n  enabled: true\n  port: ${await freePort()}\n`)
  const browser = await chromium.launch({ executablePath: process.env.HERMES_E2E_CHROMIUM })

  try {
    const context = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true })
    // An unrelated release notification must not cover the touch-sized rail.
    await context.addInitScript(() => {
      localStorage.setItem('hermes:update-toast-snooze-until', String(Date.now() + 86_400_000))
    })
    const page = await context.newPage()
    page.setDefaultTimeout(12_000)
    await page.goto(webappLaunchLink(sandbox))
    await expect(composer(page)).toBeVisible({ timeout: 120_000 })

    const go = async (route: string) =>
      page.evaluate(route => {
        location.hash = route
      }, route)

    for (const viewport of [
      { width: 320, height: 740 },
      { width: 390, height: 844 },
      { width: 768, height: 1024 },
      { width: 844, height: 390 }
    ]) {
      await page.setViewportSize(viewport)
      await go('/')
      await expect(composer(page)).toBeVisible()
      await expect(page.locator('html')).toHaveAttribute('data-shell', 'portal')
      const nav = page.getByRole('navigation', { name: 'Hermes', exact: true })
      await nav.getByRole('button', { name: 'Capabilities', exact: true }).tap()
      await expect(page).toHaveURL(/capabilities/)
      await expect(page.getByRole('heading', { name: 'Capabilities', exact: true }).first()).toBeVisible()
      await nav.getByRole('button', { name: 'Chats', exact: true }).tap()
      await expect(composer(page)).toBeVisible()
      const sidebar = page.locator('.portal-sidebar')

      if (!(await sidebar.isVisible())) {
        await page.getByRole('button', { name: 'Show sidebar', exact: true }).tap()
      }

      await expect(sidebar).toBeVisible()
      const bounds = await sidebar.boundingBox()
      expect(bounds!.x).toBeGreaterThanOrEqual(0)
      expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(viewport.width)
      await page
        .getByLabel('Window controls', { exact: true })
        .getByRole('button', { name: 'Hide sidebar', exact: true })
        .tap()
      await expect(sidebar).not.toBeVisible()
      await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
      await go('/settings?tab=config:appearance&page=window-layout')
      const toggle = page.getByRole('switch', { name: 'Redesigned interface', exact: true })
      await expect(toggle).toBeVisible()
      await expect(toggle).toBeChecked()
      await toggle.tap()
      await expect(toggle).not.toBeChecked()
      await expect(page.locator('html')).not.toHaveAttribute('data-shell', /.*/)
      await toggle.tap()
      await expect(toggle).toBeChecked()
      await expect(page.locator('html')).toHaveAttribute('data-shell', 'portal')
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
    await exerciseHistoryPreview(page)
    await exerciseManagementEntries(page, true)

    if (!(await page.locator('.portal-sidebar').isVisible())) {
      await page.getByRole('button', { name: 'Show sidebar', exact: true }).tap()
    }

    const sidebarWidth = (await page.locator('.portal-sidebar').boundingBox())!.width
    await go('/settings?tab=config:appearance&page=window-layout')
    await page.getByRole('button', { name: 'Simple', exact: true }).tap()
    await go('/')
    await expect(page.getByRole('button', { name: 'Terminal', exact: true })).not.toBeVisible()
    await go('/settings?tab=config:appearance&page=window-layout')
    await page
      .locator('[id="setting-field-appearance.interface-mode"]')
      .getByRole('button', { name: 'Advanced', exact: true })
      .tap()
    await go('/settings?tab=config:safety&page=approvals')
    await expect(page.getByRole('radio', { name: /^Smart/ })).toBeVisible()
    const settings = page.locator('[data-page-surface]')
    const rail = page.locator('[data-slot="global-nav"]')
    const settingsNav = settings.locator('[data-tour="overlay-nav"]')
    // Settings is a sibling of the shell: it must still clear the rail and
    // share the redesigned sidebar's width rather than inherit classic sizes.
    await expect
      .poll(
        async () => {
          const railBounds = (await rail.boundingBox())!

          return (await settings.boundingBox())!.x - (railBounds.x + railBounds.width)
        },
        { timeout: 5000 }
      )
      .toBeCloseTo(0)
    await expect.poll(async () => (await settingsNav.boundingBox())!.width, { timeout: 5000 }).toBe(sidebarWidth)
    await page.locator('[data-slot="portal-safety-settings"] summary').tap()
    await expect(page.locator('[data-slot="portal-safety-settings"] details')).toHaveAttribute('open', '')
    await page.screenshot({ path: test.info().outputPath('redesign-safety.png') })
    await page.emulateMedia({ colorScheme: 'dark' })
    await expect(page.locator('html')).toHaveClass(/dark/)
    await page.screenshot({ path: test.info().outputPath('redesign-safety-dark.png') })
    await page.emulateMedia({ colorScheme: 'light' })
    await expect(page.locator('html')).not.toHaveClass(/dark/)
    await go('/settings?tab=config:appearance')
    await page.getByRole('button', { name: 'Switch language', exact: true }).tap()
    await page.getByRole('combobox').fill('العربية')
    await page.getByRole('combobox').press('Enter')
    await expect(page.locator('html')).toHaveAttribute('dir', 'rtl')
    await expect
      .poll(async () => {
        const railBounds = (await rail.boundingBox())!
        const settingsBounds = (await settings.boundingBox())!

        return railBounds.x - (settingsBounds.x + settingsBounds.width)
      })
      .toBeCloseTo(0)
    await rail.getByRole('button', { name: 'المحادثات', exact: true }).tap()
    await expect(composer(page)).toBeVisible()
    await go('/settings?tab=config:appearance')
    await page.getByRole('button', { name: 'تغيير اللغة', exact: true }).tap()
    await page.getByRole('combobox').fill('English')
    await page.getByRole('combobox').press('Enter')
    await expect(page.locator('html')).toHaveAttribute('dir', 'ltr')
    await go('/')

    if (!(await page.locator('.portal-sidebar').isVisible())) {
      await page.getByRole('button', { name: 'Show sidebar', exact: true }).tap()
    }

    await page.getByRole('button', { name: 'New group chat', exact: true }).tap()
    await expect(page.getByRole('dialog')).toBeVisible()
    // Body-portalled dialogs sit outside the shell and must still wear its rounder geometry.
    await expect
      .poll(() => page.getByRole('dialog').evaluate(dialog => getComputedStyle(dialog).getPropertyValue('--radius-scalar').trim()))
      .toBe('0.6')
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
      await expect(tools.getByRole('button', { name, exact: true })).toHaveAttribute('aria-pressed', 'true', {
        timeout: 10_000
      })
      await expect(tools.getByRole('button', { name: 'Terminal', exact: true })).toHaveAttribute(
        'aria-pressed',
        'false',
        { timeout: 10_000 }
      )
      await expect(terminalPanel).toHaveAttribute('aria-hidden', 'true')
      await tools.getByRole('button', { name: 'Terminal', exact: true }).tap()
      await expect(terminal).toBeVisible()
      await expect(tools.getByRole('button', { name, exact: true })).toHaveAttribute('aria-pressed', 'false', {
        timeout: 10_000
      })
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
    await exerciseManagementEntries(page, false)
    await page.screenshot({ path: test.info().outputPath('classic-desktop.png') })
  } finally {
    await browser.close()
    sandbox.cleanup()
    await provider.close()
  }
})

async function exerciseHistoryPreview(page: Page) {
  await page.getByRole('button', { name: 'No thanks', exact: true }).tap()
  await send(page, 'U1-preview Keep this conversation for the history preview.')
  await expect(page.getByText('The history preview is ready.', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Agent settings', exact: true }).tap()
  await page.getByRole('button', { name: 'History', exact: true }).tap()

  for (const viewport of [
    { width: 1280, height: 860 },
    { width: 320, height: 740 }
  ]) {
    await page.setViewportSize(viewport)
    await page
      .getByRole('button', { name: /^Session actions/ })
      .first()
      .tap()
    await page.getByRole('menuitem', { name: 'Preview', exact: true }).tap()
    const preview = page.getByRole('dialog')
    await expect(preview.getByText('The history preview is ready.', { exact: true })).toBeVisible()
    await expect
      .poll(async () => {
        const bounds = await preview.boundingBox()

        return (
          bounds !== null &&
          bounds.x >= 0 &&
          bounds.y >= 0 &&
          bounds.x + bounds.width <= viewport.width &&
          bounds.y + bounds.height <= viewport.height
        )
      })
      .toBe(true)
    await expect(preview.getByRole('button', { name: 'Open chat', exact: true })).toBeInViewport({ ratio: 1 })
    await preview.getByRole('button', { name: 'Close', exact: true }).tap()
    await expect(preview).not.toBeVisible()
  }

  await page.setViewportSize({ width: 1280, height: 860 })
}

// Exercise the same Agent/Schedule concepts from navigation through both editors.
async function exerciseManagementEntries(page: Page, redesigned: boolean) {
  const rail = page.locator('[data-slot="global-nav"]')

  if (redesigned) {
    await rail.getByRole('button', { name: 'Schedules', exact: true }).click()
  } else {
    await page.evaluate(() => {
      location.hash = '/cron'
    })
  }

  await expect(page).toHaveURL(/#\/cron/)
  await page.getByRole('heading', { name: 'Schedules', exact: true }).first().waitFor({ state: 'visible' })
  // The empty-state button is replaced when the async blueprint catalog arrives.
  // Use the schedule-list action so this assertion tracks the same opener.
  const newSchedule = page.locator('button[aria-label="New schedule"]')
  await newSchedule.click()
  const schedule = page.getByRole('dialog', { name: 'New schedule', exact: true })
  await schedule.getByRole('textbox', { name: /^Name\b/ }).fill('Weekly briefing')
  await schedule.getByRole('textbox', { name: 'Prompt', exact: true }).fill('Summarize the week.')
  await schedule.getByRole('button', { name: 'Cancel', exact: true }).click()
  await expect(schedule).not.toBeVisible()

  if (redesigned) {
    await expect(newSchedule).toBeFocused()
  }

  if (redesigned) {
    await rail.getByRole('button', { name: 'Agents', exact: true }).click()
  } else {
    await page.evaluate(() => {
      location.hash = '/profiles?manage=true'
    })
  }

  await page.getByRole('heading', { name: 'Agents', exact: true }).first().waitFor({ state: 'visible' })

  const newAgent = page.getByRole('button', { name: 'New agent', exact: true }).first()
  await newAgent.click()
  const agent = page.getByRole('dialog', { name: 'New agent', exact: true })
  await agent.getByRole('textbox', { name: /^Name\b/ }).fill('terminology-smoke')
  await agent.getByRole('button', { name: 'Cancel', exact: true }).click()
  await expect(agent).not.toBeVisible()

  if (redesigned) {
    await expect(newAgent).toBeFocused()
  }

  await page.evaluate(() => {
    location.hash = '/'
  })
  await expect(composer(page)).toBeVisible()
}
