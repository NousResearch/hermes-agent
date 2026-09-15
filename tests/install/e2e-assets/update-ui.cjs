// Shared clicks only. Each route owns its receipt/process/relaunch observer.
async function pickAppWindow(app, log) {
  await app.firstWindow({ timeout: 120_000 })
  const deadline = Date.now() + 120_000
  for (;;) {
    for (const page of app.windows()) {
      if (await page.evaluate(() => document.querySelector('button') !== null).catch(() => false)) {
        await page.waitForLoadState('domcontentloaded')
        log(`window picked: ${page.url()}`)
        return page
      }
    }
    if (Date.now() > deadline) {
      for (const page of app.windows()) log(`window seen: ${page.url()}`)
      throw new Error('no window with app UI (a <button>) appeared within 120s')
    }
    await new Promise(resolve => setTimeout(resolve, 1_000))
  }
}

async function openAbout(page, { prepare, log, shot, confirmSettings = false, hitDump }) {
  const later = [
    page.getByRole('button', { name: /choose a provider later|skip/i }),
    page.getByText(/choose a provider later/i),
  ]
  const settings = [
    page.getByRole('button', { name: /open settings|settings/i }),
    page.getByLabel('Open settings'), page.locator('[aria-label="Open settings"]'),
    page.locator('[title="Open settings"]'),
  ]
  const deadline = Date.now() + 180_000
  let opened = false
  for (let attempt = 1; !opened; attempt++) {
    await prepare()
    for (const locator of later) {
      try {
        await locator.first().click({ timeout: 1_500 })
        log('dismissed onboarding overlay')
        await locator.first().waitFor({ state: 'hidden', timeout: 15_000 }).catch(() => {})
        await shot(page, '01b-onboarding-dismissed')
        break
      } catch { /* the overlay may still be mounting */ }
    }
    for (const locator of settings) {
      try {
        await locator.first().click({ timeout: 2_500 })
        // Source shells can remount during hydration and lose a landed click.
        if (confirmSettings) await page.waitForURL(/[#/]settings(?:[/?]|$)/, { timeout: 4_000 })
        opened = true
        break
      } catch (error) { log(`[overlay] settings click failed: ${String(error).split('\n')[0]}`) }
    }
    if (!opened && hitDump && (attempt === 1 || attempt % 5 === 0)) log(`[overlay] hit-test: ${await hitDump()}`)
    if (!opened && Date.now() > deadline) {
      await shot(page, 'ERROR-no-settings-button')
      throw new Error('Settings not clickable within 180s')
    }
  }
  await page.waitForTimeout(1_500)
  await shot(page, '02-settings-open')
  const about = [page.getByRole('tab', { name: /about/i }), page.getByRole('button', { name: /about/i }), page.getByText('About', { exact: true })]
  const aboutDeadline = Date.now() + 30_000
  for (;;) {
    for (const locator of about) {
      try {
        if (!await locator.first().isVisible()) continue
        await locator.first().click({ timeout: 2_500 })
        await page.waitForTimeout(1_500)
        await shot(page, '03-about-panel')
        return
      } catch { /* old releases use different controls */ }
    }
    if (Date.now() > aboutDeadline) {
      await shot(page, 'ERROR-no-about-tab')
      throw new Error('could not find the About section in Settings')
    }
    await page.waitForTimeout(500)
  }
}

async function waitForUpdate(page, { log, shot }) {
  const update = page.getByRole('button', { name: /update now/i }).first()
  const deadline = Date.now() + 180_000
  while (!await update.isVisible().catch(() => false) && Date.now() < deadline) {
    await page.getByRole('button', { name: /check now/i }).first().click({ timeout: 5_000 })
      .then(() => log('nudged Check now')).catch(() => {})
    await page.waitForTimeout(15_000)
  }
  if (!await update.isVisible().catch(() => false)) {
    const status = await page.evaluate(() =>
      window.hermesDesktop?.updates?.check?.() ?? Promise.resolve('no updates.check bridge')
    ).catch(error => `updates.check failed: ${error.message}`)
    log(`[update-status] ${JSON.stringify(status)}`)
    await shot(page, 'ERROR-no-update-now')
    throw new Error('"Update now" never appeared — update check did not report an update')
  }
  await shot(page, '04-update-available')
  return update
}

module.exports = { pickAppWindow, openAbout, waitForUpdate }
