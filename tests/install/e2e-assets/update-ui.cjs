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

async function assertStagedBranch(page, expectedSha, log) {
  let status
  for (let attempt = 0; attempt < 3; attempt++) {
    status = await page.evaluate(() => window.hermesDesktop.updates.check({ force: true }))
    log(`[source-branch-check] ${JSON.stringify(status)}`)
    // The app's mount-time poller can fetch the same origin/main concurrently;
    // Git rejects the losing ref update even though the winning fetch succeeded.
    // Retry only that transient lock race, never a missing channel or other error.
    if (status.error !== 'fetch-failed' || !/cannot lock ref 'refs\/remotes\/origin\/main'/.test(status.message || '')) break
    await page.waitForTimeout(1_000)
  }
  // Historical Desktop status has no updateAvailable field: its About/overlay
  // offers the button when behind > 0. A newer checker states updateAvailable
  // and leaves behind null when it cannot count (GitHub compare does not know a
  // staged commit). Never accept an explicit false, a dirty source tree, or
  // merely a matching remote tip.
  const offered = status.updateAvailable === undefined
    ? Number.isInteger(status.behind) && status.behind > 0
    : status.updateAvailable === true
  if (status.supported !== true || status.error || status.dirty === true ||
      status.branch !== 'main' || status.targetSha !== expectedSha ||
      status.currentSha === expectedSha || !offered) {
    throw new Error('Desktop source check did not offer staged Git main; refusing to click an unrelated update')
  }
}

async function waitForUpdate(page, { log, shot }) {
  const update = page.getByRole('button', { name: /update now/i }).first()
  const details = page.getByRole('button', { name: /^see what['’]s new$/i }).first()
  const deadline = Date.now() + 180_000
  while (!await update.isVisible().catch(() => false) && Date.now() < deadline) {
    if (await details.isVisible().catch(() => false)) {
      await details.click({ timeout: 5_000 })
      log("opened See what's new")
      continue
    }
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
    throw new Error('"Update now" never appeared after checking for updates and opening available update details')
  }
  await shot(page, '04-update-available')
  return update
}

async function readManualUpdateCommand(page) {
  const title = page.getByText(/^Update from your terminal$/i).first()
  try {
    await title.waitFor({ state: 'visible', timeout: 5_000 })
  } catch {
    return null
  }
  const text = await page.locator('code').filter({ hasText: /hermes update/i }).first().textContent()
  const command = (text || '').trim().replace(/^\$\s*/, '')
  if (!/^hermes update(?:\s|$)/.test(command)) {
    throw new Error('manual update card did not present a hermes update command')
  }
  return command
}

module.exports = { assertStagedBranch, pickAppWindow, openAbout, readManualUpdateCommand, waitForUpdate }
