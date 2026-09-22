/* global document */
import { chromium } from '@playwright/test'
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const url = process.argv[2] ?? 'http://127.0.0.1:5178/lunar-city-review.html'
const output = resolve(process.argv[3] ?? 'docs/lunar-city/game/evidence/interiors')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true })
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } })
  const errors = [], externalRequests = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('request', request => { if (new URL(request.url()).origin !== new URL(url).origin && /^https?:/.test(request.url())) externalRequests.push(request.url()) })
  await page.goto(url)
  await page.waitForFunction(() => document.documentElement.dataset.worldReady === 'true' || document.querySelector('#retry')?.hidden === false, null, { timeout: 90000 })
  assert.equal(await page.getAttribute('html', 'data-world-ready'), 'true', await page.locator('#loading').textContent())
  assert.equal(await page.locator('#interior').isDisabled(), true)
  const { plans, worldManifestSha256 } = JSON.parse(await readFile(new URL('../../public/lunar-city/interior-plans-v1/plans.json', import.meta.url), 'utf8'))
  const manifestResponse = await page.request.get(new URL('/lunar-city/v2-review/world-manifest.v2.json', url).href)
  assert.equal(manifestResponse.ok(), true)
  const manifestBytes = await manifestResponse.body()
  const servedManifestSha256 = createHash('sha256').update(manifestBytes).digest('hex')
  const reviewLeaderAssets = JSON.parse(manifestBytes.toString()).reviewLeaderAssets
  const results = []
  for (const plan of plans) {
    await page.selectOption('#leaders', `lunar-city:model:${plan.id}`)
    assert.equal(await page.locator('#interior').isEnabled(), true, plan.id)
    await page.click('#interior')
    assert.equal(await page.getAttribute('#interior', 'aria-pressed'), 'true')
    assert.match(await page.locator('#interior-note').textContent(), /prototype interior/)
    await page.waitForTimeout(350)
    if (['library', 'engineering-workshop'].includes(plan.id)) await page.screenshot({ path: resolve(output, `${plan.id}.png`) })
    await page.click('#interior')
    assert.equal(await page.getAttribute('#interior', 'aria-pressed'), 'false')
    results.push({ id: plan.id, openedAndClosed: true })
  }
  await page.click('#interior')
  await page.click('#overview')
  assert.equal(await page.locator('#interior').isDisabled(), true)
  assert.equal(await page.getAttribute('#interior', 'aria-pressed'), 'false')
  const leaders = await page.locator('#leaders option').evaluateAll(nodes => nodes.map(node => node.value).filter(value => value.startsWith('lunar-city:leader:')))
  const localLeaderControls = []
  for (const key of leaders) {
    await page.selectOption('#leaders', key)
    const enabled = await page.locator('#leader-life').isEnabled()
    if (enabled) {
      for (const mode of ['home', 'work', 'idle', 'automatic']) await page.selectOption('#leader-life', mode)
      await page.selectOption('#leaders', key)
      assert.equal(await page.locator('#leader-life').inputValue(), 'automatic')
    } else {
      assert.match(await page.locator('#leader-life-note').textContent(), /unavailable/i)
    }
    localLeaderControls.push({ key, enabled, automaticResetVerified: enabled, reason: await page.locator('#leader-life-note').count() ? await page.locator('#leader-life-note').textContent() : '' })
  }
  assert.ok(localLeaderControls.some(control => control.enabled), 'The review fixture must retain at least one usable local locomotion model')
  assert.deepEqual(errors, [])
  assert.deepEqual(externalRequests, [])
  await writeFile(resolve(output, 'receipt.json'), JSON.stringify({ capturedAt: new Date().toISOString(), url, worldManifestSha256, servedManifestSha256, reviewLeaderAssets, results, localLeaderControls, errors, externalRequests, scope: 'Actual browser cutaway UI open/close and selection reset for each authored plan; saved library and waterworks views. Not proof of shell fit or navigation.' }, null, 2))
  console.log({ pass: true, interiors: results.length, output })
} finally { await browser.close() }
