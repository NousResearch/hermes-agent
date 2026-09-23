import { chromium } from '@playwright/test'
import { mkdir, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const url = process.argv[2] ?? 'http://127.0.0.1:5178/lunar-city-review.html'
const output = resolve(process.argv[3] ?? '/private/tmp/lunar-city-game-evidence')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true })
const errors = []
const reviewed = []
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
  page.setDefaultTimeout(15000)
  page.on('pageerror', error => errors.push(error.message))
  await page.goto(url)
  await page.waitForFunction(() => document.documentElement.dataset.worldReady === 'true', null, { timeout: 60_000 })
  await page.click('#motion')
  const options = await page
    .locator('#leaders option')
    .evaluateAll(nodes => nodes.map(node => node.value).filter(Boolean))
  for (const key of options) {
    const previousFrame = await page.evaluate(
      async () => (await import('/src/app/lunar-city/review/main.ts')).reviewState().metrics.renderFrames
    )
    await page.selectOption('#leaders', key)
    await page.waitForFunction(
      async frame => (await import('/src/app/lunar-city/review/main.ts')).reviewState().metrics.renderFrames > frame,
      previousFrame
    )
    await page.waitForFunction(async selected => {
      const { reviewState } = await import('/src/app/lunar-city/review/main.ts')
      return reviewState().camera?.focusedEntityKey === selected
    }, key)
    const state = await page.evaluate(async () => (await import('/src/app/lunar-city/review/main.ts')).reviewState())
    if (state.metrics.activeAnimations !== 0)
      throw new Error('Static review geometry incorrectly reports active animations')
    if (state.metrics.cameraRadius > (key.includes(':model:') ? 100 : 9)) throw new Error(`Focus framing failed for ${key}`)
    if (key.includes(':model:')) await page.screenshot({ path: resolve(output, `${key.split(':').at(-1)}-building-focus.png`) })
    reviewed.push({ key, cameraRadius: state.metrics.cameraRadius, activeAnimations: state.metrics.activeAnimations })
    if (key.endsWith(':owl')) {
      await page.selectOption('#leaders', '')
      await page.mouse.click(720, 500)
      await page.waitForFunction(() => document.querySelector('#selection').textContent === 'owl')
      await page.screenshot({ path: resolve(output, 'owl-focus.png') })
    }
  }
  await page.click('#overview')
  await page.selectOption('#quality', 'efficient')
  await page.waitForFunction(
    async () => !(await import('/src/app/lunar-city/review/main.ts')).reviewState().camera?.focusedEntityKey
  )
  await page.screenshot({ path: resolve(output, 'city-overview.png') })
  if (errors.length) throw new Error(errors.join('\n'))
  const receipt = {
    url,
    capturedAt: new Date().toISOString(),
    evidence: 'Headless browser WebGL, not packaged Electron or live gateway acceptance',
    pausedReviewAssets: reviewed,
    pageErrors: errors
  }
  await writeFile(resolve(output, 'smoke.json'), JSON.stringify(receipt, null, 2) + '\n')
  console.log(JSON.stringify(receipt, null, 2))
} finally {
  await browser.close()
}
