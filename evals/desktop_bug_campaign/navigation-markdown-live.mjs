import { chromium } from '../../node_modules/playwright/index.mjs'
import fs from 'node:fs'
import { execFileSync } from 'node:child_process'
const artifact = process.env.NAVIGATION_ARTIFACT_DIR ?? '/home/teknium/.hermes/cache/desktop-bugs-74848ed3/navigation-markdown'
const tag = process.argv[2] ?? 'after'
const browser = await chromium.launch({headless: true, args: ['--no-sandbox']})
const context = await browser.newContext({permissions: ['clipboard-read', 'clipboard-write']})
const page = await context.newPage()
const errors = []
page.on('pageerror', error => errors.push(String(error)))
await page.goto('http://127.0.0.1:18160/navigation-markdown-probe.html', {waitUntil: 'networkidle', timeout:120000})
await page.waitForSelector('#assistant-code .shiki')
const cases = await page.evaluate(() => window.markdownCases)
const results = {}
for (const [id, sample] of Object.entries(cases)) {
  const section = page.locator(`#${id}`)
  const dom = await section.evaluate(el => ({breaks: el.querySelectorAll('br').length, code: el.querySelector('code')?.textContent ?? null, highlighted: !!el.querySelector('.shiki')}))
  let copy = null
  if (await section.locator('[data-slot="code-card"] button').count()) {
    await section.locator('[data-slot="code-card"] button').first().click({force:true})
    copy = await page.evaluate(() => navigator.clipboard.readText())
  }
  const expectedCode = {indented:'value = 1\n', unfinished:'value = 1  \n', code:'value = 1  \nvalue = 2 \n', blanks:'\nvalue = 1  \n\n\n'}[sample.kind]
  const passed = expectedCode === undefined ? dom.breaks === (sample.kind === 'hard' ? 1 : 0) : dom.code === expectedCode && copy === expectedCode
  results[id] = {...sample, ...dom, copy, expectedCode, passed}
}
const result = {sourceSha:execFileSync('git', ['rev-parse','HEAD'], {encoding:'utf8'}).trim(), results, errors, passed:Object.values(results).every(row => row.passed) && errors.length === 0}
fs.mkdirSync(artifact, {recursive:true})
fs.writeFileSync(`${artifact}/markdown-${tag}.json`, JSON.stringify(result,null,2))
await page.screenshot({path:`${artifact}/markdown-${tag}.png`, fullPage:true})
console.log(JSON.stringify(result,null,2))
await browser.close()
process.exitCode = result.passed ? 0 : 1
