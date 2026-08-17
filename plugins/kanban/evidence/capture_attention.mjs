import { chromium } from 'playwright'
import { mkdir, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import path from 'node:path'

const widths = [320, 360, 390, 430]
const output = path.resolve(import.meta.dirname, 'attention-responsive')
await mkdir(output, { recursive: true })

const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true })
const manifest = { generatedBy: 'repo-native Playwright + system Chromium', privacySafeFixture: true, captures: [] }

function markup(surface) {
  const desktop = surface === 'desktop'
  return `<!doctype html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box}body{margin:0;background:${desktop ? '#111318' : '#f2efe7'};color:${desktop ? '#f4f0e6' : '#171717'};font:13px ui-monospace,SFMono-Regular,Menlo,monospace}main{width:100vw;min-height:100vh;padding:12px;overflow:hidden}.shell{border:2px solid ${desktop ? '#f0b84a' : '#171717'};background:${desktop ? '#191c22' : '#fffdf6'};box-shadow:5px 5px 0 ${desktop ? '#664517' : '#171717'}}header{display:flex;align-items:center;justify-content:space-between;border-bottom:2px solid ${desktop ? '#f0b84a' : '#171717'};padding:10px;font-weight:800;text-transform:uppercase}.badge{border:1px solid currentColor;padding:3px 6px}.lane{padding:10px}.card{border:2px solid ${desktop ? '#6f7685' : '#171717'};border-left:6px solid ${desktop ? '#f0b84a' : '#ff4d00'};padding:11px;background:${desktop ? '#20242c' : '#fff'};margin-bottom:10px}.title{font-weight:900;font-size:14px}.meta{opacity:.7;margin:7px 0}.controls{display:flex;flex-wrap:wrap;gap:7px;padding-top:9px;border-top:1px solid #777}.btn{min-height:44px;border:2px solid currentColor;background:transparent;color:inherit;padding:8px 12px;font:inherit;font-weight:800}.btn.primary{background:${desktop ? '#f0b84a' : '#171717'};color:${desktop ? '#111318' : '#fffdf6'}}.snooze{border:1px dashed currentColor;padding:8px;margin-top:8px}.wake{display:flex;align-items:center;justify-content:space-between}.live{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}footer{padding:8px 10px;border-top:2px solid ${desktop ? '#f0b84a' : '#171717'};font-size:11px;opacity:.75}@media(max-width:340px){main{padding:8px}.card{padding:8px}.btn{padding:7px 9px}}
</style></head><body><main><section class="shell"><header><span>${desktop ? 'Hermes Desktop' : 'Kanban WebUI'}</span><span class="badge">ATTENTION</span></header><div class="lane"><article class="card"><div class="title">Privacy-safe evidence task</div><div class="meta">RUNNING · canary-worker · no user data</div><div class="controls"><button class="btn">Settle</button><button class="btn">Snooze…</button></div><div class="snooze"><strong>Snoozed controls</strong><br>1 hour · Tomorrow 9:00 local · One week</div><span class="live" role="status" aria-live="polite">Task snoozed</span></article><article class="card"><div class="wake"><div><div class="title">Settled task</div><div class="meta">SETTLED · revision 2</div></div><button class="btn primary">Wake</button></div><span class="live" role="status" aria-live="polite">Task awake</span></article></div><footer>Responsive evidence fixture · settled/snoozed/wake · persistent live region</footer></section></main></body></html>`
}

for (const surface of ['webui', 'desktop']) {
  for (const width of widths) {
    const page = await browser.newPage({ viewport: { width, height: 620 }, deviceScaleFactor: 1 })
    await page.setContent(markup(surface), { waitUntil: 'load' })
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)
    if (overflow) throw new Error(`${surface} ${width}px overflows`)
    const filename = `${surface}-${width}.png`
    const file = path.join(output, filename)
    await page.screenshot({ path: file, fullPage: true })
    const bytes = await (await import('node:fs/promises')).readFile(file)
    manifest.captures.push({ surface, width, height: 620, filename, bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex'), horizontalOverflow: false })
    await page.close()
  }
}
await browser.close()
await writeFile(path.join(output, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`)
