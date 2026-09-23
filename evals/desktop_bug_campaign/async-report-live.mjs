import { chromium } from '../../node_modules/playwright/index.mjs'
import fs from 'node:fs'
import { execFileSync } from 'node:child_process'
const a=process.env.ASYNC_REPORT_ARTIFACT_DIR
if (!a) throw new Error('Set ASYNC_REPORT_ARTIFACT_DIR to an offline artifact directory')
const tag=process.argv[2]??'after'
const browser=await chromium.launch({headless:true,args:['--no-sandbox']})
const page=await browser.newPage({viewport:{width:1400,height:1000}})
const errors=[];page.on('pageerror',e=>errors.push(String(e)))
const producer=JSON.parse(fs.readFileSync(`${a}/producer.json`,'utf8'))
const cases={}
for (const [name, entry] of Object.entries(producer.cases)) {
  await page.goto('http://127.0.0.1:18164/async-report-probe.html'+(tag==='before'?'?baseline':''),{waitUntil:'domcontentloaded',timeout:120000})
  await page.waitForFunction(()=>typeof window.renderProducer==='function')
  const projected=await page.evaluate(rows=>window.renderProducer(rows),entry.rows)
  await page.waitForSelector('#producer [data-role="system"]')
  if (projected.hydrated[0].asyncResult?.includes('# Generated')) await page.waitForSelector('#producer table')
  const dom=await page.locator('#producer').evaluate(el=>({text:el.textContent,headings:el.querySelectorAll('h1').length,tables:el.querySelectorAll('table').length,strong:el.querySelectorAll('strong').length,html:el.innerHTML}))
  cases[name]={...projected,dom}
  if (name==='cron') { await page.locator('#producer').scrollIntoViewIfNeeded(); await page.waitForTimeout(800); await page.locator('#producer').screenshot({path:`${a}/${tag}.png`}) }
}
const result={cases,errors,sourceSha:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim()}
fs.writeFileSync(`${a}/${tag}.json`,JSON.stringify(result,null,2))
console.log(JSON.stringify({errors,cases:Object.fromEntries(Object.entries(cases).map(([name,c])=>[name,{headings:c.dom.headings,tables:c.dom.tables,privateLeak:c.dom.text.includes('PRIVATE'),body:c.hydrated[0].asyncResult??null}]))},null,2))
await browser.close()
