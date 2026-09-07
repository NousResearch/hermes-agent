import { chromium } from '../../node_modules/playwright/index.mjs'
import fs from 'node:fs'
const a='/home/teknium/.hermes/cache/desktop-bugs-74848ed3/navigation-markdown'
const tag=process.argv[2]??'before'
const browser=await chromium.launch({headless:false,args:['--no-sandbox']})
const page=await browser.newPage({viewport:{width:1400,height:1000}})
const errors=[];page.on('pageerror',e=>errors.push(String(e)))
await page.goto('http://127.0.0.1:18160/navigation-markdown-probe.html',{waitUntil:'domcontentloaded',timeout:120000})
await page.waitForSelector('#hard p',{timeout:90000})
await page.waitForTimeout(6000)
await page.waitForSelector('#hard p',{timeout:90000})
const result=await page.evaluate(()=>({inputs:window.probeInputs,outputs:window.probeOutputs,hardBr:document.querySelectorAll('#hard br').length,softBr:document.querySelectorAll('#soft br').length,text:document.body.innerText,html:document.querySelector('main').innerHTML}))
result.errors=errors
await page.screenshot({path:`${a}/markdown-${tag}.png`,fullPage:true})
fs.writeFileSync(`${a}/markdown-${tag}.json`,JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2))
await browser.close()
