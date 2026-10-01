import {chromium} from 'playwright';
import fs from 'node:fs';
import assert from 'node:assert/strict';
const out=process.env.THREAD_SCROLL_OUTPUT;
assert.ok(out, 'Set THREAD_SCROLL_OUTPUT to the isolated artifact directory');
fs.mkdirSync(out,{recursive:true});
const url=process.env.THREAD_SCROLL_URL ?? 'http://127.0.0.1:18480/scroll-campaign-probe.html?thread';
const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
try {
const page=await browser.newPage({viewport:{width:1200,height:800}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.goto(url,{waitUntil:'domcontentloaded',timeout:120000});
const el=page.locator('[data-slot="aui_thread-viewport"]');await el.waitFor({timeout:120000});await page.waitForTimeout(4000);
const metrics=()=>el.evaluate(x=>({top:x.scrollTop,height:x.clientHeight,scrollHeight:x.scrollHeight,following:x.dataset.following,gap:x.scrollHeight-x.clientHeight-x.scrollTop}));
const load=await metrics();await el.hover();await page.mouse.wheel(0,-900);await page.waitForTimeout(500);const read=await metrics();
await page.click('#switch-first');await page.waitForTimeout(1000);const other=await metrics();await page.click('#switch-first');await page.waitForTimeout(4000);const returned=await metrics();
await el.hover();await page.mouse.wheel(0,-900);await page.waitForTimeout(500);const beforeReload=await metrics();await page.click('#reload-first');await page.waitForTimeout(4000);const reloaded=await metrics();
const result={load,read,other,returned,beforeReload,reloaded,errors};fs.writeFileSync(`${out}/${process.env.PROBE_TAG??'probe'}.json`,JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
assert.ok(load.gap<15,'new session must follow its tail');assert.ok(read.gap>400,'wheel must escape');assert.ok(Math.abs(read.gap-returned.gap)<15,'A→B→A must preserve reading distance');assert.equal(returned.following,'false');assert.ok(Math.abs(beforeReload.gap-reloaded.gap)<15,'delayed transcript reload must preserve reading distance');
}finally{await browser.close()}
