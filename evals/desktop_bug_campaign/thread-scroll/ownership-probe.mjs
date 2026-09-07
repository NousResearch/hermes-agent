import {chromium} from 'playwright';
import fs from 'node:fs';
import assert from 'node:assert/strict';
const out=process.env.THREAD_SCROLL_OUTPUT;
assert.ok(out);
const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
const result={};
try {
 const page=await browser.newPage({viewport:{width:1200,height:800}});
 const errors=[]; page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:18480/scroll-campaign-probe.html?ownership',{timeout:120000});
 const el=page.locator('[data-slot="aui_thread-viewport"]'); await el.waitFor({timeout:120000}); await page.waitForTimeout(4000);
 const metrics=()=>el.evaluate(x=>({top:x.scrollTop,height:x.clientHeight,scrollHeight:x.scrollHeight,following:x.dataset.following,gap:x.scrollHeight-x.clientHeight-x.scrollTop}));
 const storage=()=>page.evaluate(()=>Object.fromEntries(Object.entries(localStorage).filter(([k])=>k.includes('threadScroll'))));
 for(const [name,profile,remote] of [['profile','other',null],['gateway','other','https://probe.invalid']]) {
  await el.hover(); await page.mouse.wheel(0,-900); await page.waitForTimeout(500);
  const before=await metrics(); const storedBefore=await storage();
  await page.evaluate(([p,r])=>window.scrollProbe.scope(p,r),[profile,remote]); await page.waitForTimeout(4000);
  result[name]={before,storedBefore,after:await metrics(),storedAfter:await storage()};
 }
 await el.hover(); await page.mouse.wheel(0,-500); await page.waitForTimeout(500);
 result.remount={before:await metrics()}; await page.evaluate(()=>window.scrollProbe.remount()); await page.waitForTimeout(4000); result.remount.after=await metrics();
 result.reload={expected:result.profile.before}; await page.reload(); await el.waitFor(); await page.waitForTimeout(4000); result.reload.after=await metrics();
 // A clamped remembered offset keeps restoration pending while later history
 // is deferred. Trusted wheel input is delivered before releasing that history.
 await page.evaluate(()=>window.scrollProbe.partial()); await page.waitForTimeout(700);
 result.hydration={partial:await metrics()}; await el.hover(); await page.mouse.wheel(0,500); await page.waitForTimeout(100); result.hydration.beforeWheel=await metrics(); await page.mouse.wheel(0,-200); await page.waitForTimeout(100); result.hydration.wheel=await metrics();
 await page.evaluate(()=>window.scrollProbe.release()); await page.waitForTimeout(4000); result.hydration.released=await metrics();
 result.long={};
 await page.evaluate(()=>localStorage.clear());
 await page.goto('http://127.0.0.1:18480/scroll-campaign-probe.html?thread'); await el.waitFor(); await page.waitForTimeout(4000);
 await el.hover(); await page.mouse.wheel(0,-3100); await page.waitForTimeout(500);
 result.long.before=await metrics(); await page.evaluate(()=>window.scrollProbe.remount()); await page.waitForTimeout(4000); result.long.remount=await metrics();
 await page.reload(); await el.waitFor(); await page.waitForTimeout(4000); result.long.reload=await metrics();
 result.streaming={before:await metrics()}; await page.evaluate(()=>window.scrollProbe.run()); await page.waitForTimeout(100); await page.evaluate(()=>window.scrollProbe.grow()); await page.waitForTimeout(1500); result.streaming.after=await metrics();
 result.errors=errors;
 result.checks={profile:result.profile.after.gap<15,gateway:result.gateway.after.gap<15,remount:Math.abs(result.remount.before.gap-result.remount.after.gap)<15,reload:Math.abs(result.reload.expected.gap-result.reload.after.gap)<15,hydration:Math.abs(result.hydration.released.gap-result.hydration.wheel.gap)<32 && result.hydration.released.following==='false'};
 Object.assign(result.checks,{longRemount:Math.abs(result.long.before.gap-result.long.remount.gap)<15,longReload:Math.abs(result.long.before.gap-result.long.reload.gap)<15,streaming:result.streaming.after.gap>result.streaming.before.gap+100 && Math.abs(result.streaming.after.top-result.streaming.before.top)<15});
 fs.writeFileSync(`${out}/${process.env.PROBE_TAG??'ownership'}.json`,JSON.stringify(result,null,2)); console.log(JSON.stringify(result,null,2));
 assert.ok(result.profile.after.gap<15,'incoming profile must not inherit outgoing position');
 assert.ok(result.gateway.after.gap<15,'incoming gateway must not inherit outgoing position');
 assert.ok(Math.abs(result.remount.before.gap-result.remount.after.gap)<15,'remount preserves position');
 assert.ok(Math.abs(result.reload.expected.gap-result.reload.after.gap)<15,'document reload restores the default profile, not remote state');
 assert.ok(Math.abs(result.hydration.released.gap-result.hydration.wheel.gap)<32,'wheel during deferred hydration must replace the parked restore with the reader position');
 assert.equal(result.hydration.released.following,'false');
 assert.ok(Math.abs(result.long.before.gap-result.long.remount.gap)<15,'long remount preserves reading distance despite intrinsic row measurements');
 assert.ok(Math.abs(result.long.before.gap-result.long.reload.gap)<15,'long document reload preserves reading distance');
 assert.ok(result.streaming.after.gap > result.streaming.before.gap + 100,'live growth must stop restoring distance from bottom');
 assert.ok(Math.abs(result.streaming.after.top-result.streaming.before.top)<15,'live growth below the reader must not move their viewport');
 assert.deepEqual(errors,[]);
} finally {await browser.close()}
