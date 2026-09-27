import fs from 'node:fs'
import { chromium } from '@playwright/test'

const browser = await chromium.launch({ headless: true })
const page = await browser.newPage({ viewport: { width: 1220, height: 800 }, deviceScaleFactor: 1 })
const html = `<!doctype html><style>
*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;font-family:"Segoe UI",sans-serif;background:#eef2f7;color:rgb(255 255 255 / .94)}
.wallpaper{position:absolute;inset:0;background:linear-gradient(135deg,#f8fafc 0 25%,#dbeafe 25% 50%,#fef3c7 50% 75%,#e0e7ff 75%);background-size:240px 240px;padding:52px 72px}
.shell{height:100%;display:grid;grid-template-columns:220px 1fr;gap:12px}.sidebar{background:rgb(13 13 14 / .35);backdrop-filter:blur(20px);border-radius:14px;padding:22px}.chat{display:flex;flex-direction:column;min-width:0}
.frame{flex:1;display:flex;flex-direction:column;padding:44px 58px 28px;border-radius:13px}.after .frame{background:color-mix(in srgb,#0d0d0e 86%,transparent);border:1px solid rgb(255 255 255 / .12);box-shadow:0 18px 48px rgb(0 0 0 / .22);backdrop-filter:blur(24px) saturate(1.08)}
.message{font-size:14px;line-height:1.55;color:rgb(255 255 255 / .94);max-width:650px}.meta{margin-top:18px;font-size:12px;color:rgb(255 255 255 / .64)}.composer{margin-top:auto;background:rgb(13 13 14 / .94);border:1px solid rgb(255 255 255 / .12);border-radius:13px;padding:15px 18px}.placeholder{font-size:13px;color:rgb(255 255 255 / .70)}h1{font-size:16px;margin:0 0 28px}.sidebar p{color:rgb(255 255 255 / .74)}
</style><div class="wallpaper"><div class="shell"><aside class="sidebar"><h1>Agent Czesiek</h1><p>Sessions</p><p>Projects</p></aside><main class="chat"><section class="frame"><div class="message">Oto odpowiedź agenta. Tekst wiadomości powinien pozostać czytelny niezależnie od tapety i materiału okna.</div><div class="meta">Agent Czesiek · przed chwilą</div><div class="composer"><div class="placeholder">Napisz wiadomość…</div></div></section></main></div></div>`
await page.setContent(html)
const phase=process.argv[2]
if(phase==='after') await page.locator('body').evaluate(el=>el.classList.add('after'))
const result=await page.evaluate((phase)=>{
 const parse=s=>(s.match(/[\d.]+/g)||[]).map(Number); const lin=v=>{v/=255;return v<=.04045?v/12.92:Math.pow((v+.055)/1.055,2.4)}; const lum=c=>.2126*lin(c[0])+.7152*lin(c[1])+.0722*lin(c[2]); const mix=(fg,bg)=>{const a=fg[3]??1;return [0,1,2].map(i=>fg[i]*a+bg[i]*(1-a))}; const ratio=(a,b)=>{const [hi,lo]=[lum(a),lum(b)].sort((x,y)=>y-x);return Math.round(((hi+.05)/(lo+.05))*100)/100};
 const sample=(sel)=>{const el=document.querySelector(sel),cs=getComputedStyle(el),r=el.getBoundingClientRect(); const x=Math.round(r.left+8),y=Math.round(r.top+r.height/2); const backdrop=phase==='after'?mix([13,13,14,.86],[238,242,247]):[238,242,247]; const fg=mix(parse(cs.color),backdrop); return {selector:sel,color:cs.color,fontSize:parseFloat(cs.fontSize),background:backdrop.map(v=>Math.round(v*100)/100),ratio:ratio(fg,backdrop),point:[x,y]};};
 const f=getComputedStyle(document.querySelector('.frame')); return {phase,state:'dark glass chat over emulated bright native backdrop #eef2f7',frame:{background:f.backgroundColor,border:f.borderColor,borderRadius:f.borderRadius,backdropFilter:f.backdropFilter},samples:['.message','.meta','.placeholder'].map(sample)};
},phase)
fs.writeFileSync(`pr-assets/chat-contrast-${phase}.json`,JSON.stringify(result,null,2))
await page.screenshot({path:`pr-assets/chat-contrast-${phase}.png`})
console.log(JSON.stringify(result))
await browser.close()
