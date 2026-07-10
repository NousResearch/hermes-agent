#!/usr/bin/env node
/* contrast-audit-run.mjs — ตัวรัน contrast-audit.js อัตโนมัติ (headless browser)
 *
 * ทำไมต้องมี: contrast-audit.js เป็นสคริปต์ browser (ใช้ document/getComputedStyle)
 * รันด้วย `node contrast-audit.js` ตรง ๆ ไม่ได้ (document is not defined)
 * ตัวนี้เปิดหน้า HTML ใน chromium headless → ฉีด contrast-audit.js → อ่านจำนวน fail → คืน exit code
 *
 * ใช้:  node contrast-audit-run.mjs <หน้า.html> [<หน้าที่2.html> ...]
 * ผล:  exit 0 = ทุกหน้า 0 fail · exit 1 = มีข้อความตก WCAG AA · exit 2 = ใช้ผิด/เปิดหน้าไม่ได้
 * ได้ไฟล์ภาพ <หน้า>-contrast-audit.png ทุกหน้าไว้ดูด้วยตา
 *
 * ต้องมี playwright (npm i playwright) — ถ้าไม่มีจะบอกวิธีติดตั้ง ไม่ crash งง ๆ
 */
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dir = dirname(fileURLToPath(import.meta.url));
const pages = process.argv.slice(2);

if (!pages.length) {
  console.error('ใช้: node contrast-audit-run.mjs <หน้า.html> [หน้าที่2.html ...]');
  console.error('เช่น: node tools/contrast-audit-run.mjs preview/onemanfleet-ds.html');
  process.exit(2);
}

let chromium;
try {
  ({ chromium } = await import('playwright'));
} catch {
  console.error('❌ ยังไม่มี playwright — ติดตั้งครั้งเดียวด้วย: npm i playwright');
  console.error('   (ตัว browser ไม่ต้องโหลดถ้ามี cache แล้ว · ถ้าไม่มี: npx playwright install chromium)');
  process.exit(2);
}

const auditSrc = readFileSync(join(__dir, 'contrast-audit.js'), 'utf8');
let browser;
try {
  browser = await chromium.launch();
} catch (e) {
  console.error('❌ เปิด chromium ไม่ได้:', e.message.slice(0, 140));
  console.error('   ลอง: npx playwright install chromium');
  process.exit(2);
}

let totalFail = 0;
for (const rel of pages) {
  const abs = resolve(process.cwd(), rel);
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  try {
    await page.goto(pathToFileURL(abs).href, { waitUntil: 'networkidle', timeout: 15000 });
  } catch (e) {
    console.error(`❌ เปิดหน้าไม่ได้: ${rel} — ${e.message.slice(0, 100)}`);
    await page.close();
    totalFail += 1;
    continue;
  }
  const r = await page.evaluate(auditSrc);
  const shot = abs.replace(/\.html?$/i, '') + '-contrast-audit.png';
  await page.screenshot({ path: shot, fullPage: true }).catch(() => {});
  await page.close();

  const mark = r.fails > 0 ? '❌' : '✅';
  console.log(`${mark} ${rel} — ${r.fails} fail · ${r.unique} คู่สีไม่ซ้ำ · ข้ามพื้นไล่สี ${r.gradient_skipped}`);
  if (r.worst && r.worst.length) console.table(r.worst);
  console.log(`   ภาพ: ${shot}`);
  totalFail += r.fails;
}

await browser.close();
console.log(totalFail > 0 ? `\nรวม ${totalFail} จุดตก WCAG AA — ต้องแก้ก่อนปิดงาน` : '\n✅ ทุกหน้าผ่าน WCAG AA (0 fail)');
process.exit(totalFail > 0 ? 1 : 0);
