#!/usr/bin/env node
/* contrast-audit-run.mjs — ตัวรัน contrast-audit.js อัตโนมัติ (headless browser)
 *
 * ทำไมต้องมี: contrast-audit.js เป็นสคริปต์ browser (ใช้ document/getComputedStyle)
 * รันด้วย `node contrast-audit.js` ตรง ๆ ไม่ได้ (document is not defined)
 * ตัวนี้เปิดหน้า HTML ใน chromium headless → ฉีด contrast-audit.js → อ่านจำนวน fail → คืน exit code
 *
 * ใช้:  node contrast-audit-run.mjs <หน้า.html> [<หน้าที่2.html> ...]
 * ผล:  exit 0 = ทุกหน้า 0 fail · exit 1 = มีข้อความตก WCAG AA (หรือ audit คืนค่าผิดรูป) · exit 2 = ใช้ผิด/เปิดหน้าไม่ได้
 * ได้ไฟล์ภาพ <หน้า>-contrast-audit.png ทุกหน้าไว้ดูด้วยตา
 *
 * ⚠ ข้อจำกัดสำคัญ (ต้องรู้ก่อนใช้กับงานหลังบ้าน/admin):
 *   ตรวจแค่ element ที่ "แสดงบนหน้า" ตอนโหลด (ข้าม element ที่ offsetParent=null คือถูกซ่อน)
 *   งาน admin ที่มี modal / drawer / แท็บ / 5 UI states (ว่าง/โหลด/ผิดพลาด/บางส่วน/สมบูรณ์) ที่ซ่อนอยู่
 *   จะ "ไม่ถูกตรวจ" ถ้าไม่เปิดให้แสดงก่อน → contrast ตรงนั้นอาจพังโดยไม่มีใครจับ
 *   วิธีชัวร์: ทำหน้า/ไฟล์ HTML แยกต่อ state (หรือตั้งให้เปิด default) แล้วส่งทุกไฟล์เข้า runner:
 *     node contrast-audit-run.mjs admin-empty.html admin-error.html admin-modal.html ...
 *
 * ต้องมี playwright (npm i playwright) — ถ้าไม่มีจะบอกวิธีติดตั้ง ไม่ crash งง ๆ
 */
import { readFileSync } from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

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
  console.error('   (ถ้ายังไม่มี chromium: npx playwright install chromium)');
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
try {
  for (const rel of pages) {
    const abs = resolve(process.cwd(), rel);
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    try {
      // ใช้ 'load' (ไม่ใช่ networkidle ที่ค้างกับ file:// ถ้ามี asset) แล้วรอฟอนต์พร้อมก่อนวัด
      await page.goto(pathToFileURL(abs).href, { waitUntil: 'load', timeout: 15000 });
      await page.evaluate(() => (document.fonts && document.fonts.ready) || null).catch(() => {});

      const r = await page.evaluate(auditSrc);
      // ต้องได้ตัวเลข fails จริง — ถ้าไม่ (audit คืนค่าผิดรูป) นับเป็น "ไม่ผ่าน" ไม่ใช่ปล่อยผ่านเงียบ
      const fails = Number(r && r.fails);
      if (!Number.isFinite(fails)) {
        console.error(`❌ ${rel} — audit คืนค่าผิดรูป (ไม่มี fails เป็นตัวเลข) ถือว่าไม่ผ่าน`);
        totalFail += 1;
        continue;
      }

      const shot = abs.replace(/\.html?$/i, '') + '-contrast-audit.png';
      await page.screenshot({ path: shot, fullPage: true }).catch(() => {});

      const mark = fails > 0 ? '❌' : '✅';
      console.log(`${mark} ${rel} — ${fails} fail · ${r.unique ?? '?'} คู่สีไม่ซ้ำ · ข้ามพื้นไล่สี ${r.gradient_skipped ?? 0}`);
      if (r.worst && r.worst.length) console.table(r.worst);
      console.log(`   ภาพ: ${shot}`);
      totalFail += fails;
    } catch (e) {
      console.error(`❌ ${rel} — ตรวจไม่สำเร็จ: ${String(e.message || e).slice(0, 120)} (ถือว่าไม่ผ่าน)`);
      totalFail += 1;
    } finally {
      await page.close().catch(() => {});
    }
  }
} finally {
  await browser.close().catch(() => {});
}

console.log(totalFail > 0 ? `\nรวม ${totalFail} จุดตก WCAG AA / ตรวจไม่ผ่าน — ต้องแก้ก่อนปิดงาน` : '\n✅ ทุกหน้าผ่าน WCAG AA (0 fail)');
process.exit(totalFail > 0 ? 1 : 0);
