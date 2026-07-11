/* contrast-audit.js — ตัวตรวจ contrast "หน้าจริง" (แทนลิสต์ตายตัว)
 *
 * ทำไมต้องมี: contrast-check.mjs เดิมเช็คแค่คู่สีที่เขียนไว้ตายตัว + เทียบพื้นมืดสุดพื้นเดียว
 * เลยพลาดข้อความบนการ์ดพื้นสว่างกว่า (เช่นตารางหลังบ้าน) — เกิดเหตุจริง 2026-07-06
 * ตัวนี้เดินทุก element ที่มีข้อความบนหน้าที่เรนเดอร์จริง คำนวณพื้นโปร่งแสง (composite)
 * และข้ามพื้นไล่สี/รูป (ตัดสินอัตโนมัติไม่ได้) แล้วรายงานคู่ที่ตก WCAG AA
 *
 * วิธีใช้ (ต้องรันบน "หน้าที่เรนเดอร์แล้ว" — ทั้งแท็บ Front และ Admin):
 *   1) เปิดหน้าโชว์ในเบราว์เซอร์ (หรือ preview server)
 *   2) วางไฟล์นี้ทั้งไฟล์ใน DevTools Console แล้ว Enter  → พิมพ์ผล + คืน object
 *   3) สลับไปแท็บ Admin แล้วรันซ้ำ (ต้อง 0 fail ทั้งสองแท็บ)
 * หรือรันผ่าน AI: อ่านไฟล์นี้แล้วส่งเข้า preview_eval / console ของหน้าที่เปิดอยู่
 *
 * เกณฑ์: ปกติ 4.5:1 · ตัวใหญ่ (>=24px หรือ >=18.66px หนา>=700) 3:1
 * exit rule (ในบริบท CI ที่มี headless): window.__CONTRAST_FAILS__ = จำนวน fail
 */
(function () {
  if (typeof document === 'undefined') {
    var msg = '⚠ contrast-audit.js ต้องรันในเบราว์เซอร์ (มี document) ไม่ใช่ node ตรง ๆ\n' +
      '  วิธีที่ถูก:\n' +
      '   • อัตโนมัติ: node tools/contrast-audit-run.mjs <หน้า.html>\n' +
      '   • มือ: วางทั้งไฟล์นี้ใน DevTools Console ของหน้าที่เปิดอยู่';
    if (typeof console !== 'undefined') console.error(msg);
    if (typeof process !== 'undefined' && process.exit) process.exit(2);
    return;
  }
  function pRGB(s) {
    const m = s && s.match(/rgba?\(([^)]+)\)/);
    if (!m) return null;
    const p = m[1].split(',').map(parseFloat);
    return { r: p[0], g: p[1], b: p[2], a: p[3] === undefined ? 1 : p[3] };
  }
  function over(f, b) {
    const a = f.a;
    return { r: f.r * a + b.r * (1 - a), g: f.g * a + b.g * (1 - a), b: f.b * a + b.b * (1 - a), a: 1 };
  }
  // พื้นจริงใต้ข้อความ: composite ทุกชั้นโปร่งแสงลงบนชั้นทึบที่ใกล้สุด
  // คืน {grad:true} ถ้าเจอพื้นไล่สี/รูป (ตัดสินอัตโนมัติไม่ได้ — ต้องตรวจด้วยตา)
  function effBg(el) {
    const layers = [];
    let e = el;
    while (e) {
      const cs = getComputedStyle(e);
      if (cs.backgroundImage !== 'none' && /gradient|url/.test(cs.backgroundImage)) return { grad: true };
      const p = pRGB(cs.backgroundColor);
      if (p && p.a > 0) { layers.push(p); if (p.a >= 1) break; }
      e = e.parentElement;
    }
    let base = { r: 14, g: 14, b: 17, a: 1 }; // fallback: canvas มืด
    if (layers.length && layers[layers.length - 1].a >= 1) base = layers.pop();
    for (let i = layers.length - 1; i >= 0; i--) base = over(layers[i], base);
    return base;
  }
  function lum({ r, g, b }) {
    const f = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
  }
  function ratio(a, b) {
    const L1 = lum(a), L2 = lum(b), hi = Math.max(L1, L2), lo = Math.min(L1, L2);
    return (hi + 0.05) / (lo + 0.05);
  }

  const fails = [], gradSkipped = [];
  document.querySelectorAll('body *').forEach((el) => {
    if (!el.offsetParent || el.offsetWidth === 0) return;
    const own = [...el.childNodes]
      .filter((n) => n.nodeType === 3 && n.textContent.trim().length > 1)
      .map((n) => n.textContent.trim()).join(' ');
    if (!own) return;
    const cs = getComputedStyle(el);
    if (cs.webkitTextFillColor === 'rgba(0, 0, 0, 0)') return; // ข้อความไล่สี (bg-clip) ตรวจด้วยตา
    let fg = pRGB(cs.color);
    if (!fg || fg.a === 0) return;
    const bg = effBg(el);
    if (bg.grad) { gradSkipped.push(own.slice(0, 20)); return; }
    if (fg.a < 1) fg = over(fg, bg);
    const size = parseFloat(cs.fontSize), bold = parseInt(cs.fontWeight) >= 700;
    const large = size >= 24 || (size >= 18.66 && bold);
    const r = ratio(fg, bg), min = large ? 3 : 4.5;
    if (r < min - 0.05) {
      fails.push({
        text: own.slice(0, 24), size: Math.round(size) + 'px', ratio: +r.toFixed(2), need: min,
        fg: `rgb(${[fg.r, fg.g, fg.b].map(Math.round).join(',')})`,
        bg: `rgb(${[bg.r, bg.g, bg.b].map(Math.round).join(',')})`,
      });
    }
  });

  const seen = new Set(), uniq = [];
  for (const f of fails) { const k = f.fg + '|' + f.bg + '|' + f.size; if (!seen.has(k)) { seen.add(k); uniq.push(f); } }
  uniq.sort((a, b) => a.ratio - b.ratio);

  const tab = (document.querySelector('.side-head, header, [class*=head]') || {}).innerText || '';
  console.log(`\n=== contrast-audit (หน้าจริง) ===`);
  console.log(`ข้อความที่ตก WCAG AA: ${fails.length} จุด (${uniq.length} คู่สีไม่ซ้ำ)`);
  if (gradSkipped.length) console.log(`ข้าม (พื้นไล่สี ตรวจด้วยตา): ${gradSkipped.length} จุด`);
  if (uniq.length) console.table(uniq.slice(0, 25));
  else console.log('✅ ผ่านทุกข้อความบนหน้านี้');
  console.log('อย่าลืมสลับแท็บ (Front/Admin) แล้วรันซ้ำ — ต้อง 0 ทั้งสองฝั่ง');

  window.__CONTRAST_FAILS__ = fails.length;
  return { fails: fails.length, unique: uniq.length, worst: uniq.slice(0, 15), gradient_skipped: gradSkipped.length };
})();
