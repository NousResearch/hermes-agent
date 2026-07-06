// contrast-check — ตรวจคู่สีตามสูตร WCAG 2.x (relative luminance) · ใช้กับหน้า preview + token
// ใช้: node contrast-check.mjs   → exit 1 ถ้ามีคู่ตก AA (4.5:1 ข้อความปกติ · 3:1 ข้อความใหญ่/UI)

function hexToRgb(hex) {
  const h = hex.replace('#', '');
  const v = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
  return [0, 2, 4].map((i) => parseInt(v.slice(i, i + 2), 16) / 255);
}
function lum(hex) {
  const [r, g, b] = hexToRgb(hex).map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
function ratio(fg, bg) {
  const [a, b] = [lum(fg), lum(bg)].sort((x, y) => y - x);
  return (a + 0.05) / (b + 0.05);
}
// ผสมสีโปร่งบนพื้น (จำลอง color-mix 12-15%)
function mix(fg, bg, pct) {
  const f = hexToRgb(fg), b = hexToRgb(bg);
  const m = f.map((c, i) => c * pct + b[i] * (1 - pct));
  return '#' + m.map((c) => Math.round(c * 255).toString(16).padStart(2, '0')).join('');
}

const WHITE = '#ffffff', CANVAS = '#f6f6f7', INSET = '#ededef', INK = '#231F20';
const D_CANVAS = '#18181b', D_SURFACE = '#26262b', D_INSET = '#202024';

const pairs = [
  // [ชื่อ, fg, bg, เกณฑ์]
  ['text-default บนขาว', INK, WHITE, 4.5],
  ['text-default บน canvas', INK, CANVAS, 4.5],
  ['text-muted บนขาว', 'MUTED_L', WHITE, 4.5],
  ['text-muted บน canvas', 'MUTED_L', CANVAS, 4.5],
  ['text-muted บน inset (hint ในการ์ดเทา)', 'MUTED_L', INSET, 4.5],
  ['เมนู sidebar ข้อความ', '#d6d4d2', INK, 4.5],
  ['เมนู sidebar หัวกลุ่ม', 'MENUGRP', INK, 4.5],
  ['ปุ่มโซน sidebar', '#c9c7c5', INK, 4.5],
  ['badge สำเร็จ (ข้อความ)', 'SUCCESS_T', mix('#16C79A', WHITE, 0.14), 4.5],
  ['badge เตือน (ข้อความ)', 'WARNING_T', mix('#B9770E', WHITE, 0.15), 4.5],
  ['badge อันตราย (ข้อความ)', 'DANGER_T', mix('#EC2C23', WHITE, 0.13), 4.5],
  ['ข้อความเตือน conflict บนขาว', 'WARNING_T', WHITE, 4.5],
  ['สถานะ error บน inset', 'DANGER_T', INSET, 4.5],
  ['ปุ่มหลัก ขาวบนแดงเข้ม', WHITE, '#851914', 4.5],
  ['dark: text-default', '#f6f6f7', D_CANVAS, 4.5],
  ['dark: text-muted บน canvas', 'MUTED_D', D_CANVAS, 4.5],
  ['dark: text-muted บน surface', 'MUTED_D', D_SURFACE, 4.5],
  ['dark: ปุ่มหลัก ขาวบนแดง', WHITE, 'ACTION_D', 4.5],
  ['dark: badge สำเร็จ', 'SUCCESS_TD', mix('#16C79A', D_SURFACE, 0.14), 4.5],
  ['dark: badge เตือน', 'WARNING_TD', mix('#B9770E', D_SURFACE, 0.15), 4.5],
  ['dark: badge อันตราย', 'DANGER_TD', mix('#EC2C23', D_SURFACE, 0.13), 4.5],
  ['dark: ข้อความเตือน conflict', 'WARNING_TD', D_SURFACE, 4.5],
  // ==== ชุด dark-first ของหน้า explorer v2 (2026-07-05) ====
  ['v2: text-default บน canvas มืด', '#f4f4f6', '#0e0e11', 4.5],
  ['v2: text-muted บน canvas มืด', '#a4a4ae', '#0e0e11', 4.5],
  ['v2: text-dim (eyebrow/caption) บน canvas มืด', '#7c7c86', '#0e0e11', 4.5],
  ['v2: เมนู sidebar ตัวอักษร', '#b9b7b5', '#101013', 4.5],
  ['v2: หัวกลุ่มเมนู', '#a8a6a4', '#101013', 4.5],
  ['v2: gradient text ฝั่งแดง (จุดมืดสุด)', '#ff6a5e', '#0e0e11', 4.5],
  ['v2: gradient text ฝั่งทอง', '#E8C87B', '#0e0e11', 4.5],
  ['v2: lux-glass label', '#b9b9c0', '#131316', 4.5],
];

// ===== ค่าที่ใช้จริง (แก้ตรงนี้ให้ตรงกับหน้า/token) =====
const TOKENS = {
  MUTED_L: '#54545d',    // เดิม #71717c (ตกบน inset 4.29)
  MUTED_D: '#a4a4ae',    // เดิม #9a9aa4
  MENUGRP: '#a8a6a4',    // เดิม #8b8987
  SUCCESS_T: '#0b6e50',  // ข้อความสำเร็จ (เดิมใช้ #16C79A = ตกหนัก)
  WARNING_T: '#7d5300',  // ข้อความเตือน (เดิม #B9770E = ตก)
  DANGER_T: '#a81d13',   // ข้อความอันตราย (เดิม #EC2C23 = ตก)
  ACTION_D: '#c9221a',   // ปุ่มแดงโหมดมืด (เดิม #EC2C23 = 4.24 ตก)
  SUCCESS_TD: '#43e5b5', // โหมดมืด: ข้อความสถานะต้องสว่างขึ้น
  WARNING_TD: '#ffc966',
  DANGER_TD: '#ff8a80',
};

let fail = 0;
console.log('คู่สี'.padEnd(44) + 'ratio   เกณฑ์  ผล');
for (const [name, fgRaw, bgRaw, min] of pairs) {
  const fg = TOKENS[fgRaw] ?? fgRaw;
  const bg = TOKENS[bgRaw] ?? bgRaw;
  const r = ratio(fg, bg);
  const ok = r >= min;
  if (!ok) fail++;
  console.log(name.padEnd(46) + r.toFixed(2).padStart(5) + '   ' + min + '    ' + (ok ? 'ผ่าน' : '✗ ตก'));
}
console.log(fail === 0 ? '\nทุกคู่ผ่าน WCAG AA' : `\nตก ${fail} คู่`);
process.exit(fail === 0 ? 0 : 1);
