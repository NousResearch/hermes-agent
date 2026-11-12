#!/usr/bin/env node
/**
 * slopscan.mjs — AI-slop design linter, zero dependencies, read-only
 * ponytail: regex tokenization over CSS blocks, no full parser needed for these rules
 */
import { readFileSync, statSync, readdirSync } from 'node:fs';
import { join, extname, resolve } from 'node:path';

const SCAN_EXTS = new Set(['.css','.scss','.html','.jsx','.tsx','.vue','.svelte','.astro','.js','.mjs','.ts']);
// Pure-JS files get only JS-relevant rules — CSS-block heuristics false-positive on JS object literals
const PURE_JS_EXTS = new Set(['.js','.mjs','.ts']);
const SKIP_DIRS = new Set(['node_modules','dist','build','.git','.next','out']);

// ── Color helpers ─────────────────────────────────────────────────────────────

function rgbToHsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h = 0, s = 0;
  const l = (max + min) / 2;
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
    else if (max === g) h = ((b - r) / d + 2) / 6;
    else h = ((r - g) / d + 4) / 6;
  }
  return { h: h * 360, s, l, _rgb: { r, g, b } };
}

/** Returns {h, s, l, isOklch?, C?} or null */
function parseColor(tok) {
  tok = tok.trim().replace(/,$/, '');
  // oklch(L C H [/ alpha])
  let m = tok.match(/^oklch\(\s*([\d.]+%?)\s+([\d.]+)\s+([\d.]+)/i);
  if (m) {
    const L = parseFloat(m[1]) / (m[1].endsWith('%') ? 100 : 1);
    const C = parseFloat(m[2]);
    const H = parseFloat(m[3]);
    return { h: H, s: C > 0 ? 1 : 0, l: L, isOklch: true, C };
  }
  // hsl/hsla
  m = tok.match(/^hsla?\(\s*([\d.]+)(?:deg)?\s*[,\s]\s*([\d.]+)%?\s*[,\s]\s*([\d.]+)%?/i);
  if (m) return { h: parseFloat(m[1]), s: parseFloat(m[2]) / 100, l: parseFloat(m[3]) / 100 };
  // rgb/rgba
  m = tok.match(/^rgba?\(\s*([\d.]+)\s*[,\s]\s*([\d.]+)\s*[,\s]\s*([\d.]+)/i);
  if (m) return rgbToHsl(+m[1], +m[2], +m[3]);
  // #rrggbbaa / #rrggbb / #rgb / #rgba
  m = tok.match(/^#([0-9a-f]{3,8})$/i);
  if (m) {
    let h = m[1];
    if (h.length === 3 || h.length === 4) h = h.split('').map(c => c + c).join('').slice(0, 6);
    if (h.length > 6) h = h.slice(0, 6);
    if (h.length !== 6) return null;
    return rgbToHsl(parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16));
  }
  return null;
}

function isGray(c) {
  if (!c) return true;
  // ponytail: oklch chroma < 0.03 = achromatic; HSL saturation < 0.05
  if (c.isOklch) return (c.C ?? 0) < 0.03;
  return c.s < 0.05;
}

// Extract all color tokens from a gradient string
function extractGradientColors(str) {
  const re = /(#[0-9a-f]{3,8}|(?:oklch|rgba?|hsla?)\s*\([^)]+\))/gi;
  const colors = [];
  let m;
  while ((m = re.exec(str)) !== null) {
    const c = parseColor(m[1]);
    if (c) colors.push(c);
  }
  return colors;
}

// ── Line number helper ────────────────────────────────────────────────────────

function makeLineAt(text) {
  // ponytail: binary search over precomputed newline positions
  const starts = [0];
  for (let i = 0; i < text.length; i++) if (text[i] === '\n') starts.push(i + 1);
  return pos => {
    let lo = 0, hi = starts.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (starts[mid] <= pos) lo = mid; else hi = mid - 1;
    }
    return lo + 1;
  };
}

// ── CSS block extractor ───────────────────────────────────────────────────────

/**
 * Extracts leaf CSS rule blocks (those with no nested { }).
 * Returns [{selector, body, startLine}]
 */
function extractCSSBlocks(text) {
  const blocks = [];
  const lineAt = makeLineAt(text);
  const stack = []; // {selector, bodyStart}
  let i = 0, selectorStart = 0;

  while (i < text.length) {
    // Skip block comments
    if (text[i] === '/' && text[i+1] === '*') {
      const end = text.indexOf('*/', i + 2);
      i = end === -1 ? text.length : end + 2;
      continue;
    }
    // Skip strings
    if (text[i] === '"' || text[i] === "'") {
      const q = text[i++];
      while (i < text.length && text[i] !== q) { if (text[i] === '\\') i++; i++; }
      i++;
      continue;
    }
    if (text[i] === '{') {
      const sel = text.slice(selectorStart, i).replace(/\/\*[\s\S]*?\*\//g, '').trim().replace(/\s+/g, ' ');
      stack.push({ selector: sel, bodyStart: i + 1, startLine: lineAt(i + 1) });
      selectorStart = i + 1;
      i++;
      continue;
    }
    if (text[i] === '}') {
      const frame = stack.pop();
      if (frame) {
        const body = text.slice(frame.bodyStart, i);
        if (!/{/.test(body)) { // leaf block
          const fullSel = [...stack.map(f => f.selector), frame.selector].filter(Boolean).join(' ');
          blocks.push({ selector: fullSel || frame.selector, body, startLine: frame.startLine });
        }
      }
      selectorStart = i + 1;
      i++;
      continue;
    }
    i++;
  }
  return blocks;
}

// ── Suppression parser ────────────────────────────────────────────────────────

function parseSuppressions(text) {
  const lineAt = makeLineAt(text);
  const suppressions = new Map();
  const patterns = [
    /\/\*\s*auteur-allow:\s*(\w+)\s*--\s*([\s\S]*?)\*\//g,
    /\/\/\s*auteur-allow:\s*(\w+)\s*--(.*)/gm,
    /<!--\s*auteur-allow:\s*(\w+)\s*--\s*(.*?)-->/g,
  ];
  for (const re of patterns) {
    let m;
    while ((m = re.exec(text)) !== null) {
      const ruleId = m[1].trim();
      const reason = m[2].trim();
      const reasonOk = reason.replace(/\s+/g, '').length >= 10;
      const line = lineAt(m.index);
      if (!suppressions.has(ruleId)) suppressions.set(ruleId, { line, reasonOk, reason });
    }
  }
  return suppressions;
}

// ── FAIL rules ────────────────────────────────────────────────────────────────

function checkFontDefaultSlop(text, findings) {
  const lineAt = makeLineAt(text);
  // CSS font-family declarations
  const re = /font-family\s*:\s*([^\n;{}]+)/gi;
  let m;
  while ((m = re.exec(text)) !== null) {
    const first = m[1].split(',')[0].trim().replace(/['"]/g, '').trim();
    if (/^inter$/i.test(first) || /^space grotesk$/i.test(first)) {
      findings.push({ rule: 'FONT_DEFAULT_SLOP', severity: 'fail', line: lineAt(m.index),
        detail: `font-family first family is "${first}"` });
    }
  }
  // Tailwind fontFamily config: fontFamily: { key: ['Inter', ...] }
  const twRe = /fontFamily\s*:\s*\{[^}]*\}/gs;
  while ((m = twRe.exec(text)) !== null) {
    const block = m[0];
    const entryRe = /:\s*\[['"]([^'"]+)['"]/g;
    let em;
    while ((em = entryRe.exec(block)) !== null) {
      const first = em[1].trim();
      if (/^inter$/i.test(first) || /^space grotesk$/i.test(first)) {
        findings.push({ rule: 'FONT_DEFAULT_SLOP', severity: 'fail', line: lineAt(m.index),
          detail: `Tailwind fontFamily first entry is "${first}"` });
      }
    }
  }
}

function checkAiGradient(text, findings) {
  const lineAt = makeLineAt(text);
  const re = /(?:linear|radial|conic)-gradient\s*\(/gi;
  let m;
  while ((m = re.exec(text)) !== null) {
    // Extract balanced parens
    let depth = 0, j = m.index + m[0].length - 1;
    const start = j;
    while (j < text.length) {
      if (text[j] === '(') depth++;
      else if (text[j] === ')') { depth--; if (depth === 0) break; }
      j++;
    }
    const gradStr = text.slice(start, j + 1);
    const colors = extractGradientColors(gradStr);
    const aiColors = colors.filter(c => !isGray(c) && c.h >= 250 && c.h <= 290);
    if (aiColors.length >= 2) {
      findings.push({ rule: 'AI_GRADIENT', severity: 'fail', line: lineAt(m.index),
        detail: `${aiColors.length} gradient stops in hue 250–290 (purple-blue AI default)` });
    }
  }
}

function checkGradientText(blocks, findings) {
  const gradRe = /(?:linear|radial|conic)-gradient/i;
  const clipRe = /-webkit-background-clip\s*:\s*text|background-clip\s*:\s*text|-webkit-text-fill-color\s*:\s*transparent/i;
  for (const block of blocks) {
    if (gradRe.test(block.body) && clipRe.test(block.body)) {
      findings.push({ rule: 'GRADIENT_TEXT', severity: 'fail', line: block.startLine,
        detail: `gradient + background-clip:text (or -webkit-text-fill-color:transparent) in same block` });
    }
  }
}

function checkTransitionAll(text, findings) {
  const lineAt = makeLineAt(text);
  const re = /transition\s*:\s*all\b/gi;
  let m;
  while ((m = re.exec(text)) !== null) {
    findings.push({ rule: 'TRANSITION_ALL', severity: 'fail', line: lineAt(m.index),
      detail: `transition: all — use specific properties instead` });
  }
}

function checkRawScrollListener(text, findings) {
  const lineAt = makeLineAt(text);
  let m;
  const re = /addEventListener\s*\(\s*['"]scroll['"]/g;
  while ((m = re.exec(text)) !== null) {
    findings.push({ rule: 'RAW_SCROLL_LISTENER', severity: 'fail', line: lineAt(m.index),
      detail: `addEventListener('scroll') — use IntersectionObserver or animation-timeline` });
  }
  const re2 = /\bonscroll\s*=/g;
  while ((m = re2.exec(text)) !== null) {
    findings.push({ rule: 'RAW_SCROLL_LISTENER', severity: 'fail', line: lineAt(m.index),
      detail: `onscroll= attribute — use IntersectionObserver or animation-timeline` });
  }
}

// ── WARN rules ────────────────────────────────────────────────────────────────

function checkGlassCard(blocks, findings) {
  for (const block of blocks) {
    if (!/card|tile|panel/i.test(block.selector)) continue;
    if (/backdrop-filter\s*:\s*blur\s*\(/i.test(block.body)) {
      findings.push({ rule: 'GLASS_CARD', severity: 'warn', line: block.startLine,
        detail: `backdrop-filter:blur() in "${block.selector.slice(0, 60)}"` });
    }
  }
}

function checkCardCloneGrid(blocks, findings) {
  const getTriple = body => {
    const get = prop => { const m = body.match(new RegExp(prop + '\\s*:\\s*([^;\\n]+)')); return m ? m[1].trim() : null; };
    const br = get('border-radius'), p = get('padding'), bs = get('box-shadow');
    return br && p && bs ? `${br}|${p}|${bs}` : null;
  };
  const seen = new Map();
  for (const block of blocks) {
    const key = getTriple(block.body);
    if (!key) continue;
    if (!seen.has(key)) seen.set(key, []);
    seen.get(key).push(block.startLine);
  }
  for (const [, lines] of seen) {
    if (lines.length >= 4) {
      findings.push({ rule: 'CARD_CLONE_GRID', severity: 'warn', line: lines[0],
        detail: `${lines.length} blocks share identical border-radius/padding/box-shadow (lines ${lines.join(', ')})` });
    }
  }
}

// Ban #16 is "em-dash-HEAVY sentences" — a density property, not the presence of the glyph.
// Flagging every occurrence trains you to suppress the rule, which is how a linter loses its
// authority; and the old remediation ("use &mdash;") renders the identical character, so
// following it changed nothing the rule claimed to detect.
const PROSE_EXT = new Set(['.html', '.htm', '.md', '.jsx', '.tsx', '.vue', '.svelte']);
function checkEmDashCopy(text, ext, findings) {
  if (!PROSE_EXT.has(ext)) return;
  // Keep only what a visitor actually reads: no script/style, no comments, no <title>/<meta>.
  const prose = text
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<title[\s\S]*?<\/title>/gi, ' ')
    .replace(/<meta[^>]*>/gi, ' ')
    // Headings, terms and captions are LABELS, not sentences. "-200m — The Blue" is a dash doing
    // exactly the job a dash should do, and counting it made the rule argue against good typography.
    .replace(/<(h[1-6]|dt|summary|figcaption|legend|caption|th)\b[\s\S]*?<\/\1>/gi, ' ')
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/(^|\s)\/\/[^\n]*/g, ' ')
    .replace(/<[^>]+>/g, ' ');

  // Entity-encoded dashes render the identical glyph. Counting only the literal character meant the
  // rule was defeated by `&mdash;` — the exact dodge the comment above says the old advice suffered
  // from. Decode first, then count.
  const decoded = prose
    .replace(/&mdash;|&#8212;|&#[xX]2014;/g, '—')
    .replace(/&ndash;|&#8211;|&#[xX]2013;/g, '–')
    .replace(/&hellip;|&#8230;/g, '…')
    .replace(/&nbsp;|&#160;/g, ' ');
  const dashes = (decoded.match(/—/g) || []).length;
  if (dashes < 3) return;
  const sentences = (decoded.match(/[.!?]["')\]]?(\s|$)/g) || []).length || 1;
  const words = (decoded.match(/\S+/g) || []).length || 1;
  const perSentence = dashes / sentences;
  const per100w = (dashes / words) * 100;
  // One criterion, because the ban is about the dash replacing sentence structure: if more than
  // half your sentences carry an em dash, it is structural. A per-100-words test looked reasonable
  // and flagged prose with one dash every four sentences, which is just writing.
  if (perSentence < 0.5) return;

  const lineAt = makeLineAt(text);
  const first = text.indexOf('—');
  findings.push({
    rule: 'EM_DASH_COPY', severity: 'warn', line: lineAt(first < 0 ? 0 : first),
    detail: `${dashes} em dashes across ${sentences} sentences / ${words} words of visible copy (${perSentence.toFixed(2)}/sentence, ${per100w.toFixed(1)} per 100 words) — the dash is doing the work sentence structure should. Rewrite the densest ones as full sentences; an occasional em dash is fine.`,
  });
}

function checkEyebrowEverywhere(blocks, findings) {
  const matched = [];
  for (const block of blocks) {
    if (!/text-transform\s*:\s*uppercase/i.test(block.body)) continue;
    const lsM = block.body.match(/letter-spacing\s*:\s*([\d.]+)(em|rem)/i);
    if (!lsM || parseFloat(lsM[1]) < 0.05) continue;
    const fsM = block.body.match(/font-size\s*:\s*([\d.]+)(rem|px|em)/i);
    if (!fsM) continue;
    const fs = parseFloat(fsM[1]), unit = fsM[2].toLowerCase();
    const fsRem = unit === 'px' ? fs / 16 : fs;
    if (fsRem > 0.875) continue;
    matched.push(block.startLine);
  }
  if (matched.length > 3) {
    findings.push({ rule: 'EYEBROW_EVERYWHERE', severity: 'warn', line: matched[0],
      detail: `${matched.length} blocks with text-transform:uppercase + letter-spacing≥0.05em + font-size≤0.875rem` });
  }
}

function checkCreamDefault(blocks, findings) {
  for (const block of blocks) {
    if (!/^(?::root|body|html)\b/i.test(block.selector.trim())) continue;
    const bgM = block.body.match(/background(?:-color)?\s*:\s*([^\n;]+)/i);
    if (!bgM) continue;
    const val = bgM[1].trim().split(/\s+/)[0];
    const c = parseColor(val);
    if (!c) continue;

    let inBand = false;
    if (c.isOklch) {
      inBand = c.l >= 0.84 && c.l <= 0.97 && (c.C ?? 0) < 0.06 && c.h >= 40 && c.h <= 100;
    } else {
      // ponytail: sRGB→OKLCH approximation via HSL; check lightness, small chroma gap, warm hue
      // max-min distance in [0,1] maps roughly to OKLCH chroma < 0.06 for near-white colors
      const rgb = c._rgb;
      if (!rgb) continue;
      const dRgb = Math.max(rgb.r, rgb.g, rgb.b) - Math.min(rgb.r, rgb.g, rgb.b);
      inBand = c.l >= 0.84 && c.l <= 0.97 && dRgb > 0.002 && dRgb < 0.08 && c.h >= 30 && c.h <= 110;
    }
    if (inBand) {
      findings.push({ rule: 'CREAM_DEFAULT', severity: 'warn', line: block.startLine,
        detail: `body/:root background "${val}" is in warm-cream default band` });
    }
  }
}

// ── Tier-1 motion / WebGL / audio rules ─────────────────────────────────────────

function checkAutoplaySound(text, findings) {
  const lineAt = makeLineAt(text);
  const re = /<(audio|video)\b([^>]*)>/gi;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (/\bautoplay\b/i.test(m[2]) && !/\bmuted\b/i.test(m[2])) {
      findings.push({ rule: 'AUTOPLAY_SOUND', severity: 'fail', line: lineAt(m.index),
        detail: `<${m[1].toLowerCase()} autoplay> without muted — sound must start on a user gesture` });
    }
  }
}

function checkVideoNoPoster(text, findings) {
  const lineAt = makeLineAt(text);
  const re = /<video\b([^>]*)>/gi;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (!/\bposter\s*=/i.test(m[1])) {
      findings.push({ rule: 'VIDEO_NO_POSTER', severity: 'warn', line: lineAt(m.index),
        detail: `<video> without poster — blank hero until the clip decodes` });
    }
  }
}

function checkWebglNoReducedMotion(text, findings) {
  const lineAt = makeLineAt(text);
  const ctxRe = /new\s+THREE\.WebGLRenderer|getContext\s*\(\s*['"](?:webgl2?|webgpu)['"]|new\s+GPUDevice|new\s+OGLRenderer/;
  const cm = ctxRe.exec(text);
  if (!cm) return;
  if (!/prefers-reduced-motion/i.test(text)) {
    findings.push({ rule: 'WEBGL_NO_REDUCED_MOTION', severity: 'warn', line: lineAt(cm.index),
      detail: `WebGL/WebGPU scene with no prefers-reduced-motion branch — reduced-motion is an alternative art direction, not an afterthought` });
  }
}

function checkPointerNoRaf(text, findings) {
  const lineAt = makeLineAt(text);
  const pm = /addEventListener\s*\(\s*['"](?:pointermove|mousemove)['"]/.exec(text);
  if (!pm) return;
  if (!/requestAnimationFrame/.test(text)) {
    findings.push({ rule: 'POINTER_NO_RAF', severity: 'warn', line: lineAt(pm.index),
      detail: `pointermove/mousemove handler with no requestAnimationFrame — throttle DOM/uniform writes to rAF, never per-event` });
  }
}

// ── File scanner ──────────────────────────────────────────────────────────────

function scanFile(filePath) {
  const text = readFileSync(filePath, 'utf8');
  const ext = extname(filePath).toLowerCase();
  const suppressions = parseSuppressions(text);
  const findings = [];
  const blocks = extractCSSBlocks(text);

  if (PURE_JS_EXTS.has(ext)) {
    checkTransitionAll(text, findings);
    checkRawScrollListener(text, findings);
    checkWebglNoReducedMotion(text, findings);
    checkPointerNoRaf(text, findings);
  } else {
    checkFontDefaultSlop(text, findings);
    checkAiGradient(text, findings);
    checkGradientText(blocks, findings);
    checkTransitionAll(text, findings);
    checkRawScrollListener(text, findings);
    checkGlassCard(blocks, findings);
    checkCardCloneGrid(blocks, findings);
    checkEmDashCopy(text, ext, findings);
    checkEyebrowEverywhere(blocks, findings);
    checkCreamDefault(blocks, findings);
    checkAutoplaySound(text, findings);
    checkVideoNoPoster(text, findings);
    checkWebglNoReducedMotion(text, findings);
    checkPointerNoRaf(text, findings);
  }

  const result = findings.map(f => {
    const sup = suppressions.get(f.rule);
    return { ...f, suppressed: !!(sup && sup.reasonOk) };
  });

  // Invalid suppressions: present but reason too short
  for (const [ruleId, sup] of suppressions) {
    if (!sup.reasonOk) {
      result.push({ rule: 'SUPPRESSION_NO_REASON', severity: 'warn', line: sup.line,
        detail: `auteur-allow: ${ruleId} — reason too short (need 10+ non-whitespace chars)`, suppressed: false });
    }
  }

  return { path: filePath, findings: result };
}

// ── File walker ───────────────────────────────────────────────────────────────

function walk(entry) {
  const files = [];
  let stat;
  try { stat = statSync(entry); } catch { return files; }
  if (stat.isFile()) {
    if (SCAN_EXTS.has(extname(entry).toLowerCase())) files.push(entry);
    return files;
  }
  for (const name of readdirSync(entry)) {
    if (SKIP_DIRS.has(name)) continue;
    const full = join(entry, name);
    try {
      const s = statSync(full);
      if (s.isDirectory()) files.push(...walk(full));
      else if (SCAN_EXTS.has(extname(name).toLowerCase())) files.push(full);
    } catch { /* skip unreadable */ }
  }
  return files;
}

// ── Main ──────────────────────────────────────────────────────────────────────

function main() {
  const argv = process.argv.slice(2);
  const jsonMode = argv.includes('--json');
  const target = argv.find(a => !a.startsWith('-'));

  if (!target) {
    process.stderr.write('Usage: node slopscan.mjs <dir-or-file> [--json]\n');
    process.exit(2);
  }

  const rootDir = resolve(target);
  const files = walk(rootDir);
  const fileResults = files.map(f => scanFile(f));

  let totalFails = 0, totalWarns = 0, totalSuppressed = 0;
  for (const fr of fileResults) {
    for (const f of fr.findings) {
      if (f.suppressed) totalSuppressed++;
      else if (f.severity === 'fail') totalFails++;
      else totalWarns++;
    }
  }

  if (jsonMode) {
    process.stdout.write(JSON.stringify({ files: fileResults, summary: { fails: totalFails, warns: totalWarns, suppressed: totalSuppressed } }, null, 2) + '\n');
  } else {
    for (const fr of fileResults) {
      if (!fr.findings.length) continue;
      const rel = fr.path.replace(rootDir, '').replace(/^[\\/]/, '');
      for (const f of fr.findings) {
        const tag = f.suppressed ? 'SKIP' : f.severity.toUpperCase();
        process.stdout.write(`${tag} ${f.rule} ${rel}:${f.line} — ${f.detail}\n`);
      }
    }
    process.stdout.write(`\nSummary: ${totalFails} fails, ${totalWarns} warns, ${totalSuppressed} suppressed\n`);
  }

  process.exit(totalFails > 0 ? 1 : 0);
}

main();
