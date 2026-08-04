/**
 * Fast token estimator — TypeScript port of token_count/fast_token_estimate.py
 * Targets cl100k_base encoding (GPT-4, Claude, DeepSeek, GLM, MiniMax).
 *
 * Precision: ~2-7% error for most languages (tested vs tiktoken cl100k_base).
 * Speed: ~5ms per 100K chars (vs tiktoken ~3-10s).
 *
 * Algorithm: single-pass character classification with per-category coefficients.
 * No BPE encoding, no regex splitting — pure arithmetic on codepoints.
 */
import { readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

// ─── CJK Lookup Table ──────────────────────────────────────────────────────
// Each byte = token cost × 255 for one CJK character (U+4E00..U+9FFF).
// Pre-computed from tiktoken cl100k_base actual encoding.
const CJK_START = 0x4E00;
const CJK_END = 0x9FFF;
let _cjkTable: Uint8Array | null = null;

function loadCjkTable(): Uint8Array | null {
  if (_cjkTable) return _cjkTable;
  try {
    // Try multiple paths for the CJK table
    const paths = [
      join(dirname(fileURLToPath(import.meta.url)), "../../token_count/cjk_token_table.bin"),
      join(dirname(fileURLToPath(import.meta.url)), "../../../token_count/cjk_token_table.bin"),
    ];
    for (const p of paths) {
      try {
        const buf = readFileSync(p);
        if (buf.length === CJK_END - CJK_START + 1) {
          _cjkTable = new Uint8Array(buf.buffer, buf.byteOffset, buf.length);
          return _cjkTable;
        }
      } catch { /* try next */ }
    }
  } catch { /* ignore */ }
  return null;
}

// ─── Character Classification ──────────────────────────────────────────────

function isLatinLetter(cp: number): boolean {
  return (
    (cp >= 0x41 && cp <= 0x5A) || (cp >= 0x61 && cp <= 0x7A) ||
    (cp >= 0x00C0 && cp <= 0x00FF && cp !== 0x00D7 && cp !== 0x00F7) ||
    (cp >= 0x0100 && cp <= 0x024F)
  );
}

function isCjkHan(cp: number): boolean {
  return (
    (cp >= 0x4E00 && cp <= 0x9FFF) ||
    (cp >= 0x3400 && cp <= 0x4DBF) ||
    (cp >= 0xF900 && cp <= 0xFAFF)
  );
}

function isKana(cp: number): boolean {
  return (cp >= 0x3040 && cp <= 0x309F) || (cp >= 0x30A0 && cp <= 0x30FF);
}

function isHangul(cp: number): boolean {
  return (cp >= 0xAC00 && cp <= 0xD7AF) || (cp >= 0x1100 && cp <= 0x11FF) || (cp >= 0x3130 && cp <= 0x318F);
}

function isCyrillic(cp: number): boolean {
  return (cp >= 0x0400 && cp <= 0x04FF) || (cp >= 0x0500 && cp <= 0x052F);
}

function isArabic(cp: number): boolean {
  return (
    (cp >= 0x0600 && cp <= 0x06FF) || (cp >= 0x0750 && cp <= 0x077F) ||
    (cp >= 0x08A0 && cp <= 0x08FF) || (cp >= 0xFB50 && cp <= 0xFDFF) ||
    (cp >= 0xFE70 && cp <= 0xFEFF)
  );
}

function isGreek(cp: number): boolean {
  return (cp >= 0x0370 && cp <= 0x03FF) || (cp >= 0x1F00 && cp <= 0x1FFF);
}

// ─── Main Estimator ────────────────────────────────────────────────────────

/**
 * Estimate token count for a string without doing BPE encoding.
 * Targets cl100k_base (GPT-4/Claude/DeepSeek/GLM/MiniMax).
 * Error typically <5% for code/English, <10% for CJK/mixed.
 */
export function fastEstimateTokens(text: string): number {
  if (!text) return 0;

  const n = text.length;
  const cjkTable = loadCjkTable();
  let tokens = 0.0;
  let i = 0;

  // Pre-scan: detect if text is non-English Latin (French, Spanish, etc.)
  let accentCount = 0;
  let latinCount = 0;
  const sampleEnd = Math.min(n, 50000);
  for (let s = 0; s < sampleEnd; s++) {
    const cp = text.charCodeAt(s);
    if (cp >= 0x80 && cp <= 0x024F &&
        ((cp >= 0x00C0 && cp <= 0x00FF && cp !== 0x00D7 && cp !== 0x00F7) || (cp >= 0x0100 && cp <= 0x024F))) {
      accentCount++;
    }
    if ((cp >= 0x41 && cp <= 0x5A) || (cp >= 0x61 && cp <= 0x7A)) {
      latinCount++;
    }
  }
  const isNonEnglishLatin = latinCount > 100 && accentCount > latinCount * 0.005;

  while (i < n) {
    const cp = text.charCodeAt(i);

    // ── Latin word ──
    if (isLatinLetter(cp)) {
      let j = i + 1;
      while (j < n) {
        const c = text.charCodeAt(j);
        if (isLatinLetter(c)) { j++; }
        else if (c === 0x27 && j + 1 < n && isLatinLetter(text.charCodeAt(j + 1))) { j += 2; }
        else { break; }
      }
      const wl = j - i;

      // Check if this word contains accented characters
      let hasAccent = false;
      if (isNonEnglishLatin) {
        for (let k = i; k < j; k++) {
          if (text.charCodeAt(k) >= 0x80) { hasAccent = true; break; }
        }
        if (!hasAccent) {
          // Check nearby window
          const lo = Math.max(0, i - 100);
          const hi = Math.min(n, j + 100);
          for (let k = lo; k < hi; k++) {
            const cc = text.charCodeAt(k);
            if (cc >= 0x00C0 && cc <= 0x024F && cc !== 0x00D7 && cc !== 0x00F7) {
              hasAccent = true; break;
            }
          }
        }
      }

      if (hasAccent) {
        // Non-English Latin words are longer in tokens
        if (wl <= 3) tokens += 1.0;
        else if (wl <= 5) tokens += 1.35;
        else if (wl <= 7) tokens += 1.85;
        else if (wl <= 9) tokens += 2.5;
        else if (wl <= 12) tokens += 3.2;
        else tokens += 3.2 + (wl - 12) * 0.32;
      } else {
        // English word
        if (wl <= 4) tokens += 1.0;
        else if (wl <= 8) tokens += 1.1;
        else if (wl <= 13) tokens += 1.5;
        else tokens += 1.5 + (wl - 13) * 0.3;
      }
      i = j;
      continue;
    }

    // ── CJK Han characters ──
    if (isCjkHan(cp)) {
      let j = i + 1;
      let segTokens = 0.0;
      if (cjkTable && cp >= CJK_START && cp <= CJK_END) {
        segTokens += cjkTable[cp - CJK_START]; // table values are direct token counts (1-3)
      } else {
        segTokens += 1.3;
      }
      while (j < n && isCjkHan(text.charCodeAt(j))) {
        const cp2 = text.charCodeAt(j);
        if (cjkTable && cp2 >= CJK_START && cp2 <= CJK_END) {
          segTokens += cjkTable[cp2 - CJK_START];
        } else {
          segTokens += 1.3;
        }
        j++;
      }
      const run = j - i;
      // BPE merges adjacent CJK characters. Longer segments get more merges.
      if (run >= 4) segTokens *= 0.94;
      else if (run >= 2) segTokens *= 0.97;
      tokens += segTokens;
      i = j;
      continue;
    }

    // ── Japanese Kana ──
    if (isKana(cp)) {
      let j = i + 1;
      while (j < n && isKana(text.charCodeAt(j))) j++;
      const run = j - i;
      if (run === 1) tokens += 1.0;
      else if (run === 2) tokens += 1.6;
      else if (run === 3) tokens += 2.65;
      else if (run === 4) tokens += 3.7;
      else if (run <= 6) tokens += run * 0.93;
      else tokens += run * 0.95;
      i = j;
      continue;
    }

    // ── Korean Hangul ──
    if (isHangul(cp)) {
      tokens += 1.4;
      i++;
      continue;
    }

    // ── Cyrillic (Russian etc.) ──
    if (isCyrillic(cp)) {
      let j = i + 1;
      while (j < n && isCyrillic(text.charCodeAt(j))) j++;
      tokens += (j - i) * 0.55;
      i = j;
      continue;
    }

    // ── Arabic ──
    if (isArabic(cp)) {
      let j = i + 1;
      while (j < n && isArabic(text.charCodeAt(j))) j++;
      tokens += (j - i) * 0.82;
      i = j;
      continue;
    }

    // ── Greek ──
    if (isGreek(cp)) {
      let j = i + 1;
      while (j < n && isGreek(text.charCodeAt(j))) j++;
      tokens += (j - i) * 0.85;
      i = j;
      continue;
    }

    // ── Digits (with commas, dots) ──
    if (cp >= 0x30 && cp <= 0x39) {
      let j = i + 1;
      let digits = 1;
      let commas = 0;
      let dots = 0;
      while (j < n) {
        const c = text.charCodeAt(j);
        if (c >= 0x30 && c <= 0x39) { digits++; j++; }
        else if (c === 0x2C && j + 1 < n && text.charCodeAt(j + 1) >= 0x30 && text.charCodeAt(j + 1) <= 0x39) {
          commas++; j += 2; digits++;
        }
        else if (c === 0x2E && j + 1 < n && text.charCodeAt(j + 1) >= 0x30 && text.charCodeAt(j + 1) <= 0x39) {
          dots++; j += 2; digits++;
        }
        else { break; }
      }
      if (digits <= 3 && commas === 0 && dots === 0) tokens += 1.0;
      else if (commas > 0) tokens += commas * 2 + 1.0;
      else if (dots > 0) tokens += Math.max(2.0, digits / 3.0 + dots * 1.5);
      else tokens += Math.max(1.0, digits / 2.5);
      i = j;
      continue;
    }

    // ── Whitespace (space, tab) ──
    if (cp === 0x20 || cp === 0x09) { i++; continue; }

    // ── Newline ──
    if (cp === 0x0A || cp === 0x0D) { tokens += 1.0; i++; continue; }

    // ── Fullwidth punctuation ──
    if ((cp >= 0x3000 && cp <= 0x303F) || (cp >= 0xFF00 && cp <= 0xFFEF) ||
        cp === 0x2018 || cp === 0x2019 || cp === 0x201C || cp === 0x201D ||
        cp === 0x2014 || cp === 0x2026 || cp === 0x2013) {
      tokens += 1.0; i++; continue;
    }

    // ── ASCII punctuation ──
    if (cp >= 0x21 && cp <= 0x7E) { tokens += 0.6; i++; continue; }

    // ── Other Unicode (emoji etc.) ──
    if (cp > 0x7F) { tokens += 2.5; i++; continue; }

    i++;
  }

  return Math.max(1, Math.round(tokens));
}

/**
 * Estimate tokens for an array of messages (same as buildTiktokenContextSnapshot
 * but using fast estimation instead of tiktoken).
 */
export function fastEstimateMessages(messages: any[], jsonReplacer?: (key: string, value: unknown) => unknown): number {
  let total = 0;
  for (const msg of messages) {
    const str = JSON.stringify(msg, jsonReplacer as any);
    total += fastEstimateTokens(str);
  }
  // JSON array overhead
  total += Math.ceil(messages.length * 0.5);
  return total;
}
