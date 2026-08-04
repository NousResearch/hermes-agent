/**
 * Shared text utility functions for the memory-tdai plugin.
 */

/**
 * Extract meaningful words from text (supports CJK and Latin).
 *
 * Used by both auto-recall (keyword search) and l1-dedup (keyword candidate recall).
 * Extracted to a shared module to prevent implementation drift.
 */
export function extractWords(text: string): Set<string> {
  const words = new Set<string>();

  // Latin words (2+ chars)
  const latinWords = text.toLowerCase().match(/[a-z0-9]{2,}/g);
  if (latinWords) {
    for (const w of latinWords) words.add(w);
  }

  // CJK characters (each char as a "word", plus 2-gram)
  const cjkChars = text.match(/[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]/g);
  if (cjkChars) {
    for (const c of cjkChars) words.add(c);
    // 2-grams for better matching
    for (let i = 0; i < cjkChars.length - 1; i++) {
      words.add(cjkChars[i] + cjkChars[i + 1]);
    }
  }

  return words;
}
