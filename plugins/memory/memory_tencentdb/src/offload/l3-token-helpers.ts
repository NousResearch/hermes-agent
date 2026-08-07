/**
 * Heuristic token estimate (中文/1.7 + 非中文/4) when tiktoken is disabled or fails.
 */

function countCjkChars(text: string): number {
  let n = 0;
  for (const ch of text) {
    const c = ch.codePointAt(0)!;
    if (
      (c >= 0x4e00 && c <= 0x9fff) ||
      (c >= 0x3400 && c <= 0x4dbf) ||
      (c >= 0xf900 && c <= 0xfaff)
    ) {
      n++;
    }
  }
  return n;
}

export function estimateL3MixedTokensHeuristic(text: string): number {
  const cjk = countCjkChars(text);
  const rest = Math.max(0, text.length - cjk);
  return Math.ceil(cjk / 1.7 + rest / 4);
}
