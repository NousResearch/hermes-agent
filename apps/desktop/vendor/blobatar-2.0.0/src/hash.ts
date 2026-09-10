/**
 * Seed hashing.
 *
 * Two guarantees this file exists to provide:
 *
 * 1. Avalanche — "alain" and "alaim" must produce visually unrelated blobatars.
 *    Plain FNV-1a does not give you this; the murmur3 finalizer does.
 * 2. Streaming — the seed is hashed once, then each trait key continues from
 *    that state. Trait values are therefore independent of one another, so
 *    adding a trait in a later version cannot disturb existing blobatars.
 */

const SEP = 0xff;

/** Mixes bytes into a 32-bit state. */
function feed(h: number, bytes: Uint8Array): number {
  for (let i = 0; i < bytes.length; i++) {
    h = Math.imul(h ^ bytes[i]!, 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return h;
}

/** murmur3 fmix32 — a bijection on uint32 with full avalanche. */
function finalize(h: number): number {
  h = Math.imul(h ^ (h >>> 16), 2246822507);
  h = Math.imul(h ^ (h >>> 13), 3266489909);
  return (h ^ (h >>> 16)) >>> 0;
}

const utf8 = new TextEncoder();

/**
 * Normalizes a seed so that inputs a human considers equal hash equally.
 *
 * NFC first, so precomposed "é" and decomposed "é" agree; then trim, then
 * lowercase. Without this, `Alain@x.com` and `alain@x.com` produce different
 * blobatars for the same person — which gets reported as a bug, every time.
 */
export function normalizeSeed(seed: string): string {
  return seed.normalize("NFC").trim().toLowerCase();
}

/**
 * Hashes the seed once into a reusable state. Non-ASCII seeds are encoded to
 * UTF-8 bytes first, so hashing is over codepoints rather than UTF-16 units
 * (surrogate pairs would otherwise hash inconsistently across engines).
 */
export function seedState(seed: string, normalize = true): number {
  const s = normalize ? normalizeSeed(seed) : seed;
  return feed(1779033703 ^ s.length, utf8.encode(s));
}

/** Derives one uniform float in [0, 1) for `key`, independent of every other key. */
export function stream(state: number, key: string): number {
  return finalize(feed(feed(state, Uint8Array.of(SEP)), utf8.encode(key))) / 4294967296;
}
