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
/**
 * Normalizes a seed so that inputs a human considers equal hash equally.
 *
 * NFC first, so precomposed "é" and decomposed "é" agree; then trim, then
 * lowercase. Without this, `Alain@x.com` and `alain@x.com` produce different
 * blobatars for the same person — which gets reported as a bug, every time.
 */
export declare function normalizeSeed(seed: string): string;
/**
 * Hashes the seed once into a reusable state. Non-ASCII seeds are encoded to
 * UTF-8 bytes first, so hashing is over codepoints rather than UTF-16 units
 * (surrogate pairs would otherwise hash inconsistently across engines).
 */
export declare function seedState(seed: string, normalize?: boolean): number;
/** Derives one uniform float in [0, 1) for `key`, independent of every other key. */
export declare function stream(state: number, key: string): number;
//# sourceMappingURL=hash.d.ts.map