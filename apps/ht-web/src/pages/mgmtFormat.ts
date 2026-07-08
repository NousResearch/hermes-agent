// Pure formatting helpers shared by the ported management pages (Files,
// Channels, Analytics). Kept dependency-free and side-effect-free so they are
// trivially unit-testable.

/** Human-readable byte size, e.g. 2048 -> "2.0 KB". `null` renders as "—". */
export function fmtBytes(n: number | null | undefined): string {
  if (n == null) return "—";
  if (n < 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u++;
  }
  return `${v.toFixed(v < 10 && u > 0 ? 1 : 0)} ${units[u]}`;
}

/** Compact token/count formatting: 1234 -> "1.2K", 3_400_000 -> "3.4M". */
export function fmtNumber(n: number | null | undefined): string {
  if (n == null) return "—";
  const abs = Math.abs(n);
  if (abs < 1000) return String(n);
  if (abs < 1_000_000) return `${(n / 1000).toFixed(1)}K`;
  if (abs < 1_000_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  return `${(n / 1_000_000_000).toFixed(1)}B`;
}

/** USD cost with sensible precision: <$1 shows 4 dp, otherwise 2 dp. */
export function fmtCost(n: number | null | undefined): string {
  if (n == null) return "—";
  const digits = Math.abs(n) > 0 && Math.abs(n) < 1 ? 4 : 2;
  return `$${n.toFixed(digits)}`;
}

/** Relative time from a unix-epoch-seconds timestamp. 0/falsy -> "—". */
export function relTime(epochSeconds: number | null | undefined): string {
  if (!epochSeconds) return "—";
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - epochSeconds));
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}
