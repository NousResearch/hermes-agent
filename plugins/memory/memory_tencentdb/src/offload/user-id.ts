/**
 * User ID resolver for backend reporting.
 *
 * The backend `/offload/v1/store` endpoint keys state by `X-User-Id`.
 * If the plugin config does not provide one, we fall back to the host's
 * primary non-loopback IPv4 address so each machine still maps to a
 * stable identifier. Falls back further to `"unknown-host"` on failure.
 *
 * The resolved value is cached on first read; IP lookup is cheap but
 * callers invoke this per request so caching keeps the hot path clean.
 */
import * as os from "node:os";

let _cachedUserId: string | null = null;
let _cachedSource: "config" | "ip" | "fallback" | null = null;

/**
 * Find the first non-loopback, non-internal IPv4 address on the host.
 * Returns null when the host has no external-facing interface.
 */
function detectLocalIPv4(): string | null {
  try {
    const interfaces = os.networkInterfaces();
    for (const name of Object.keys(interfaces)) {
      const addrs = interfaces[name];
      if (!addrs) continue;
      for (const addr of addrs) {
        // node >= 18 exposes `family` as "IPv4" / "IPv6"; older versions use 4 / 6.
        const isV4 = addr.family === "IPv4" || (addr.family as unknown as number) === 4;
        if (isV4 && !addr.internal && typeof addr.address === "string") {
          return addr.address;
        }
      }
    }
  } catch {
    /* ignore — detection best-effort */
  }
  return null;
}

/**
 * Resolve the effective user ID. Priority:
 *   1. `configuredUserId` from plugin config (trimmed, non-empty)
 *   2. Primary non-loopback IPv4 address of the host
 *   3. Literal `"unknown-host"` fallback
 *
 * Result and source are cached — subsequent calls are O(1).
 */
export function resolveUserId(configuredUserId?: string | null): string {
  if (_cachedUserId) return _cachedUserId;

  const trimmed = typeof configuredUserId === "string" ? configuredUserId.trim() : "";
  if (trimmed) {
    _cachedUserId = trimmed;
    _cachedSource = "config";
    return _cachedUserId;
  }

  const ip = detectLocalIPv4();
  if (ip) {
    _cachedUserId = ip;
    _cachedSource = "ip";
    return _cachedUserId;
  }

  _cachedUserId = "unknown-host";
  _cachedSource = "fallback";
  return _cachedUserId;
}

/** Returns how the currently-cached user id was resolved (or null if unresolved). */
export function getUserIdSource(): "config" | "ip" | "fallback" | null {
  return _cachedSource;
}

/** Testing hook: wipe the cache so the next resolve() re-evaluates. */
export function _resetUserIdCacheForTests(): void {
  _cachedUserId = null;
  _cachedSource = null;
}
