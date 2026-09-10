/**
 * Persistent lifecycle events for the WhatsApp bridge.
 *
 * The bridge's stdout is redirected by the gateway to a per-profile
 * `whatsapp/bridge.log` (human prose), which is great for debugging but hard
 * to consume programmatically. This module appends small structured JSONL
 * entries (`bridge_started` / `connected` / `disconnected` / `logged_out`)
 * next to the session dir so operators, health checks, and dashboards can
 * read the channel's live state without parsing prose logs.
 *
 * Design notes:
 * - Best-effort: never throws; a failed write must never take the bridge down.
 * - Bounded: rotates to `<file>.1` past ~5MB, so it can run unattended forever.
 * - Location: `<session>/../bridge-events.jsonl` by default (peer of the
 *   session dir and media caches), overridable via WHATSAPP_BRIDGE_EVENTS_FILE.
 */
import path from 'path';
import { mkdirSync, appendFileSync, statSync, renameSync } from 'fs';

export function resolveEventsFile(sessionDir) {
  return (
    process.env.WHATSAPP_BRIDGE_EVENTS_FILE ||
    path.join(sessionDir, '..', 'bridge-events.jsonl')
  );
}

const EVENTS_MAX_BYTES = 5 * 1024 * 1024;

export function recordBridgeEvent(sessionDir, event, extra = {}) {
  try {
    const file = resolveEventsFile(sessionDir);
    mkdirSync(path.dirname(file), { recursive: true });
    try {
      if (statSync(file).size > EVENTS_MAX_BYTES) renameSync(file, `${file}.1`);
    } catch {
      // file does not exist yet — fine
    }
    appendFileSync(file, `${JSON.stringify({ ts: new Date().toISOString(), event, ...extra })}\n`);
  } catch (err) {
    try { console.error('[bridge] event write failed:', err.message); } catch {}
  }
}
