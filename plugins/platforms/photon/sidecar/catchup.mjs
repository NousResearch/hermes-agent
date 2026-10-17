// Durable inbound catch-up for the Photon sidecar.
//
// spectrum-ts replays missed events after an in-process reconnect, but its
// resume cursor lives in memory only, so every new sidecar process (a gateway
// restart, a degraded-stream exit 75, a watchdog respawn) starts live-only and
// anything sent while no sidecar ran is lost. This module keeps the cursor on
// disk and hands it back to spectrum's own catch-up through the hook that
// patch-spectrum-resume-cursor.mjs installs:
//
//   - a cursor is committed only after every message it covers has been
//     written to the gateway (or deliberately skipped), so a crash replays
//     rather than drops;
//   - the ids of the last delivered messages are kept, so a replay never hands
//     the gateway the same message twice;
//   - when Photon refuses the saved cursor (its log no longer reaches back that
//     far) the gap is reported through `onGap` instead of passing silently.
//
// No SDK and no timers here, so tests can execute it under node — see
// tests/plugins/platforms/photon/test_catchup_cursor.py.

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

export const RESUME_HOOK = "__hermesPhotonResumeCursor";
const DELIVERED_IDS = 256;
// An item the main loop never sees (dropped between spectrum and the loop)
// stops holding back later cursor-only advances after this long.
const PENDING_TTL_MS = 2 * 60 * 1000;
const CURSOR_RE = /^\d+$/;

/** True when PHOTON_CATCHUP asks for the stock, live-only behavior. */
export function catchUpDisabled(value) {
  return /^(0|false|no|off)$/i.test(String(value || "").trim());
}

// The ids the main loop will see for one emitted item: a group (text plus a
// photo) reaches it as its children, one by one.
function messageIds(values) {
  const ids = [];
  for (const value of values || []) {
    const parts =
      value?.content?.type === "group" && Array.isArray(value.content.items)
        ? value.content.items
        : [value];
    for (const part of parts) {
      if (typeof part?.id === "string" && part.id) ids.push(part.id);
    }
  }
  return ids;
}

export function createCatchUp({ statePath, projectId = "", onGap = () => {}, log = console.error, now = Date.now }) {
  const enabled = Boolean(statePath);
  const project = crypto.createHash("sha256").update(String(projectId)).digest("hex").slice(0, 16);
  const state = {
    cursors: {}, // stream label -> newest cursor whose messages all reached the gateway
    delivered: [], // ids of the last messages written to the gateway, oldest first
    lastDeliveredAt: null, // send time of the newest delivered message
  };
  const stats = { patch: enabled ? "unknown" : "disabled", hooked: false, resumedFrom: {}, replaysSkipped: 0, gaps: 0, saveError: null };
  const pending = new Map(); // message id -> {label, cursor, remaining, at}
  const open = new Map(); // label -> emitted items not yet settled
  const deferred = new Map(); // label -> newest cursor-only advance held back by open items

  function load() {
    if (!enabled) return;
    try {
      const raw = JSON.parse(fs.readFileSync(statePath, "utf8"));
      // A cursor from another Photon project would skip this one's events.
      if (raw?.project !== project) return;
      for (const [label, cursor] of Object.entries(raw.cursors || {})) {
        if (typeof cursor === "string" && CURSOR_RE.test(cursor)) state.cursors[label] = cursor;
      }
      if (Array.isArray(raw.delivered)) {
        state.delivered = raw.delivered.filter((id) => typeof id === "string" && id).slice(-DELIVERED_IDS);
      }
      if (typeof raw.lastDeliveredAt === "string") state.lastDeliveredAt = raw.lastDeliveredAt;
    } catch (e) {
      if (e?.code !== "ENOENT") log(`photon-sidecar: catch-up state unreadable; starting live-only: ${e?.message || e}`);
    }
  }

  function save() {
    const tmp = `${statePath}.${process.pid}.tmp`;
    try {
      fs.mkdirSync(path.dirname(statePath), { recursive: true, mode: 0o700 });
      const fd = fs.openSync(tmp, "w", 0o600);
      try {
        fs.writeSync(fd, JSON.stringify({ version: 1, project, ...state, savedAt: new Date(now()).toISOString() }));
        fs.fsyncSync(fd);
      } finally {
        fs.closeSync(fd);
      }
      fs.renameSync(tmp, statePath);
      stats.saveError = null;
    } catch (e) {
      stats.saveError = e?.code || "save_failed";
      try {
        fs.unlinkSync(tmp);
      } catch {
        /* ignore */
      }
      log(`photon-sidecar: catch-up state not saved: ${e?.message || e}`);
    }
  }

  function commit(label, cursor) {
    if (!label || !CURSOR_RE.test(cursor)) return false;
    const current = state.cursors[label];
    if (current !== undefined && Number(cursor) <= Number(current)) return false;
    state.cursors[label] = cursor;
    return true;
  }

  // Returns true when the cursor moved (the caller saves).
  function close(entry, { commitOwn }) {
    const left = (open.get(entry.label) || 1) - 1;
    let moved = commitOwn && commit(entry.label, entry.cursor);
    if (left > 0) {
      open.set(entry.label, left);
      return moved;
    }
    open.delete(entry.label);
    const held = deferred.get(entry.label);
    deferred.delete(entry.label);
    if (held !== undefined && commit(entry.label, held)) moved = true;
    return moved;
  }

  function prune(at) {
    const expired = new Set();
    for (const [id, entry] of pending) {
      if (at - entry.at > PENDING_TTL_MS) {
        pending.delete(id);
        expired.add(entry);
      }
    }
    // An expired item's own cursor is never committed.
    let moved = false;
    for (const entry of expired) moved = close(entry, { commitOwn: false }) || moved;
    if (moved) save();
  }

  // The object spectrum's resumableOrderedStream calls (see the patch module).
  const hook = {
    initial(label) {
      stats.hooked = true;
      if (!enabled || !label) return undefined;
      const cursor = state.cursors[label];
      if (cursor !== undefined) {
        stats.resumedFrom[label] = cursor;
        log(`photon-sidecar: catch-up resumes ${label} after cursor ${cursor}`);
      }
      return cursor;
    },
    note(label, item) {
      if (!enabled || !label || item?.cursor === undefined) return;
      const cursor = String(item.cursor);
      const ids = messageIds(item.values);
      if (ids.length === 0) {
        // Nothing for the gateway (our own send, an unmappable event, the end
        // of a replay): advance now, or once everything earlier has settled.
        if (open.get(label) > 0) {
          const held = deferred.get(label);
          if (held === undefined || Number(cursor) > Number(held)) deferred.set(label, cursor);
        } else if (commit(label, cursor)) {
          save();
        }
        return;
      }
      const entry = { label, cursor, remaining: ids.length, at: now() };
      for (const id of ids) pending.set(id, entry);
      open.set(label, (open.get(label) || 0) + 1);
      prune(entry.at);
    },
    rejected(label) {
      if (!enabled) return;
      stats.gaps += 1;
      const since = state.lastDeliveredAt;
      if (label) delete state.cursors[label];
      for (const [id, entry] of pending) if (entry.label === label) pending.delete(id);
      open.delete(label);
      deferred.delete(label);
      save();
      log(`photon-sidecar: Photon refused the saved catch-up cursor for ${label}; messages sent since ${since || "the last delivery"} may be missing`);
      if (String(label).includes(".messages")) onGap({ since, until: new Date(now()).toISOString() });
    },
  };

  /** True for a replay of a message already written to the gateway; counts it. */
  function isReplay(message) {
    const id = message?.id;
    if (!enabled || typeof id !== "string" || !state.delivered.includes(id)) return false;
    stats.replaysSkipped += 1;
    return true;
  }

  /** The main loop is done with `message`: written to the gateway, or skipped. */
  function settle(message, { delivered }) {
    const id = typeof message?.id === "string" ? message.id : null;
    if (!enabled || !id) return;
    let dirty = false;
    if (delivered) {
      state.delivered.push(id);
      if (state.delivered.length > DELIVERED_IDS) state.delivered.splice(0, state.delivered.length - DELIVERED_IDS);
      const ts = message.timestamp;
      const sentAt = ts instanceof Date ? ts.toISOString() : ts ? String(ts) : null;
      // A replayed older message never moves "since" back for a later gap.
      if (sentAt && !(Date.parse(state.lastDeliveredAt) > Date.parse(sentAt))) state.lastDeliveredAt = sentAt;
      dirty = true;
    }
    const entry = pending.get(id);
    if (entry) {
      pending.delete(id);
      entry.remaining -= 1;
      if (entry.remaining <= 0 && close(entry, { commitOwn: true })) dirty = true;
    }
    if (dirty) save();
  }

  function snapshot() {
    return {
      enabled,
      ...stats,
      cursors: { ...state.cursors },
      resumedFrom: { ...stats.resumedFrom },
      lastDeliveredAt: state.lastDeliveredAt,
      inFlight: pending.size,
    };
  }

  return { enabled, hook, load, isReplay, settle, snapshot, setPatch: (status) => { stats.patch = status; } };
}
