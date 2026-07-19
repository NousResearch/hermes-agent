// Cross-device state sync.
//
// The server keeps one versioned copy of the dashboard state (SQLite).
// Strategy: localStorage stays the source of truth for the current tab;
// every local change is pushed (debounced) with the revision it was based
// on. A 409 or a background poll detecting a newer revision means another
// device wrote — we adopt the server state and re-render. Simple, robust,
// and good enough for a single-owner dashboard.

import { api, getToken } from "./api.js";
import { store } from "./store.js";
import { toast } from "./utils.js";

const REV_KEY = "hermesHub.syncRev";
const POLL_MS = 25_000;
const DEBOUNCE_MS = 900;    // quiet time before a push
const MAX_PENDING_MS = 1500; // …but never sit on changes longer than this

let enabled = false;
let syncRev = Number(localStorage.getItem(REV_KEY) || "0");
let applyingRemote = false;
let onRemoteState = () => {};

function setRev(rev) {
  syncRev = rev;
  localStorage.setItem(REV_KEY, String(rev));
}

function adopt(remote) {
  applyingRemote = true;
  try {
    store.replace(remote.state);
    setRev(remote.rev);
    onRemoteState();
  } finally {
    applyingRemote = false;
  }
}

let pushTimer = null;
let pendingSince = 0;

async function doPush() {
  pushTimer = null;
  pendingSince = 0;
  if (!enabled || applyingRemote) return;
  try {
    const { rev } = await api.statePut(store.state, syncRev);
    setRev(rev);
  } catch (err) {
    if (err.status === 409) {
      // Someone else (your phone?) wrote first — their version wins.
      const remote = await api.stateGet().catch(() => null);
      if (remote?.state) {
        adopt(remote);
        toast("Synced newer state from another device");
      }
    }
    // Other failures (offline, locked) are retried on the next change/poll.
  }
}

/** Debounced push with a max-pending cap, so a stream of rapid edits still
 *  syncs every couple of seconds instead of waiting for total silence. */
function push() {
  const now = Date.now();
  if (!pendingSince) pendingSince = now;
  clearTimeout(pushTimer);
  const wait = now - pendingSince >= MAX_PENDING_MS ? 0 : DEBOUNCE_MS;
  pushTimer = setTimeout(doPush, wait);
}

/** Unload-time flush. sendBeacon is the only transport Chromium reliably
 *  delivers during a same-tab navigation/close; the access code rides in the
 *  body because beacons cannot set headers. The next boot reconciles the
 *  revision (server rev > local syncRev → adopt, same content). */
function flushNow() {
  if (!enabled || !pendingSince) return;
  clearTimeout(pushTimer);
  pushTimer = null;
  pendingSince = 0;
  const token = getToken();
  const payload = JSON.stringify({
    state: store.state,
    baseRev: syncRev,
    ...(token ? { token } : {}),
  });
  const sent = navigator.sendBeacon?.(
    "/api/state", new Blob([payload], { type: "application/json" }),
  );
  if (!sent) {
    fetch("/api/state", {
      method: "POST",
      keepalive: true,
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: payload,
    }).catch(() => { /* page is closing; next boot reconciles */ });
  }
}

async function poll() {
  if (!enabled) return;
  try {
    const { rev } = await api.stateRev();
    if (rev > syncRev) {
      const remote = await api.stateGet();
      if (remote?.state) {
        adopt(remote);
        toast("Dashboard updated from another device");
      }
    }
  } catch { /* offline or locked; try again next tick */ }
}

/**
 * Start syncing. `rerender` is invoked after remote state is adopted.
 * Resolution on boot: server has state → adopt it unless our unsynced local
 * copy is the only data (fresh server) → then seed the server from local.
 */
let initialized = false;

export async function initSync(rerender) {
  onRemoteState = rerender;
  if (initialized) {
    // Re-entry after unlocking: reconcile now, and seed the server if it is
    // still empty (the locked boot couldn't).
    await poll();
    if (enabled) push();
    return enabled;
  }
  initialized = true;
  let remote;
  try {
    remote = await api.stateGet();
  } catch (err) {
    if (String(err.message).includes("sync is not enabled")) return false;
    // Locked or unreachable — sync will start working once unlocked.
    remote = null;
  }

  enabled = true;
  if (remote?.state && remote.rev > 0) {
    if (remote.rev !== syncRev) {
      adopt(remote);
    }
  } else if (remote) {
    // Fresh server: seed it with what this browser has.
    try {
      const { rev } = await api.statePut(store.state, null);
      setRev(rev);
    } catch { /* ignore; will retry on next change */ }
  }

  store.subscribe(() => {
    if (!applyingRemote) push();
  });
  setInterval(poll, POLL_MS);
  // Phones background tabs constantly: flush pending changes when the page
  // hides or unloads, reconcile when it comes back. A hidden-but-alive page
  // can use the normal async push (which tracks the new revision); a dying
  // page needs the beacon.
  window.addEventListener("pagehide", flushNow);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      if (pendingSince) { clearTimeout(pushTimer); doPush(); }
    } else {
      poll();
    }
  });
  return true;
}
