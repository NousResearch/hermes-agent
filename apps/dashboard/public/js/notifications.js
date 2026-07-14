// Client side of the automations engine: polls the server for fired
// notifications, shows them as toasts, and mirrors them to the system
// notification tray when the user has granted permission.

import { api } from "./api.js";
import { toast } from "./utils.js";

const SEEN_KEY = "hermesHub.notifSeen";
const POLL_MS = 30_000;

let lastSeen = Number(localStorage.getItem(SEEN_KEY) || "0");
let started = false;

async function check() {
  let payload;
  try {
    payload = await api.notifications(lastSeen);
  } catch {
    return; // locked or offline — try again next tick
  }
  for (const note of payload.notifications) {
    toast(`${note.title}`);
    if ("Notification" in window && Notification.permission === "granted") {
      try {
        new Notification(note.title, {
          body: note.body.slice(0, 400),
          icon: "/icons/icon-192.png",
          tag: `hub-${note.id}`,
        });
      } catch { /* some platforms restrict constructor notifications */ }
    }
    window.dispatchEvent(new CustomEvent("hub:notification", { detail: note }));
  }
  if (payload.last > lastSeen) {
    lastSeen = payload.last;
    localStorage.setItem(SEEN_KEY, String(lastSeen));
  }
}

export function initNotifications() {
  if (started) return;
  started = true;
  // On the very first run, don't replay the whole backlog as toasts.
  if (!localStorage.getItem(SEEN_KEY)) {
    api.notifications(0).then((p) => {
      lastSeen = p.last;
      localStorage.setItem(SEEN_KEY, String(lastSeen));
    }).catch(() => {});
  } else {
    check();
  }
  setInterval(check, POLL_MS);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) check();
  });
}

/** Ask for system-notification permission (must run in a user gesture). */
export async function enableSystemAlerts() {
  if (!("Notification" in window)) return "unsupported";
  if (Notification.permission === "granted") return "granted";
  return Notification.requestPermission();
}
