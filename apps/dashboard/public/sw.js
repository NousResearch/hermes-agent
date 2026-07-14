// HERMES//HUB service worker: instant loads + a usable shell when offline.
//
// Static shell  → cache-first, refreshed in the background (stale-while-revalidate)
// GET /api/*    → network-first with cache fallback, so the last known data
//                 (news, weather, worldstate…) still renders with no signal
// POST /api/*   → network only (agent, sync writes never come from cache)

const VERSION = "hub-v3";
const SHELL = [
  "/",
  "/css/dashboard.css",
  "/js/main.js", "/js/store.js", "/js/api.js", "/js/utils.js",
  "/js/viewer.js", "/js/actions.js", "/js/summarize.js", "/js/auth.js", "/js/sync.js",
  "/js/notifications.js", "/js/sources.js",
  "/js/widgets/clock.js", "/js/widgets/worldstate.js", "/js/widgets/agent.js",
  "/js/widgets/weather.js", "/js/widgets/launcher.js", "/js/widgets/news.js",
  "/js/widgets/tasks.js", "/js/widgets/notes.js", "/js/widgets/calendar.js",
  "/js/widgets/markets.js",
  "/manifest.webmanifest",
  "/icons/icon-192.png", "/icons/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(VERSION).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== location.origin) return;

  // Sync state must always be authoritative — never intercepted, never cached.
  if (url.pathname.startsWith("/api/state")) return;

  if (url.pathname.startsWith("/api/")) {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const copy = response.clone();
          if (response.ok) {
            caches.open(VERSION).then((cache) => cache.put(request, copy));
          }
          return response;
        })
        .catch(() => caches.match(request).then((hit) => hit
          || new Response(JSON.stringify({ error: "offline and no cached data" }),
            { status: 503, headers: { "Content-Type": "application/json" } }))),
    );
    return;
  }

  // Static shell: serve from cache, refresh in the background.
  event.respondWith(
    caches.match(request).then((hit) => {
      const refresh = fetch(request)
        .then((response) => {
          if (response.ok) {
            const copy = response.clone();
            caches.open(VERSION).then((cache) => cache.put(request, copy));
          }
          return response;
        })
        .catch(() => hit);
      return hit || refresh;
    }),
  );
});
