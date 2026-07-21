// HERMES//HUB service worker: instant loads + a usable shell when offline.
//
// Static shell  → network-first with cache fallback, so an online load always
//                 gets the latest build (installed PWAs can't hard-refresh)
// GET /api/*    → network-first with cache fallback, so the last known data
//                 (news, weather, worldstate…) still renders with no signal
// POST /api/*   → network only (agent, sync writes never come from cache)

const VERSION = "hub-v56";
const SHELL = [
  "/",
  "/css/dashboard.css",
  "/js/main.js", "/js/store.js", "/js/api.js", "/js/utils.js",
  "/js/viewer.js", "/js/actions.js", "/js/summarize.js", "/js/auth.js", "/js/sync.js",
  "/js/notifications.js", "/js/sources.js", "/js/reading.js", "/js/calendars.js", "/js/evolve.js",
  "/js/routing.js", "/js/detail.js", "/js/chart.js",
  "/js/widgets/clock.js", "/js/widgets/worldstate.js", "/js/widgets/agent.js",
  "/js/widgets/weather.js", "/js/widgets/launcher.js", "/js/widgets/news.js",
  "/js/widgets/tasks.js", "/js/widgets/notes.js", "/js/widgets/calendar.js",
  "/js/widgets/markets.js", "/js/widgets/scores.js", "/js/widgets/racing.js", "/js/widgets/socials.js", "/js/widgets/gaming.js", "/js/widgets/stocks.js", "/js/widgets/glance.js", "/js/widgets/worldclock.js", "/js/widgets/quakes.js", "/js/widgets/fx.js", "/js/widgets/convert.js", "/js/widgets/air.js", "/js/widgets/marine.js", "/js/widgets/space.js", "/js/widgets/alerts.js", "/js/widgets/flights.js", "/js/widgets/podcasts.js", "/js/widgets/medbot.js", "/js/widgets/pubmed.js", "/js/widgets/trials.js", "/js/widgets/drug.js", "/js/widgets/calc.js", "/js/widgets/meded.js", "/js/widgets/codelab.js", "/js/widgets/ailearn.js", "/js/widgets/snippets.js", "/js/widgets/repos.js", "/js/widgets/papers.js", "/js/widgets/ainews.js", "/js/widgets/aidaily.js", "/js/widgets/commodities.js", "/js/widgets/changelog.js", "/js/widgets/tracker.js", "/js/widgets/topicnews.js", "/js/widgets/focus.js", "/js/widgets/system.js",
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

  // Static shell: NETWORK-FIRST so an online load always gets the latest HTML/JS
  // (installed PWAs can't hard-refresh — stale-while-revalidate could otherwise
  // strand a user on an old build). Falls back to cache only when offline.
  event.respondWith(
    fetch(request)
      .then((response) => {
        if (response.ok) {
          const copy = response.clone();
          caches.open(VERSION).then((cache) => cache.put(request, copy));
        }
        return response;
      })
      .catch(() => caches.match(request)),
  );
});
