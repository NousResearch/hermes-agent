// Minimal service worker for PWA fixture shape (valid tiny content)
self.addEventListener('install', (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('fetch', (event) => {
  // Minimal pass-through for fixture; real PWA would cache
  event.respondWith(fetch(event.request));
});
