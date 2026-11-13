// The Webapp's notification worker, registered at the Webapp's base path by
// src/lib/browser-notifications.ts. Mobile browsers refuse `new Notification()`
// and show only what a service worker registration shows, so the Webapp shows
// its notifications through this worker's registration, and a click on one
// lands here.
//
// It has no fetch handler and caches nothing, on purpose: it never serves a
// page, so it can never keep an old renderer alive after `hermes update`.

/** What the tab a click brings forward receives; src/lib/browser-notifications.ts reads it. */
const FOCUS_SESSION_MESSAGE = 'hermes:focus-session'

// An update takes over at once: there is no page state to hand over.
self.addEventListener('install', () => self.skipWaiting())

self.addEventListener('notificationclick', event => {
  event.notification.close()
  event.waitUntil(openHermes(event.notification.data?.focusSessionId))
})

/** Whether *url* is a page of this Webapp: its base path, with or without the trailing slash. */
function isWebappPage(url) {
  const base = new URL(self.registration.scope).pathname.replace(/\/$/, '')
  const { pathname } = new URL(url)

  return pathname === base || pathname.startsWith(`${base}/`)
}

/** Bring the Webapp tab used last forward and have it open *sessionId*; with no tab open, open the Webapp at it. */
async function openHermes(sessionId) {
  const session = typeof sessionId === 'string' && sessionId ? sessionId : ''
  // Focused first, then by most recent focus. A tab opened before this worker registered is not
  // controlled by it, and still counts.
  const windows = await self.clients.matchAll({ includeUncontrolled: true, type: 'window' })
  const tab = windows.find(client => isWebappPage(client.url))

  if (!tab) {
    // The chat's route, as sessionRoute() in src/app/routes.ts builds it.
    await self.clients.openWindow(`${self.registration.scope}${session ? `#/${encodeURIComponent(session)}` : ''}`)

    return
  }

  // The message goes first: the tab opens the session while the browser brings it forward.
  if (session) {
    tab.postMessage({ sessionId: session, type: FOCUS_SESSION_MESSAGE })
  }

  await tab.focus()
}
