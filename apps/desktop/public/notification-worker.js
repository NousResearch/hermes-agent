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
/** Asks a tab for the id it puts in the notifications it shows; it answers on the port sent with it. */
const TAB_QUERY = 'hermes:tab-query'
/** How long a tab gets to answer. Firefox lets a click focus a window for one second only, and a tab that
 *  is awake answers within milliseconds. */
const TAB_ANSWER_MS = 250

// An update takes over at once: there is no page state to hand over.
self.addEventListener('install', () => self.skipWaiting())

self.addEventListener('notificationclick', event => {
  event.notification.close()
  event.waitUntil(openHermes(event.notification.data ?? {}))
})

/** Whether *url* is a Webapp tab: its base path, with or without the trailing slash, or its index.html. The
 *  app routes in the hash, so its address never leaves that path; the sign-in and launch pages under it
 *  can't open a chat. */
function isWebappPage(url) {
  const base = new URL(self.registration.scope).pathname.replace(/\/$/, '')
  const { pathname } = new URL(url)

  return pathname === base || pathname === `${base}/` || pathname === `${base}/index.html`
}

/** The id *tab* puts in its notifications, or null when it does not answer in time (a frozen tab). */
function tabIdOf(tab) {
  return new Promise(resolve => {
    const { port1, port2 } = new MessageChannel()

    const answer = id => {
      clearTimeout(timer)
      port1.close()
      resolve(id)
    }

    const timer = setTimeout(() => answer(null), TAB_ANSWER_MS)
    port1.onmessage = event => answer(event.data)
    tab.postMessage({ type: TAB_QUERY }, [port2])
  })
}

/** Of *tabs* (most recently focused first), the one that showed the notification (*id*), else the first
 *  that did not answer and may have, else the first: another tab can still look the session up. */
async function notifyingTab(tabs, id) {
  const answers = await Promise.all(tabs.map(tabIdOf))

  return tabs.find((_, i) => answers[i] === id) ?? tabs.find((_, i) => answers[i] === null) ?? tabs[0]
}

/** Bring forward the Webapp tab that showed the notification, else the one used last, and have it open the
 *  notification's session; with no Webapp tab open, open the Webapp at the session on the notifying tab's
 *  profile. */
async function openHermes({ focusSessionId, profile, tab: notifier }) {
  const session = typeof focusSessionId === 'string' && focusSessionId ? focusSessionId : ''
  const owner = typeof profile === 'string' ? profile.trim() : ''
  // Focused first, then by most recent focus. A tab opened before this worker registered is not
  // controlled by it, and still counts.
  const windows = await self.clients.matchAll({ includeUncontrolled: true, type: 'window' })
  const tabs = windows.filter(client => isWebappPage(client.url))

  // Only a choice between tabs asks: a lone tab takes the click, as the constructor's onclick did. A
  // client's URL can't tell the tabs apart: it is the one its document was created with, from before
  // any profile switch.
  const tab =
    session && typeof notifier === 'string' && notifier && tabs.length > 1
      ? await notifyingTab(tabs, notifier)
      : tabs[0]

  if (!tab) {
    // The chat's route, as sessionRoute() in src/app/routes.ts builds it, on the notifying tab's profile,
    // written out as sessionWindowTarget() does: without it the window boots on the origin's saved default.
    const query = owner ? `?profile=${encodeURIComponent(owner)}` : ''
    await self.clients.openWindow(
      `${self.registration.scope}${query}${session ? `#/${encodeURIComponent(session)}` : ''}`
    )

    return
  }

  // The message goes first: the tab opens the session while the browser brings it forward.
  if (session) {
    tab.postMessage({ sessionId: session, type: FOCUS_SESSION_MESSAGE })
  }

  await tab.focus()
}
