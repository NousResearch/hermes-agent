import type { HermesApiRequest } from '@/global'

import type { UiPreferencesBridge, UiPreferencesSnapshot } from './ui-preferences/sync'

const PATH = '/api/ui-preferences'
const POLL_MS = 10_000

/**
 * The Webapp's door to the serving machine's UI preferences store. The server
 * cannot push, so changes made by another device, origin or native Desktop on
 * that machine arrive by a cheap poll while this tab is visible, and at once
 * when the tab regains focus.
 */
export function createBrowserUiPreferences(api: <T>(request: HermesApiRequest) => Promise<T>): UiPreferencesBridge {
  const read = () => api<UiPreferencesSnapshot>({ path: PATH, timeoutMs: 10_000 })

  return {
    read,
    write: payload => api<UiPreferencesSnapshot>({ body: payload, method: 'PATCH', path: PATH, timeoutMs: 10_000 }),
    onChanged: callback => {
      const poll = () => {
        if (document.visibilityState === 'visible') {
          read().then(callback, () => undefined)
        }
      }

      const timer = window.setInterval(poll, POLL_MS)
      window.addEventListener('focus', poll)
      document.addEventListener('visibilitychange', poll)

      return () => {
        window.clearInterval(timer)
        window.removeEventListener('focus', poll)
        document.removeEventListener('visibilitychange', poll)
      }
    }
  }
}
