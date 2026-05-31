// Opt-in demo / fixture mode. When on, the renderer runs against canned data and
// a fake gateway (no backend) — for screenshots, e2e tests, and demo videos.
// Off by default; the demo module is dynamically imported only when this returns
// true (see main.tsx), so production builds tree-shake it out entirely.
//
// Enable via any of:
//   - build flag:   VITE_HERMES_DEMO=1
//   - URL param:    #/...?demo=1   (HashRouter, so the query lives on the hash)
//   - localStorage: hermes.demo = "1"
export function isDemoMode(): boolean {
  try {
    if (import.meta.env.VITE_HERMES_DEMO === '1') {
      return true
    }

    const hash = window.location.hash
    const query = hash.includes('?') ? hash.slice(hash.indexOf('?') + 1) : ''

    if (new URLSearchParams(query).get('demo') === '1') {
      return true
    }

    return window.localStorage.getItem('hermes.demo') === '1'
  } catch {
    return false
  }
}
