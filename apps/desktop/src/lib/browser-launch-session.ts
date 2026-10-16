const FRAGMENT_KEY = 'hermes-session'

export const WEBAPP_LAUNCH_REQUIRED = 'Open the private launch link printed by hermes webapp in this tab. If the server restarted or the link is lost, restart that Webapp to get a new link. Keep the link private: it grants host access.'

/** Pasting the launch link over a bare URL is a same-document navigation. */
export function watchWebappLaunchLink(reload: () => void): void {
  const onHash = () => {
    // Leave the fragment intact: startup consumes it even if storage is disabled.
    if (window.location.hash.startsWith(`#${FRAGMENT_KEY}=`)) {reload()}
  }

  window.addEventListener('hashchange', onHash)
  window.addEventListener('beforeunload', () => window.removeEventListener('hashchange', onHash), { once: true })
}

/** HTTP never supplies this credential; reloads keep only this tab's grant. */
export function consumeWebappSession(basePath: string): string {
  const key = `hermes.webapp.session.v1:${JSON.stringify([window.location.origin, basePath])}`

  // HashRouter owns other fragments; don't parse/rewrite its route on reload.
  if (window.location.hash.startsWith(`#${FRAGMENT_KEY}=`)) {
    const supplied = new URLSearchParams(window.location.hash.slice(1)).getAll(FRAGMENT_KEY)
    window.history.replaceState(window.history.state, '', `${window.location.pathname}${window.location.search}`)
    const token = supplied.length === 1 && /^[A-Za-z0-9_-]{43}$/.test(supplied[0]) ? supplied[0] : ''

    try {
      if (token) {sessionStorage.setItem(key, token)} else {sessionStorage.removeItem(key)}
    } catch { /* Storage may be disabled; the current page can still use its link. */ }

    return token
  }

  try { return sessionStorage.getItem(key) || '' } catch { return '' }
}
