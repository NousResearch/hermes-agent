import { randomUuid } from '@hermes/shared'

import type { HermesApiRequest } from '@/global'

interface BrowserWindowOptions {
  api: <T>(request: HermesApiRequest) => Promise<T>
  basePath: string
  privateSession: boolean
}

const POPUP_BLOCKED = 'Webapp window did not open. Allow popups for this site and try again.'

// `noopener` makes a successful open return null (HTML window open step 18),
// so a blocked popup is indistinguishable. Sever opener on the handle instead.
function openWebappWindow(href: string): Window | null {
  const opened = window.open(href, '_blank')

  if (!opened) {return null}

  opened.opener = null

  return opened
}

/** Open synchronously in the user gesture; never rely on an opener/storage copy. */
export function createBrowserWindowOpener({ api, basePath, privateSession }: BrowserWindowOptions) {
  return (target: URL): Promise<{ ok: true }> => {
    if (!privateSession) {
      if (!openWebappWindow(target.href)) {
        return Promise.reject(new Error(POPUP_BLOCKED))
      }

      return Promise.resolve({ ok: true })
    }

    const id = randomUuid()
    const channel = new BroadcastChannel(`hermes.webapp.window:${id}`)
    const launch = new URL(`${basePath}/webapp/window`, window.location.origin)
    launch.hash = new URLSearchParams({ channel: id, query: target.search, route: target.hash.slice(1) }).toString()

    return new Promise((resolve, reject) => {
      let minting = false
      let settled = false

      const finish = (error?: Error) => {
        if (settled) {return}
        settled = true
        window.clearTimeout(timeout)
        channel.close()

        if (error) {reject(error)} else {resolve({ ok: true })}
      }

      const timeout = window.setTimeout(() => finish(new Error(POPUP_BLOCKED)), 30_000)

      channel.onmessage = event => {
        if (settled) {return}

        if (event.data?.type === 'ready' && !minting) {
          minting = true
          void api<{ ticket: string }>({ method: 'POST', path: '/api/webapp/window-ticket' }).then(({ ticket }) => {
            if (settled) {return}
            channel.postMessage({ ticket })
          }).catch(error => {
            if (settled) {return}
            channel.postMessage({ error: 'Could not authorize this window. Try opening it again from the original tab.' })
            finish(error instanceof Error ? error : new Error(String(error)))
          })
        } else if (event.data?.type === 'done') {
          finish()
        } else if (event.data?.type === 'error') {
          finish(new Error('Could not authorize the Webapp window. Try opening it again.'))
        }
      }

      try {
        if (!openWebappWindow(launch.href)) {
          finish(new Error(POPUP_BLOCKED))
        }
      } catch (error) {
        finish(error instanceof Error ? error : new Error(String(error)))
      }
    })
  }
}

/** The secondary-window URL showing one session, or null for an empty session id. */
export function sessionWindowTarget(
  currentHref: string,
  sessionId: string,
  opts?: { profile?: null | string; watch?: boolean }
): URL | null {
  const id = sessionId.trim()

  if (!id) {return null}
  const target = new URL(currentHref)
  target.searchParams.set('win', 'secondary')

  // Written out even for Default: without `?profile=` the window boots on
  // this origin's saved default, which may be another profile.
  const profile = opts?.profile?.trim()

  if (profile) {
    target.searchParams.set('profile', profile)
  }

  if (opts?.watch) {
    target.searchParams.set('watch', '1')
  } else {
    target.searchParams.delete('watch')
  }

  target.hash = `/${encodeURIComponent(id)}`

  return target
}
