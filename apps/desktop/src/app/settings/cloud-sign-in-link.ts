import { useEffect, useState } from 'react'

import { notify, notifyError } from '@/store/notifications'

// The authorize URL exists only once main's loopback listener is up, a beat
// after the sign-in starts; poll briefly for it while the flow is pending.
const POLL_INTERVAL_MS = 250
const POLL_ATTEMPTS = 40

/**
 * The pending Hermes Cloud sign-in's authorize URL (for "browser didn't
 * open? copy the link"), or null until main has one. Shared by Settings →
 * Gateway and the boot-failure recovery card.
 */
export function useCloudSignInLink(pending: boolean): null | string {
  const [url, setUrl] = useState<null | string>(null)

  useEffect(() => {
    if (!pending) {
      return
    }

    let cancelled = false

    void (async () => {
      for (let attempt = 0; attempt < POLL_ATTEMPTS && !cancelled; attempt++) {
        const result = await window.hermesDesktop?.cloud?.loginUrl?.().catch(() => null)

        if (cancelled) {
          return
        }

        setUrl(result?.url ?? null)

        if (result?.url) {
          return
        }

        await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL_MS))
      }
    })()

    return () => {
      cancelled = true
    }
  }, [pending])

  return pending ? url : null
}

export async function copyCloudSignInLink(
  url: string,
  copy: { cloudSignInLinkCopied: string; cloudSignInLinkCopiedMessage: string; cloudSignInFailed: string }
): Promise<void> {
  try {
    await navigator.clipboard.writeText(url)
    notify({ kind: 'success', title: copy.cloudSignInLinkCopied, message: copy.cloudSignInLinkCopiedMessage })
  } catch (err) {
    notifyError(err, copy.cloudSignInFailed)
  }
}

/** Abort a pending browser sign-in (closing the tab alone leaves it waiting). */
export function cancelCloudSignIn(): void {
  void window.hermesDesktop?.cloud?.cancelLogin?.().catch(() => undefined)
}
