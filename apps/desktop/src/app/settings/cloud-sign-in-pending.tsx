import { useEffect, useRef } from 'react'

import { Button } from '@/components/ui/button'
import type { Translations } from '@/i18n/types'

import { cancelCloudSignIn, copyCloudSignInLink, useCloudSignInLink } from './cloud-sign-in-link'
import { ListRow } from './primitives'

type GatewayCopy = Translations['settings']['gateway']

// Mounted only while a Hermes Cloud browser sign-in is pending, so the
// authorize-URL poll runs exactly as long as the flow does.
function CloudSignInActions({ copy, focusCancel }: { copy: GatewayCopy; focusCancel?: boolean }) {
  const url = useCloudSignInLink(true)
  const cancelRef = useRef<HTMLButtonElement>(null)

  // Keyboard users land on the way out of the pending browser flow.
  useEffect(() => {
    if (focusCancel) {
      cancelRef.current?.focus()
    }
  }, [focusCancel])

  return (
    <>
      {/* "Browser didn't open?" — xdg-open and friends can report success
          without a browser appearing; hand over the link. */}
      <Button disabled={!url} onClick={() => url && void copyCloudSignInLink(url, copy)} size="sm" variant="text">
        {copy.cloudCopySignInLink}
      </Button>
      <Button onClick={cancelCloudSignIn} ref={cancelRef} size="sm" variant="outline">
        {copy.cloudCancelSignIn}
      </Button>
    </>
  )
}

/** Settings → Gateway: the pending browser sign-in row (Copy link / Cancel). */
export function CloudSignInPendingRow({ copy }: { copy: GatewayCopy }) {
  return (
    <div aria-live="polite" role="status">
      <ListRow
        action={
          <div className="flex items-center gap-2">
            <CloudSignInActions copy={copy} focusCancel />
          </div>
        }
        description={copy.cloudBrowserPendingDesc}
        title={copy.cloudBrowserPendingTitle}
      />
    </div>
  )
}

/** Boot-failure recovery card: the same affordances in a compact strip. */
export function CloudSignInPendingStrip({ copy, pending }: { copy: GatewayCopy; pending: boolean }) {
  if (!pending) {
    return null
  }

  return (
    <div aria-live="polite" className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground" role="status">
      <span className="mr-auto">{copy.cloudBrowserPendingTitle}</span>
      <CloudSignInActions copy={copy} />
    </div>
  )
}
