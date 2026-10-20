import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { CopyButton } from '@/components/ui/copy-button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { useI18n } from '@/i18n'
import { dismissAuthorizationUrl, subscribeAuthorizationUrl } from '@/lib/authorization-url'
import { isBrowserWindow, isHudWindow } from '@/store/windows'

/** Authorization URLs open in whatever Chrome profile the OS picks. Show the
 *  URL so the user can paste it into the right profile, and only open the
 *  default browser when they ask. */
export function AuthorizationUrlDialog() {
  const { t } = useI18n()
  const [url, setUrl] = useState<string | null>(null)

  useEffect(() => subscribeAuthorizationUrl(next => setUrl(next?.url ?? null)), [])

  if (isHudWindow() || isBrowserWindow() || !url) {
    return null
  }

  const copy = t.authorizationUrl

  return (
    <Dialog onOpenChange={open => (!open ? dismissAuthorizationUrl() : undefined)} open>
      <DialogContent className="max-w-[30rem]" showCloseButton={false}>
        <DialogHeader>
          <DialogTitle>{copy.title}</DialogTitle>
          <DialogDescription>{copy.message}</DialogDescription>
        </DialogHeader>
        <div className="max-h-40 overflow-auto break-all rounded-md border bg-muted p-3 select-text">{url}</div>
        <DialogFooter>
          <CopyButton label={copy.copyUrl} text={url}>
            {copy.copyUrl}
          </CopyButton>
          <Button
            onClick={() => {
              void window.hermesDesktop?.openExternal?.(url)
            }}
            variant="secondary"
          >
            {copy.openInBrowser}
          </Button>
          <Button onClick={dismissAuthorizationUrl} variant="ghost">
            {copy.close}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
