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
import type { ExternalOpenFailedPayload } from '@/global.d'
import { useI18n } from '@/i18n'
import { isBrowserWindow, isHudWindow } from '@/store/windows'

// Global fallback for a failed external URL open. main's openExternalUrl
// reports every `shell.openExternal` rejection over a dedicated IPC event
// (e.g. no https handler registered on Linux — a dead click otherwise). The
// dialog shows the URL so the user can copy it and open it manually.
export function ExternalOpenFailedDialog() {
  const { t } = useI18n()
  const [failure, setFailure] = useState<ExternalOpenFailedPayload | null>(null)

  useEffect(() => {
    if (!window.hermesDesktop?.onExternalOpenFailed) {
      return
    }

    return window.hermesDesktop.onExternalOpenFailed(setFailure)
  }, [])

  if (isHudWindow() || isBrowserWindow()) {
    return null
  }

  // A missing file and a missing URL handler are different failures with
  // different remedies; main tags the payload so copy stays localized.
  const copy = t.externalOpenFailed
  const header = failure?.code === 'missing-file' ? copy.missing : copy

  return (
    <Dialog onOpenChange={open => (!open ? setFailure(null) : undefined)} open={Boolean(failure)}>
      <DialogContent className="max-w-[30rem]" showCloseButton={false}>
        <DialogHeader>
          <DialogTitle>{header.title}</DialogTitle>
          <DialogDescription>{header.message}</DialogDescription>
        </DialogHeader>
        <div className="max-h-40 overflow-auto break-all rounded-md border bg-muted p-3 select-all">{failure?.url}</div>
        <DialogFooter>
          {failure ? <CopyButton text={failure.url}>{copy.copyUrl}</CopyButton> : null}
          <Button onClick={() => setFailure(null)} variant="ghost">
            {copy.close}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
