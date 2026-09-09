// Send Diagnostics — the consent-gated debug-bundle upload dialog.
//
// Rendered globally (wiring.tsx, beside ConfirmHost) and driven by the
// $sendDiagnostics store: any surface (the failed-turn error card today)
// opens it via requestSendDiagnostics(). Three faces:
//   consent   — privacy notice (what's collected, who can see it, retention)
//               with an explicit Upload button; nothing is sent before it.
//   uploading — spinner while the backend collects, redacts and uploads.
//   done      — the private view link (copyable) + where to pick up the
//               discussion: GitHub Issues · Nous Portal Support · Discord.
import { useStore } from '@nanostores/react'

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
import { ExternalLink as ExternalLinkAnchor, openExternalLink } from '@/lib/external-link'
import { ExternalLink, Loader2Icon, Lock } from '@/lib/icons'
import { $sendDiagnostics, confirmSendDiagnostics, dismissSendDiagnostics } from '@/store/send-diagnostics'

const SUPPORT_LINKS = [
  { key: 'github', url: 'https://github.com/NousResearch/hermes-agent/issues' },
  { key: 'portal', url: 'https://portal.nousresearch.com/help' },
  { key: 'discord', url: 'https://discord.gg/NousResearch' }
] as const

export function SendDiagnosticsHost() {
  const { t } = useI18n()
  const copy = t.sendDiagnostics
  const state = useStore($sendDiagnostics)

  if (!state) {
    return null
  }

  const busy = state.phase === 'uploading'

  return (
    // Dismissal is allowed in EVERY phase, including mid-upload: the store's
    // generation guard makes a dismissed upload's completion a no-op, so Esc/
    // backdrop/Cancel are always an immediate way out (cancellation of the
    // in-flight request itself stays best-effort).
    <Dialog onOpenChange={open => (!open ? dismissSendDiagnostics() : undefined)} open>
      <DialogContent className="max-w-[30rem]">
        {state.phase === 'consent' || state.phase === 'uploading' ? (
          <>
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <Lock className="size-4 text-(--ui-text-tertiary)" />
                {copy.title}
              </DialogTitle>
              <DialogDescription className="whitespace-pre-line text-left">{copy.privacyNotice}</DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button onClick={dismissSendDiagnostics} variant="ghost">
                {copy.cancel}
              </Button>
              <Button disabled={busy} onClick={() => void confirmSendDiagnostics()}>
                {busy ? (
                  <span className="flex items-center gap-1.5">
                    <Loader2Icon className="size-3.5 animate-spin" />
                    {copy.uploading}
                  </span>
                ) : (
                  copy.upload
                )}
              </Button>
            </DialogFooter>
          </>
        ) : state.phase === 'error' ? (
          <>
            <DialogHeader>
              <DialogTitle>{copy.failedTitle}</DialogTitle>
              <DialogDescription className="text-left">
                {state.error}
                {'\n'}
                {copy.failedHint}
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button onClick={dismissSendDiagnostics} variant="ghost">
                {copy.close}
              </Button>
            </DialogFooter>
          </>
        ) : (
          <>
            <DialogHeader>
              <DialogTitle>{copy.doneTitle}</DialogTitle>
              <DialogDescription className="text-left">{copy.doneDescription}</DialogDescription>
            </DialogHeader>
            {(state.result?.viewUrl || state.result?.uploadId) && (
              <div
                className="flex items-center gap-2 rounded-md border border-(--ui-stroke-tertiary) px-3 py-2"
                data-selectable-text="true"
              >
                {state.result.viewUrl ? (
                  // A real anchor, not a <code> span: right-click resolves the
                  // link context menu (open / copy URL), left-click opens it,
                  // and the row's data-selectable-text keeps drag-to-select
                  // working. `truncate` only clips the paint — selection and
                  // copy still carry the full URL. `native` because the in-app
                  // preview pane would open BEHIND this modal dialog; the
                  // support buttons below already go to the system browser.
                  <ExternalLinkAnchor
                    className="min-w-0 flex-1 truncate font-mono text-[0.78rem] text-(--ui-text-secondary)"
                    href={state.result.viewUrl}
                    native
                    title={state.result.viewUrl}
                  >
                    {state.result.viewUrl}
                  </ExternalLinkAnchor>
                ) : (
                  <code className="min-w-0 flex-1 truncate text-[0.78rem] text-(--ui-text-secondary)">
                    {copy.uploadIdFallback(state.result.uploadId ?? '')}
                  </code>
                )}
                <CopyButton
                  appearance="inline"
                  className="shrink-0"
                  label={copy.copyLink}
                  text={state.result.viewUrl ?? state.result.uploadId ?? ''}
                />
              </div>
            )}
            <div className="text-[0.8rem] text-(--ui-text-secondary)">{copy.handoffLead}</div>
            <div className="flex flex-wrap gap-1.5">
              {SUPPORT_LINKS.map(link => (
                <Button key={link.key} onClick={() => openExternalLink(link.url)} size="sm" variant="outline">
                  <ExternalLink className="size-3" />
                  {copy.links[link.key]}
                </Button>
              ))}
            </div>
            <DialogFooter>
              <Button onClick={dismissSendDiagnostics} variant="ghost">
                {copy.close}
              </Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}
