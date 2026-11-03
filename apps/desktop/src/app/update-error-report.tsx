// The update error box's support affordances: the updater output the user
// just watched scroll by, and a one-click debug share whose private link lands
// right here instead of in a terminal (`hermes debug share --nous`). The upload
// runs backend-side (diagnostics.share_nous collects that runtime's update.log
// and hand-off log); the output shown here rides along as a client file.
import { useStore } from '@nanostores/react'
import { type SyntheticEvent, useRef } from 'react'

import { DiagnosticsLinkRow } from '@/components/send-diagnostics-dialog'
import { Button } from '@/components/ui/button'
import { CopyButton } from '@/components/ui/copy-button'
import { useI18n } from '@/i18n'
import { Loader2Icon } from '@/lib/icons'
import {
  $sendDiagnostics,
  confirmSendDiagnostics,
  dismissSendDiagnostics,
  requestSendDiagnostics
} from '@/store/send-diagnostics'

export const UPDATE_OUTPUT_LABEL = 'update-output.log'

export function UpdateErrorReport({ log, message }: { log: readonly string[]; message: string }) {
  const { t } = useI18n()
  const copy = t.sendDiagnostics
  const state = useStore($sendDiagnostics)
  const report = state?.inline ? state : null
  const logText = log.join('\n')
  const logRef = useRef<HTMLPreElement>(null)

  // The failing line is the last one: open the output scrolled to its tail.
  const revealTail = (event: SyntheticEvent<HTMLDetailsElement>) => {
    if (event.currentTarget.open && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }

  const startReport = () =>
    requestSendDiagnostics(`Desktop update failed: ${message}`, {
      extraFiles: logText ? { [UPDATE_OUTPUT_LABEL]: logText } : {},
      inline: true
    })

  return (
    <div className="grid min-w-0 gap-3 text-left">
      {logText ? (
        <details className="text-xs text-muted-foreground" onToggle={revealTail}>
          <summary className="cursor-pointer select-none text-center">{t.updates.technicalDetails}</summary>
          <div className="relative mt-2">
            <pre
              className="max-h-48 overflow-auto whitespace-pre-wrap wrap-break-word rounded-md border border-border/70 bg-muted/35 px-3 py-2 pr-8 font-mono text-[0.6875rem] leading-4"
              data-selectable-text="true"
              ref={logRef}
            >
              {logText}
            </pre>
            <CopyButton appearance="inline" className="absolute right-1.5 top-1.5" text={logText} />
          </div>
        </details>
      ) : null}

      {!report ? (
        <Button onClick={startReport} variant="text">
          {t.errors.sendDiagnostics}
        </Button>
      ) : report.phase === 'consent' || report.phase === 'uploading' ? (
        <div className="grid gap-2 rounded-md border border-border/70 px-3 py-2">
          <p className="text-[0.6875rem] leading-4 text-muted-foreground">{copy.privacyNotice}</p>
          <div className="flex justify-end gap-2">
            <Button onClick={dismissSendDiagnostics} size="sm" variant="ghost">
              {copy.cancel}
            </Button>
            <Button disabled={report.phase === 'uploading'} onClick={() => void confirmSendDiagnostics()} size="sm">
              {report.phase === 'uploading' ? (
                <span className="flex items-center gap-1.5">
                  <Loader2Icon className="size-3.5 animate-spin" />
                  {copy.uploading}
                </span>
              ) : (
                copy.upload
              )}
            </Button>
          </div>
        </div>
      ) : report.phase === 'error' ? (
        <div className="grid gap-1 text-xs" data-selectable-text="true">
          <p className="font-medium text-destructive">{copy.failedTitle}</p>
          <p className="wrap-break-word text-muted-foreground">{report.error}</p>
          <p className="text-muted-foreground">{copy.failedHint}</p>
          <Button className="justify-self-center" onClick={startReport} size="sm" variant="ghost">
            {t.common.retry}
          </Button>
        </div>
      ) : (
        <div className="grid gap-2">
          <p className="text-center text-xs font-medium">{copy.doneTitle}</p>
          <p className="text-xs text-muted-foreground">{copy.doneDescription}</p>
          <DiagnosticsLinkRow result={report.result} />
        </div>
      )}
    </div>
  )
}
