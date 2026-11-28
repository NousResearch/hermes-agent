import { useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { exportSessionFile, type ExportSessionFileTarget } from '@/lib/session-export'
import { DEFAULT_TRANSCRIPT_OPTIONS, type TranscriptOptions } from '@/lib/transcript-document'
import { notify, notifyError } from '@/store/notifications'

import { SessionExportOptions } from './session-export-options'

type ExportTarget = Pick<ExportSessionFileTarget, 'sessionId' | 'profile' | 'title'>

interface SessionExportDialogProps {
  target: ExportTarget | null
  onClose: () => void
}

export function SessionExportDialog({ target, onClose }: SessionExportDialogProps) {
  return target ? <ExportSessionForm onClose={onClose} target={target} /> : null
}

function ExportSessionForm({ target, onClose }: { target: ExportTarget; onClose: () => void }) {
  const { t } = useI18n()
  const [format, setFormat] = useState<ExportSessionFileTarget['format']>('md')
  const [redact, setRedact] = useState(false)

  const [options, setOptions] = useState<TranscriptOptions>({ ...DEFAULT_TRANSCRIPT_OPTIONS })

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const busyRef = useRef(false)

  const submit = async () => {
    if (busyRef.current) {
      return
    }

    busyRef.current = true
    setBusy(true)
    setError('')

    try {
      const result = await exportSessionFile({ ...target, format, options, redact })

      if (result.kind === 'saved' || result.kind === 'json-download') {
        notify({ kind: 'success', message: t.desktop.sessionExported, durationMs: 2_000 })
        onClose()
      } else if (result.kind === 'needs-live-backend') {
        setError(t.desktop.sessionExportNeedsLive)
      } else if (result.kind === 'unsupported') {
        setError(t.desktop.sessionExportUnsupported)
      } else if (result.kind === 'failed') {
        setError(t.desktop.sessionExportFailed)
        notifyError(result.error, t.desktop.sessionExportFailed)
      }
    } catch (err) {
      setError(t.desktop.sessionExportFailed)
      notifyError(err, t.desktop.sessionExportFailed)
    } finally {
      busyRef.current = false
      setBusy(false)
    }
  }

  return (
    <Dialog
      onOpenChange={open => {
        if (!open && !busyRef.current) {
          onClose()
        }
      }}
      open
    >
      <DialogContent
        className="max-h-[calc(100dvh-3rem)] max-w-lg overflow-y-auto"
        onEscapeKeyDown={event => {
          if (busyRef.current) {
            event.preventDefault()
          }
        }}
        onInteractOutside={event => {
          if (busyRef.current) {
            event.preventDefault()
          }
        }}
        showCloseButton={!busy}
      >
        <DialogHeader>
          <DialogTitle>{t.desktop.sessionExportTitle}</DialogTitle>
          <DialogDescription>{target.title || target.sessionId}</DialogDescription>
        </DialogHeader>
        <div className="grid gap-2">
          <span>{t.desktop.sessionExportFormat}</span>
          <SegmentedControl
            className="w-full"
            disabled={busy}
            onChange={setFormat}
            options={[
              { id: 'docx', label: 'DOCX' },
              { id: 'json', label: 'JSON' },
              { id: 'odt', label: 'ODT' },
              { id: 'md', label: 'Markdown' },
              { id: 'html', label: 'HTML' }
            ]}
            sizing="content"
            value={format}
          />
        </div>
        <SessionExportOptions
          busy={busy}
          format={format}
          onChange={(key, value) => setOptions(previous => ({ ...previous, [key]: value }))}
          options={options}
          redact={redact}
        />
        <div className="grid gap-1">
          <label className="flex items-center gap-2">
            <Checkbox checked={redact} disabled={busy} onCheckedChange={value => setRedact(value === true)} />
            {t.desktop.sessionExportRedact}
          </label>
          <p className="text-sm text-(--ui-text-tertiary)">{t.desktop.sessionExportRedactHint}</p>
        </div>
        {error && (
          <p className="text-destructive" role="alert">
            {error}
          </p>
        )}
        <DialogFooter>
          <Button disabled={busy} onClick={onClose} type="button" variant="ghost">
            {t.common.cancel}
          </Button>
          <Button disabled={busy} onClick={() => void submit()} type="button">
            {busy ? t.common.loading : t.desktop.sessionExportAction}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
