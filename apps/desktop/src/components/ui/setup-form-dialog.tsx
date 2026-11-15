import type { ConnectionTargetState } from '@hermes/shared'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { FieldHint } from '@/components/ui/field'
import { type SetupField, SetupFieldList } from '@/components/ui/setup-field-list'

interface SetupFormDialogCopy {
  cancel: string
  connect: string
  openInBrowser: string
  setup: (server: string) => string
}

interface SetupFormDialogProps {
  copy: SetupFormDialogCopy
  detail?: string
  fields: SetupField[]
  instructions?: null | string
  onCancel: () => void
  onConnect: (env: Record<string, string>) => void
  onOpenBrowser: () => void
  open: boolean
  pending: boolean
  server: string
  status: ConnectionTargetState
  url?: null | string
}

const initialDraft = (fields: SetupField[]): Record<string, string> =>
  Object.fromEntries(fields.map(field => [field.name, field.secret ? '' : field.default]))

export function SetupFormDialog({
  copy,
  detail,
  fields,
  instructions,
  onCancel,
  onConnect,
  onOpenBrowser,
  open,
  pending,
  server,
  status,
  url
}: SetupFormDialogProps) {
  const [draft, setDraft] = useState<Record<string, string>>({})

  // The backend sends a fresh field list with every frame (and an empty one while an attempt runs).
  // Fields only fill in what the draft lacks, so a failed Connect keeps what was typed. Closing the
  // dialog drops the draft: a typed secret does not outlive the form.
  useEffect(() => {
    setDraft(current => (open ? { ...initialDraft(fields), ...current } : {}))
  }, [fields, open])

  const missingRequired = fields.some(field => field.required && !draft[field.name]?.trim())
  const awaitingBrowser = status === 'initiated' && Boolean(url)

  return (
    <Dialog onOpenChange={nextOpen => !nextOpen && onCancel()} open={open}>
      <DialogContent className="w-[min(32rem,calc(100vw-2rem))]" showCloseButton={false}>
        <DialogHeader>
          <DialogTitle className="truncate">{copy.setup(server)}</DialogTitle>
          {instructions ? (
            <DialogDescription className="max-h-32 overflow-y-auto whitespace-pre-wrap break-words">
              {instructions}
            </DialogDescription>
          ) : null}
        </DialogHeader>

        <SetupFieldList
          disabled={pending}
          draft={draft}
          fields={fields}
          onChange={(name, value) => setDraft(current => ({ ...current, [name]: value }))}
        />

        {status === 'failed' && detail ? <FieldHint error>{detail}</FieldHint> : null}

        {awaitingBrowser ? (
          <div className="grid min-w-0 gap-2">
            <p className="break-all text-xs text-(--ui-text-secondary)">{url}</p>
            <Button className="justify-self-start" onClick={onOpenBrowser} size="sm" variant="secondary">
              {copy.openInBrowser}
            </Button>
          </div>
        ) : null}

        <DialogFooter>
          <Button disabled={pending} onClick={onCancel} variant="outline">
            {copy.cancel}
          </Button>
          <Button
            disabled={missingRequired || pending || awaitingBrowser}
            loading={pending}
            onClick={() => onConnect(draft)}
          >
            {copy.connect}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
