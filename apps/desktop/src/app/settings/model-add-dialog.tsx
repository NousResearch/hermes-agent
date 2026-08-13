import { useEffect, useState } from 'react'

import { ActionStatus } from '@/components/ui/action-status'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { AlertTriangle } from '@/lib/icons'
import type { CustomProviderModel } from '@/lib/custom-provider-config'

export function ModelAddDialog({
  existingIds = [],
  onClose,
  onSave,
  open
}: {
  /** Existing model ids for the provider, for duplicate checks. */
  existingIds?: string[]
  onClose: () => void
  onSave: (model: CustomProviderModel) => Promise<void> | void
  open: boolean
}) {
  const { t } = useI18n()
  const p = t.providerManager

  const [modelId, setModelId] = useState('')
  const [modelName, setModelName] = useState('')
  const [status, setStatus] = useState<'done' | 'idle' | 'saving'>('idle')
  const [error, setError] = useState<null | string>(null)

  useEffect(() => {
    if (!open) {
      return
    }
    setModelId('')
    setModelName('')
    setError(null)
    setStatus('idle')
  }, [open])

  const trimmedId = modelId.trim()
  const idExists = trimmedId !== '' && existingIds.includes(trimmedId)
  const invalid = idExists
  const busy = status === 'saving' || status === 'done'

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()

    if (!trimmedId) {
      setError(p.modelIdRequired)
      return
    }

    if (idExists) {
      setError(p.modelExists)
      return
    }

    setStatus('saving')
    setError(null)

    try {
      await onSave({ id: trimmedId, name: modelName.trim() || undefined })
      setStatus('done')
      window.setTimeout(onClose, 600)
    } catch (err) {
      setStatus('idle')
      setError(err instanceof Error ? err.message : p.modelIdRequired)
    }
  }

  return (
    <Dialog onOpenChange={value => !value && !busy && onClose()} open={open}>
      <DialogContent
        className="max-w-md"
        onEscapeKeyDown={e => e.preventDefault()}
        onInteractOutside={e => e.preventDefault()}
        onPointerDownOutside={e => e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle>{p.manualModelTitle}</DialogTitle>
          <DialogDescription>{p.manualModelDescription}</DialogDescription>
        </DialogHeader>

        <form className="grid gap-4" onSubmit={handleSubmit}>
          <div className="grid gap-1.5">
            <label className="text-xs font-medium" htmlFor="model-add-id">
              {p.modelId}
            </label>
            <Input
              aria-invalid={idExists}
              autoFocus
              disabled={busy}
              id="model-add-id"
              onChange={event => setModelId(event.target.value)}
              placeholder={p.modelIdPlaceholder}
              value={modelId}
            />
          </div>

          <div className="grid gap-1.5">
            <label className="text-xs font-medium" htmlFor="model-add-name">
              {p.modelName}
            </label>
            <Input
              disabled={busy}
              id="model-add-name"
              onChange={event => setModelName(event.target.value)}
              placeholder={p.modelNamePlaceholder}
              value={modelName}
            />
          </div>

          <details className="rounded-md border border-border/60 px-3 py-2">
            <summary className="cursor-pointer text-xs font-medium">{p.advancedParameters}</summary>
            <p className="mt-2 text-xs text-muted-foreground">{p.advancedParametersEmpty}</p>
          </details>

          {error && (
            <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <DialogFooter className="gap-2">
            <Button disabled={busy} onClick={onClose} type="button" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button disabled={busy || !trimmedId || invalid} type="submit">
              <ActionStatus
                busy={t.common.saving}
                done={t.common.done}
                idle={t.common.save}
                state={status}
              />
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
