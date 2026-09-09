import { useEffect, useState } from 'react'

import type { ProfileScope } from '@/api/client'
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
import { Field, FieldHint } from '@/components/ui/field'
import { SanitizedInput } from '@/components/ui/sanitized-input'
import { renameProfile } from '@/hermes'
import { useI18n } from '@/i18n'
import { AlertTriangle } from '@/lib/icons'
import { slug } from '@/lib/sanitize'
import { retireLocalProfileGateways } from '@/store/gateway'

import { isValidProfileName } from './create-profile-dialog'

// Display names are free text (Unicode fine) — no slug sanitizing.
const identity = (raw: string) => raw

// Self-contained rename (owns the renameProfile call) so every caller just
// reacts via onRenamed. Unchanged name is a no-op close.
export function RenameProfileDialog({
  currentName,
  isDefault = false,
  onClose,
  onRenamed,
  open,
  scope
}: {
  currentName: string
  /** Default profile: sets a presentation-only display name (Unicode ok);
   *  the canonical id stays "default" and no backend teardown is needed. */
  isDefault?: boolean
  onClose: () => void
  onRenamed?: (name: string) => Promise<void> | void
  open: boolean
  /** Explicit (connection, profile) owner for a remote-gateway profile: the
   *  rename executes there and no local backend is retired. */
  scope?: ProfileScope
}) {
  const { t } = useI18n()
  const p = t.profiles
  const [name, setName] = useState(currentName)
  const [status, setStatus] = useState<'done' | 'idle' | 'saving'>('idle')
  const [error, setError] = useState<null | string>(null)

  useEffect(() => {
    if (!open) {
      return
    }

    // Display-name mode starts blank — "default" is the id, not a name.
    setName(isDefault ? '' : currentName)
    setError(null)
    setStatus('idle')
  }, [currentName, isDefault, open])

  const trimmed = name.trim()
  const unchanged = !isDefault && trimmed === currentName
  const invalid = trimmed !== '' && !unchanged && !isDefault && !isValidProfileName(trimmed)
  const busy = status === 'saving' || status === 'done'

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()

    if (unchanged) {
      onClose()

      return
    }

    if (!trimmed || invalid) {
      setError(invalid ? p.invalidName(p.nameHint) : p.nameRequired)

      return
    }

    setStatus('saving')
    setError(null)

    try {
      // A retained renderer socket for the old name would treat the rename's
      // backend teardown as a transient drop and redial, resurrecting the
      // old-name backend whose ensure_hermes_home() recreates the directory
      // the rename just moved (same class as the delete path, #88638).
      if (!isDefault && scope == null) {
        retireLocalProfileGateways(currentName)
      }

      await (scope == null ? renameProfile(currentName, trimmed) : renameProfile(currentName, trimmed, scope))
      await onRenamed?.(trimmed)
      setStatus('done')
      window.setTimeout(onClose, 800)
    } catch (err) {
      setStatus('idle')
      setError(err instanceof Error ? err.message : p.failedRename)
    }
  }

  return (
    <Dialog onOpenChange={value => !value && !busy && onClose()} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{isDefault ? p.displayNameTitle : p.renameTitle}</DialogTitle>
          <DialogDescription>
            {isDefault ? (
              p.displayNameDesc
            ) : (
              <>
                {p.renameDescPrefix}
                <span className="font-mono">~/.local/bin</span>
                {p.renameDescSuffix}
              </>
            )}
          </DialogDescription>
        </DialogHeader>

        <form className="grid gap-4" onSubmit={handleSubmit}>
          <Field htmlFor="rename-profile-name" label={isDefault ? p.displayNameLabel : p.newNameLabel}>
            <SanitizedInput
              aria-invalid={invalid}
              autoFocus
              id="rename-profile-name"
              onValueChange={setName}
              sanitize={isDefault ? identity : slug}
              value={name}
            />
            {!isDefault && <FieldHint error={invalid}>{p.nameHint}</FieldHint>}
          </Field>

          {error && (
            <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <DialogFooter>
            <Button disabled={busy} onClick={onClose} type="button" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button disabled={busy || invalid || unchanged} type="submit">
              <ActionStatus busy={p.renaming} done={p.renamed} idle={p.rename} state={status} />
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
