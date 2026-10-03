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
import { Field, FieldHint } from '@/components/ui/field'
import { SanitizedInput } from '@/components/ui/sanitized-input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { createProfile, updateProfileSoul } from '@/hermes'
import { useI18n } from '@/i18n'
import { AlertTriangle } from '@/lib/icons'
import { findProfileNameConflict } from '@/lib/profile-name-guard'
import { slug } from '@/lib/sanitize'
import type { ProfileInfo } from '@/types/hermes'

const PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/

export function isValidProfileName(name: string): boolean {
  return PROFILE_NAME_RE.test(name.trim())
}

// Self-contained create flow (name + clone toggle + optional SOUL.md). Owns the
// createProfile/updateProfileSoul calls so every caller just refreshes/selects
// via onCreated. SOUL left blank keeps the cloned/blank persona untouched.
export function CreateProfileDialog({
  onClose,
  onCreated,
  onUseExisting,
  open,
  profiles = []
}: {
  onClose: () => void
  onCreated?: (name: string) => Promise<void> | void
  // "Take me to the profile this name was probably meant to be" — the caller owns
  // selection, so a typo is one click from the profile that already exists.
  onUseExisting: (name: string) => Promise<void> | void
  open: boolean
  profiles?: ProfileInfo[]
}) {
  const { t } = useI18n()
  const p = t.profiles
  const [name, setName] = useState('')
  const [cloneFrom, setCloneFrom] = useState<null | string>('default')
  const [soul, setSoul] = useState('')
  const [status, setStatus] = useState<'done' | 'idle' | 'saving'>('idle')
  const [error, setError] = useState<null | string>(null)

  useEffect(() => {
    if (!open) {
      return
    }

    setName('')
    setCloneFrom('default')
    setSoul('')
    setError(null)
    setStatus('idle')
  }, [open])

  const trimmed = name.trim()
  const invalid = trimmed !== '' && !isValidProfileName(trimmed)
  const busy = status === 'saving' || status === 'done'

  // This dialog is the one free-text door that mints a profile. A name one
  // keystroke from an existing one is almost always a typo, and a typo'd profile
  // boots as a real agent (stock SOUL.md, its own backend), so say so before the
  // scaffold instead of letting it happen silently.
  const conflict =
    trimmed && !invalid
      ? findProfileNameConflict(
          trimmed,
          profiles.map(profile => profile.name)
        )
      : null

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()

    if (!trimmed || invalid) {
      setError(invalid ? p.invalidName(p.nameHint) : p.nameRequired)

      return
    }

    if (conflict?.kind === 'duplicate') {
      setError(p.createConflict.duplicate(conflict.name))

      return
    }

    setStatus('saving')
    setError(null)

    try {
      await createProfile({ name: trimmed, clone_from: cloneFrom })

      if (soul.trim()) {
        await updateProfileSoul(trimmed, soul)
      }

      await onCreated?.(trimmed)
      setStatus('done')
      window.setTimeout(onClose, 800)
    } catch (err) {
      setStatus('idle')
      setError(err instanceof Error ? err.message : p.failedCreate)
    }
  }

  return (
    <Dialog onOpenChange={value => !value && !busy && onClose()} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{p.newProfile}</DialogTitle>
          <DialogDescription>{p.createDesc}</DialogDescription>
        </DialogHeader>

        <form className="grid gap-4" onSubmit={handleSubmit}>
          <Field htmlFor="new-profile-name" label={p.nameLabel}>
            <SanitizedInput
              aria-invalid={invalid}
              autoFocus
              id="new-profile-name"
              onValueChange={setName}
              placeholder="my-profile"
              sanitize={slug}
              value={name}
            />
            <FieldHint error={invalid}>{p.nameHint}</FieldHint>
          </Field>

          <Field htmlFor="new-profile-clone-from" label={p.cloneFrom}>
            <Select
              onValueChange={value => setCloneFrom(value === '__none__' ? null : value)}
              value={cloneFrom ?? '__none__'}
            >
              <SelectTrigger className="h-9 rounded-md" id="new-profile-clone-from">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__none__">{p.cloneFromNone}</SelectItem>
                {profiles.map(profile => (
                  <SelectItem key={profile.name} value={profile.name}>
                    {profile.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <FieldHint>{p.cloneFromDesc}</FieldHint>
          </Field>

          <Field htmlFor="new-profile-soul" label="SOUL.md" optional optionalLabel={p.soulOptional}>
            <Textarea
              className="min-h-28 font-mono text-xs leading-5"
              id="new-profile-soul"
              onChange={event => setSoul(event.target.value)}
              placeholder={p.soulPlaceholder(cloneFrom ? p.soulPlaceholderCloned : p.soulPlaceholderEmpty)}
              value={soul}
            />
          </Field>

          {conflict?.kind === 'near-miss' && (
            <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
              <div className="grid gap-1.5">
                <span>{p.createConflict.nearMiss(trimmed, conflict.name)}</span>
                <span>{p.createConflict.scaffoldNote}</span>
                <Button
                  className="justify-self-start"
                  onClick={() => onUseExisting(conflict.name)}
                  size="xs"
                  type="button"
                  variant="outline"
                >
                  {p.createConflict.useExisting(conflict.name)}
                </Button>
              </div>
            </div>
          )}

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
            <Button disabled={busy || !trimmed || invalid} type="submit">
              <ActionStatus
                busy={p.creating}
                done={p.created}
                idle={conflict?.kind === 'near-miss' ? p.createConflict.createAnyway : p.createAction}
                state={status}
              />
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
