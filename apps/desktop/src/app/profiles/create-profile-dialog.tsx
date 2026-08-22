import { useEffect, useState } from 'react'

import { ActionStatus } from '@/components/ui/action-status'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
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
import { PROFILE_GLYPHS } from '@/lib/profile-glyphs'
import { slug } from '@/lib/sanitize'
import { cn } from '@/lib/utils'
import { setProfileGlyph } from '@/store/profile'
import type { ProfileInfo } from '@/types/hermes'

const PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/

export function isValidProfileName(name: string): boolean {
  return PROFILE_NAME_RE.test(name.trim())
}

// Self-contained create flow (name + glyph + clone toggle + optional SOUL.md).
// Owns the createProfile/updateProfileSoul calls so every caller just
// refreshes/selects via onCreated. SOUL left blank keeps the cloned/blank
// persona untouched; the glyph stays auto (initial / home mark) when unset.
export function CreateProfileDialog({
  onClose,
  onCreated,
  open,
  profiles = []
}: {
  onClose: () => void
  onCreated?: (name: string) => Promise<void> | void
  open: boolean
  profiles?: ProfileInfo[]
}) {
  const { t } = useI18n()
  const p = t.profiles
  const [name, setName] = useState('')
  const [cloneFrom, setCloneFrom] = useState<null | string>('default')
  const [glyph, setGlyph] = useState<null | string>(null)
  const [soul, setSoul] = useState('')
  const [status, setStatus] = useState<'done' | 'idle' | 'saving'>('idle')
  const [error, setError] = useState<null | string>(null)

  useEffect(() => {
    if (!open) {
      return
    }

    setName('')
    setCloneFrom('default')
    setGlyph(null)
    setSoul('')
    setError(null)
    setStatus('idle')
  }, [open])

  const trimmed = name.trim()
  const invalid = trimmed !== '' && !isValidProfileName(trimmed)
  const busy = status === 'saving' || status === 'done'

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()

    if (!trimmed || invalid) {
      setError(invalid ? p.invalidName(p.nameHint) : p.nameRequired)

      return
    }

    setStatus('saving')
    setError(null)

    try {
      await createProfile({ name: trimmed, clone_from: cloneFrom })

      // Persist the picked glyph before onCreated so the rail's refresh already
      // paints the chosen mark instead of flashing the auto one.
      setProfileGlyph(trimmed, glyph)

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

          <Field label={p.glyphLabel} optional optionalLabel={p.soulOptional}>
            <div aria-label={p.glyphLabel} className="flex flex-wrap gap-1" role="group">
              <button
                aria-label={p.glyphAuto}
                aria-pressed={!glyph}
                className={cn(
                  'grid size-6 place-items-center rounded-md text-(--ui-text-tertiary) transition hover:bg-(--ui-control-hover-background) hover:text-foreground',
                  !glyph && 'bg-(--ui-control-active-background) text-foreground'
                )}
                onClick={() => setGlyph(null)}
                title={p.glyphAuto}
                type="button"
              >
                <Codicon name="sync" size="0.875rem" />
              </button>
              {PROFILE_GLYPHS.map(id => (
                <button
                  aria-label={id}
                  aria-pressed={glyph === id}
                  className={cn(
                    'grid size-6 place-items-center rounded-md text-(--ui-text-secondary) transition hover:bg-(--ui-control-hover-background) hover:text-foreground',
                    glyph === id && 'bg-(--ui-control-active-background) text-foreground'
                  )}
                  key={id}
                  onClick={() => setGlyph(id)}
                  title={id}
                  type="button"
                >
                  <Codicon name={id} size="0.875rem" />
                </button>
              ))}
            </div>
            <FieldHint>{p.glyphHint}</FieldHint>
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
              <ActionStatus busy={p.creating} done={p.created} idle={p.createAction} state={status} />
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
