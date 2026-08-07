import { useQuery, useQueryClient } from '@tanstack/react-query'
import { type FormEvent, useEffect, useState } from 'react'

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
import { getGlobalModelInfo, getHermesConfigRecord, saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import { compactNumber } from '@/lib/format'
import {
  type ContextWindowInfo,
  effectiveContextLength,
  hasContextOverride,
  parseContextLengthInput
} from '@/store/context-window'

import { InlineNotice } from './notifications'

export const contextWindowQueryKey = (profile: string) => ['context-window', profile] as const

/**
 * Fetch the context-window figures from the backend.
 *
 * Deliberately the SAME `/api/model/info` endpoint the Settings page uses:
 * its `auto_context_length` comes from `get_model_context_length()`, the
 * provider-aware resolver that knows Codex OAuth / Copilot / Nous limits. A
 * models.dev value read client-side would disagree with the CLI's `/model`
 * output for the same model.
 */
async function fetchContextWindow(): Promise<ContextWindowInfo> {
  const info = await getGlobalModelInfo()

  return {
    autoContextLength: info.auto_context_length ?? 0,
    configContextLength: info.config_context_length ?? 0,
    effectiveContextLength: info.effective_context_length ?? 0,
    model: info.model ?? '',
    provider: info.provider ?? ''
  }
}

/**
 * Persist the override through the existing config surface.
 *
 * `model_context_length` is the virtual top-level field the backend flattens
 * out of `model.context_length`; `0` means "auto-detect". The `model` key must
 * ride along because the backend's denormalizer only folds the override back
 * into the model dict when it can see the model it belongs to.
 */
async function saveContextOverride(tokens: number): Promise<void> {
  const record = await getHermesConfigRecord()

  await saveHermesConfig({ ...record, model_context_length: tokens })
}

interface ContextWindowDialogProps {
  onOpenChange: (open: boolean) => void
  open: boolean
  profile?: string
}

export function ContextWindowDialog({ onOpenChange, open, profile = 'default' }: ContextWindowDialogProps) {
  const { t } = useI18n()
  const copy = t.contextWindow
  const queryClient = useQueryClient()

  // `null` means "the user hasn't edited the field yet", so the displayed
  // value falls back to whatever the backend reports. Modelling the draft this
  // way (instead of seeding state from an effect) means a slow
  // `/api/model/info` response can never land on top of what the user is
  // typing — the fetch resolves after the field is already interactive.
  const [draft, setDraft] = useState<string | null>(null)
  const [invalid, setInvalid] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const info = useQuery({
    queryKey: contextWindowQueryKey(profile),
    queryFn: fetchContextWindow,
    enabled: open
  })

  const data = info.data

  // Discard any unsaved edit when the dialog is dismissed, so reopening shows
  // the persisted pin rather than a stale draft.
  useEffect(() => {
    if (open) {
      return
    }

    setDraft(null)
    setInvalid(false)
    setSaveError(null)
  }, [open])

  // What the field shows: the user's edit if there is one, else the pinned
  // override, else blank (blank == auto-detect).
  const pinnedText = data && hasContextOverride(data) ? String(data.configContextLength) : ''
  const fieldValue = draft ?? pinnedText

  const commit = async (tokens: number) => {
    setSaving(true)
    setSaveError(null)

    try {
      await saveContextOverride(tokens)
      await queryClient.invalidateQueries({ queryKey: contextWindowQueryKey(profile) })
      onOpenChange(false)
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err))
    } finally {
      setSaving(false)
    }
  }

  const submit = (event: FormEvent) => {
    event.preventDefault()

    const tokens = parseContextLengthInput(fieldValue)

    if (tokens === null) {
      setInvalid(true)

      return
    }

    setInvalid(false)
    void commit(tokens)
  }

  const auto = data?.autoContextLength ?? 0
  const pinned = data ? hasContextOverride(data) : false
  const effective = data ? effectiveContextLength(data) : 0

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md gap-0 overflow-hidden p-0">
        <DialogHeader className="border-b border-border px-4 py-3">
          <DialogTitle>{copy.title}</DialogTitle>
          <DialogDescription className="text-xs leading-relaxed">{copy.description}</DialogDescription>
        </DialogHeader>

        <form className="flex flex-col gap-3 bg-card px-4 py-4" onSubmit={submit}>
          <div className="flex flex-col gap-1 font-mono text-xs text-muted-foreground">
            <span>
              {copy.effective}: <span className="text-foreground">{effective ? compactNumber(effective) : copy.unknown}</span>
              {!pinned && effective ? ` · ${copy.usingAuto}` : ''}
            </span>
            <span>{copy.autoDetected(auto ? compactNumber(auto) : copy.unknown)}</span>
            {data?.model ? <span>{copy.forModel(data.model, data.provider || copy.unknown)}</span> : null}
          </div>

          <label className="flex flex-col gap-1.5 text-xs" htmlFor="context-window-override">
            <span className="text-muted-foreground">{copy.overrideLabel}</span>
            <Input
              autoFocus
              disabled={saving}
              id="context-window-override"
              inputMode="numeric"
              onChange={event => setDraft(event.target.value)}
              placeholder={copy.placeholder}
              value={fieldValue}
            />
          </label>

          {invalid && <InlineNotice kind="warning">{copy.invalid}</InlineNotice>}
          {saveError && (
            <InlineNotice kind="error" title={copy.saveFailed}>
              {saveError}
            </InlineNotice>
          )}

          {/* The pin is route-scoped: `should_clear_context_pin` drops it
              fail-closed when the model/provider/base_url no longer match. */}
          <p className="text-[0.68rem] leading-relaxed text-muted-foreground">{copy.routeScoped}</p>
          {/* Config is adopted at the start of the next turn; nothing about the
              live conversation's cached prefix is rebuilt. */}
          <p className="text-[0.68rem] leading-relaxed text-muted-foreground">{copy.nextTurnNotice}</p>

          <DialogFooter className="flex-row items-center justify-end gap-2 p-0 pt-1">
            <Button disabled={saving || !pinned} onClick={() => void commit(0)} type="button" variant="ghost">
              {copy.useAuto}
            </Button>
            <Button disabled={saving} type="submit">
              {t.common.save}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
