import { useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'

import { ListRow, Pill } from '../primitives'

import { RowValue } from './account-row-value'
import type { BillingRefusal } from './api'
import { useBillingApi } from './api'
import { initialAutoReloadAmount, validateAutoReloadInputs } from './billing-amounts'
import { BillingRefusalInline } from './inline-feedback'
import type { BillingAutoReload, BillingStateResponse } from './types'
import type { BillingAccountRowView } from './use-billing-state'

export function AutoReloadRow({
  autoReload,
  bounds,
  row
}: {
  autoReload: BillingAutoReload
  bounds: Pick<BillingStateResponse, 'max_usd' | 'min_usd'>
  row: BillingAccountRowView
}) {
  const api = useBillingApi()
  const queryClient = useQueryClient()
  const [confirmDisable, setConfirmDisable] = useState(false)
  const [editing, setEditing] = useState(false)
  // Validation errors are silent until the user edits a field or attempts a
  // save — opening Manage on a prefilled (possibly below-min) config must not
  // flash an error (spec §9).
  const [showErrors, setShowErrors] = useState(false)
  const [message, setMessage] = useState<null | { kind: 'error' | 'success'; text: string }>(null)
  const [refusal, setRefusal] = useState<BillingRefusal | null>(null)

  const [reloadTo, setReloadTo] = useState(
    initialAutoReloadAmount(autoReload.reload_to_usd, autoReload.reload_to_display)
  )

  const [saving, setSaving] = useState(false)

  const [threshold, setThreshold] = useState(
    initialAutoReloadAmount(autoReload.threshold_usd, autoReload.threshold_display)
  )

  const validation = validateAutoReloadInputs(threshold, reloadTo, bounds)
  const busy = saving
  const maxBound = bounds.max_usd ?? undefined
  const minBound = bounds.min_usd ?? undefined

  // Only the canonical-card enabled state edits in place (flagged in the view model).
  // Off / divergent-card rows have no Manage affordance (or a portal link) and render
  // read-only.
  const editable = row.manageInApp === true

  const resetFeedback = () => {
    setConfirmDisable(false)
    setMessage(null)
    setRefusal(null)
  }

  const openEdit = () => {
    resetFeedback()
    setShowErrors(false)
    setEditing(true)
  }

  const cancelEdit = () => {
    resetFeedback()
    setEditing(false)
  }

  const save = async () => {
    if (!validation.values || busy) {
      return
    }

    resetFeedback()
    setSaving(true)

    const result = await api.updateAutoReload({
      enabled: true,
      reload_to_usd: validation.values.reloadTo,
      threshold_usd: validation.values.threshold
    })

    setSaving(false)

    if (!result.ok) {
      setRefusal(result.refusal)

      return
    }

    await queryClient.invalidateQueries({ queryKey: ['billing', 'state'] })
    setMessage({ kind: 'success', text: 'Auto-refill updated.' })
    setEditing(false)
  }

  const disable = async () => {
    if (busy) {
      return
    }

    resetFeedback()
    setSaving(true)

    // The gateway's billing.auto_reload handler unconditionally requires threshold
    // + top_up_amount (invalid_request otherwise), so a disable must still carry the
    // current amounts — mirroring the TUI, which always sends both.
    const result = await api.updateAutoReload({
      enabled: false,
      reload_to_usd: initialAutoReloadAmount(autoReload.reload_to_usd, autoReload.reload_to_display),
      threshold_usd: initialAutoReloadAmount(autoReload.threshold_usd, autoReload.threshold_display)
    })

    setSaving(false)

    if (!result.ok) {
      setRefusal(result.refusal)

      return
    }

    await queryClient.invalidateQueries({ queryKey: ['billing', 'state'] })
    setMessage({ kind: 'success', text: 'Auto-refill turned off.' })
    setEditing(false)
  }

  // Read-only states (off / divergent card) keep the original ListRow shape.
  if (!editable) {
    return (
      <ListRow
        action={<RowValue row={row} />}
        below={
          <>
            {row.caption ? (
              <div className="mt-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                {row.caption}
              </div>
            ) : null}
            <BillingRefusalInline refusal={refusal} />
            {message && <InlineMessage kind={message.kind}>{message.text}</InlineMessage>}
          </>
        }
        description={row.description}
        key={row.id}
        title={row.title}
      />
    )
  }

  const onField = (setter: (value: string) => void) => (event: { target: { value: string } }) => {
    resetFeedback()
    setShowErrors(true)
    setter(event.target.value)
  }

  // Zero-shift by exact reservation, not a magic min-height: the edit form is
  // ALWAYS rendered and both states share a single grid cell (`[grid-area:stack]`),
  // so the row's height always equals the tallest state at EVERY container width —
  // no breakpoint math that under-reserves when the two inputs stack on narrow
  // panes. The form is `invisible` + `aria-hidden` when not editing.
  return (
    <div className="@container">
      <div className="grid gap-3 py-3 @2xl:grid-cols-[minmax(0,1fr)_minmax(15rem,22rem)] @2xl:items-start">
        <div className="min-w-0">
          <div className="text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
            {row.title}
          </div>
          <div className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
            {row.description}
          </div>
          <div className="mt-3 grid [grid-template-areas:'stack']">
            {/* EDIT layer — always in layout (reserves exact height); hidden until editing. */}
            <div
              aria-hidden={!editing}
              className={cn('space-y-2 [grid-area:stack]', !editing && 'invisible')}
            >
              <div className="grid gap-2 @2xl:grid-cols-2">
                <label className="min-w-0 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                  Threshold
                  <Input
                    aria-label="Auto-refill threshold"
                    className="mt-1 py-[3px]"
                    disabled={busy || !editing}
                    inputMode="decimal"
                    max={maxBound}
                    min={minBound}
                    onChange={onField(setThreshold)}
                    size="sm"
                    step="0.01"
                    tabIndex={editing ? undefined : -1}
                    type="number"
                    value={threshold}
                  />
                </label>
                <label className="min-w-0 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                  Reload to
                  <Input
                    aria-label="Auto-refill reload-to amount"
                    className="mt-1 py-[3px]"
                    disabled={busy || !editing}
                    inputMode="decimal"
                    max={maxBound}
                    min={minBound}
                    onChange={onField(setReloadTo)}
                    size="sm"
                    step="0.01"
                    tabIndex={editing ? undefined : -1}
                    type="number"
                    value={reloadTo}
                  />
                </label>
              </div>
              {/* Pre-allocated error line — occupies height whether or not shown. */}
              <div className="min-h-4 text-[length:var(--conversation-caption-font-size)] text-destructive">
                {showErrors && validation.error ? validation.error : ''}
              </div>
              {confirmDisable ? (
                <div className="flex min-w-0 flex-wrap items-center gap-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                  <span>Turn off auto-refill?</span>
                  <Button disabled={busy} onClick={() => void disable()} size="sm" type="button" variant="outline">
                    Turn off
                  </Button>
                  <Button disabled={busy} onClick={() => setConfirmDisable(false)} size="sm" type="button" variant="ghost">
                    Cancel
                  </Button>
                </div>
              ) : (
                <Button
                  disabled={busy}
                  onClick={() => setConfirmDisable(true)}
                  size="sm"
                  tabIndex={editing ? undefined : -1}
                  type="button"
                  variant="outline"
                >
                  Disable
                </Button>
              )}
              {/* Refusal stays INSIDE the reserved layer so it never pushes Usage. */}
              <BillingRefusalInline refusal={refusal} />
            </div>
            {/* VIEW layer — success feedback overlaid in the same cell when not editing. */}
            {!editing && message && (
              <div className="[grid-area:stack]">
                <InlineMessage kind={message.kind}>{message.text}</InlineMessage>
              </div>
            )}
          </div>
        </div>
        {/* Action column swaps Manage ↔ Save/Cancel in place (top-aligned, no move). */}
        <div className="flex min-w-0 flex-wrap items-center justify-start gap-2 @2xl:justify-end">
          {row.pill && <Pill tone={row.pill.tone}>{row.pill.label}</Pill>}
          {editing ? (
            <>
              <Button disabled={busy || !validation.values} onClick={() => void save()} size="sm" type="button">
                {busy ? 'Saving…' : 'Save'}
              </Button>
              <Button disabled={busy} onClick={cancelEdit} size="sm" type="button" variant="outline">
                Cancel
              </Button>
            </>
          ) : (
            <Button onClick={openEdit} size="sm" type="button" variant="outline">
              Manage
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}

// A one-line success/error note under the row — the only consumer of this shape.
function InlineMessage({ children, kind }: { children: string; kind: 'error' | 'success' }) {
  return (
    <div
      className={cn(
        'mt-2 text-[length:var(--conversation-caption-font-size)]',
        kind === 'error' ? 'text-destructive' : 'text-(--ui-text-tertiary)'
      )}
    >
      {children}
    </div>
  )
}
