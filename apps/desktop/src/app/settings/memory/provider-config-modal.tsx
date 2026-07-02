import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { saveMemoryProviderConfig } from '@/hermes'
import { ExternalLink, Loader2, Save, SlidersHorizontal } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'
import type { MemoryProviderConfig, MemoryProviderField } from '@/types/hermes'

import { FieldControl } from './field-control'
import { ListRow } from '../primitives'

/** Seed every editable field (inline and modal-only). Secrets start blank —
 *  their value is never returned and submitting blank keeps the stored one. */
function seedAll(config: MemoryProviderConfig): Record<string, string> {
  return Object.fromEntries(config.fields.map(field => [field.key, field.kind === 'secret' ? '' : field.value]))
}

/** Group fields in declared order, preserving first-seen group sequence. */
function groupFields(fields: MemoryProviderField[]): [string, MemoryProviderField[]][] {
  const groups: [string, MemoryProviderField[]][] = []
  for (const field of fields) {
    const name = field.group || 'Other'
    const bucket = groups.find(([key]) => key === name)
    if (bucket) {
      bucket[1].push(field)
    } else {
      groups.push([name, [field]])
    }
  }
  return groups
}

export function ProviderConfigModal({
  config,
  provider,
  open,
  onOpenChange,
  onSaved
}: {
  config: MemoryProviderConfig
  provider: string
  open: boolean
  onOpenChange: (open: boolean) => void
  onSaved: () => Promise<void> | void
}) {
  const [values, setValues] = useState<Record<string, string>>({})
  const [seeded, setSeeded] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState(false)

  // Reseed from the latest config each time the dialog opens so edits never
  // start from a stale snapshot left over from a prior session.
  useEffect(() => {
    if (open) {
      const seed = seedAll(config)
      setSeeded(seed)
      setValues(seed)
    }
  }, [open, config])

  const save = async () => {
    // Unstored fields render their schema default; persisting untouched keys
    // would pin values that runtime defaults (e.g. migration guards) still own.
    const edited = Object.fromEntries(Object.entries(values).filter(([key, value]) => value !== seeded[key]))

    setSaving(true)
    try {
      await saveMemoryProviderConfig(provider, edited)
      notify({ kind: 'success', title: `${config.label} saved`, message: 'Memory provider configuration updated.' })
      await onSaved()
      onOpenChange(false)
    } catch (err) {
      notifyError(err, `Failed to save ${config.label} settings`)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-2xl dt-portal-scrollbar">
        <DialogHeader>
          <DialogTitle icon={SlidersHorizontal}>{config.label} — full configuration</DialogTitle>
          <DialogDescription>
            Every {config.label} option for the active profile. Blank fields fall back to the resolved host or
            built-in default.
          </DialogDescription>
          {config.docs_url && (
            <a
              className="inline-flex w-fit items-center gap-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-accent-secondary) underline-offset-4 transition-colors hover:underline"
              href={config.docs_url}
              onClick={event => {
                event.preventDefault()
                void window.hermesDesktop?.openExternal?.(config.docs_url)
              }}
              rel="noreferrer"
              target="_blank"
            >
              {config.label} configuration reference
              <ExternalLink className="size-3" />
            </a>
          )}
        </DialogHeader>

        <div className="min-w-0">
          {groupFields(config.fields).map(([group, fields]) => (
            <section className="mt-6 first:mt-2" key={group}>
              <h3 className="border-b border-(--ui-accent-secondary)/30 pb-1.5 font-mono text-[0.68rem] uppercase tracking-wide text-(--ui-accent-secondary)">
                {group}
              </h3>
              <div className="pl-1">
                {fields.map(field => (
                  <div className="border-b border-border/40 last:border-b-0" key={field.key}>
                    <ListRow
                      action={
                        <FieldControl
                          field={field}
                          onChange={value => setValues(current => ({ ...current, [field.key]: value }))}
                          value={values[field.key] ?? ''}
                        />
                      }
                      description={field.description}
                      title={field.label}
                    />
                  </div>
                ))}
              </div>
            </section>
          ))}
        </div>

        <DialogFooter>
          <DialogClose asChild>
            <Button size="sm" type="button" variant="ghost">
              Cancel
            </Button>
          </DialogClose>
          <Button disabled={saving} onClick={() => void save()} size="sm">
            {saving ? <Loader2 className="size-3.5 animate-spin" /> : <Save />}
            Save changes
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
