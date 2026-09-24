import { useQuery } from '@tanstack/react-query'
import { useRef, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
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
import { ErrorState } from '@/components/ui/error-state'
import { getMemoryProviderConfig, type OwnerScope, saveMemoryProviderConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import type { MemoryProviderConfig, MemoryProviderField } from '@/types/hermes'

import { ListRow } from '../primitives'

import { FieldControl, FieldTitle } from './field-control'
import { ownerKey } from './owner'

export const providerConfigQueryKey = (owner: OwnerScope, provider: string) =>
  ['memory-provider-config', ownerKey(owner), provider] as const

const BOOL_WORDS: Record<string, boolean> = {
  true: true,
  1: true,
  yes: true,
  on: true,
  false: false,
  0: false,
  no: false,
  off: false
}

const DECIMAL = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i

// Normalizes a form value or `when` operand by the dependency's kind; text compares against Python spellings.
function conditionValue(field: MemoryProviderField | undefined, value: unknown): unknown {
  const kind = field?.condition_kind ?? field?.kind

  if (kind === 'bool') {
    if (typeof value === 'boolean') {
      return value
    }

    if (value === null || value === '') {
      return false
    }

    return typeof value === 'number' ? value !== 0 : BOOL_WORDS[String(value).trim().toLowerCase()]
  }

  if (kind === 'number') {
    const decimal = typeof value === 'number' || (typeof value === 'string' && DECIMAL.test(value.trim()))
    const number = decimal ? Number(value) : NaN

    return Number.isFinite(number) ? number : undefined
  }

  if (value === null) {
    return 'None'
  }

  if (typeof value === 'boolean') {
    return value ? 'True' : 'False'
  }

  return typeof value === 'string' || typeof value === 'number' ? String(value) : undefined
}

function visibleFields(config: MemoryProviderConfig, fields: MemoryProviderField[], values: Record<string, string>) {
  const byKey = new Map(config.fields.map(field => [field.key, field]))

  const effective = new Map(
    config.fields.map(field => {
      const value = values[field.key] === '' ? (field.default === undefined ? '' : field.default) : values[field.key]

      return [field.key, field.kind === 'secret' ? '' : conditionValue(field, value)]
    })
  )

  return fields.filter(field =>
    Object.entries(field.when ?? {}).every(([key, expected]) => {
      const actual = effective.get(key)

      return actual !== undefined && actual === conditionValue(byKey.get(key), expected)
    })
  )
}

// Secrets never arrive from the backend; a blank secret keeps the stored one.
const stored = (field: MemoryProviderField) => (field.kind === 'secret' ? '' : field.value)

interface ProviderConfigFormProps {
  config: MemoryProviderConfig
  provider: string
  owner: OwnerScope
  full?: boolean
  onSaved?: () => void
}

function ProviderConfigForm({ config, provider, owner, full = false, onSaved }: ProviderConfigFormProps) {
  const { t } = useI18n()
  const c = t.memoryProviders
  const [edits, setEdits] = useState<Record<string, string>>({})
  const [seen, setSeen] = useState(config)
  const [saving, setSaving] = useState(false)
  const [result, setResult] = useState<'saved' | 'failed' | null>(null)
  const form = useRef<HTMLFormElement>(null)
  const capabilities = config.capabilities ?? {}
  const requiresFullForm = capabilities.requires_full_form === true

  // A refetched config makes the edits it now stores clean; hidden drafts that were never sent stay.
  if (seen !== config) {
    setSeen(config)
    setEdits(current =>
      Object.fromEntries(
        Object.entries(current).filter(([key, value]) => {
          const field = config.fields.find(candidate => candidate.key === key)

          return field && value !== stored(field)
        })
      )
    )
  }

  const notice =
    requiresFullForm && !full
      ? c.fullFormRequired
      : capabilities.supports_partial_updates === false && !requiresFullForm
        ? c.partialUpdatesUnsupported
        : null

  if (notice) {
    return <p className="py-3 text-sm text-muted-foreground">{notice}</p>
  }

  const values = Object.fromEntries(config.fields.map(field => [field.key, edits[field.key] ?? stored(field)]))

  const visible = visibleFields(
    config,
    config.fields.filter(field => full || field.inline),
    values
  )

  // A native writer replaces its configuration, so it receives the whole visible form; host storage takes the changed keys.
  const submitted = visible.filter(
    field =>
      (field.kind !== 'secret' || values[field.key].trim()) && (requiresFullForm || values[field.key] !== stored(field))
  )

  const canSubmit = requiresFullForm ? visible.length > 0 : submitted.length > 0

  async function save() {
    if (saving || !canSubmit || !form.current?.reportValidity()) {
      return
    }

    setSaving(true)
    setResult(null)

    try {
      const submission = Object.fromEntries(submitted.map(field => [field.key, values[field.key]]))
      const legacyActive = capabilities.save_without_activation !== true
      const response = await saveMemoryProviderConfig(provider, submission, owner, { legacyActive })

      if (!response.ok) {
        throw new Error('Save rejected')
      }

      const secrets = submitted.filter(field => field.kind === 'secret').map(field => [field.key, ''])
      setEdits(current => ({ ...current, ...Object.fromEntries(secrets) }))
      setResult('saved')
      onSaved?.()
    } catch {
      // Server error text may echo the submitted credentials, so it is never shown.
      setResult('failed')
    } finally {
      setSaving(false)
    }
  }

  const groups = new Map<string, MemoryProviderField[]>()

  for (const field of visible) {
    const group = full ? field.group || c.other : ''
    groups.set(group, [...(groups.get(group) ?? []), field])
  }

  return (
    <form
      className="grid gap-3"
      onSubmit={event => {
        event.preventDefault()
        void save()
      }}
      ref={form}
    >
      <fieldset disabled={saving}>
        {[...groups].map(([group, grouped]) => (
          <section key={group}>
            {group && <h3 className="mt-4 text-sm font-medium">{group}</h3>}
            {grouped.map(field => (
              <ListRow
                action={
                  <FieldControl
                    field={field}
                    onChange={value => setEdits(current => ({ ...current, [field.key]: value }))}
                    value={values[field.key]}
                  />
                }
                description={field.description}
                key={field.key}
                title={<FieldTitle field={field} />}
              />
            ))}
          </section>
        ))}
      </fieldset>
      {result === 'failed' && (
        <div role="alert">
          <ErrorState title={c.saveFailed} />
        </div>
      )}
      {result === 'saved' && (
        <p className="text-sm text-muted-foreground" role="status">
          {c.saved}
        </p>
      )}
      {visible.length > 0 && (
        <div>
          <Button disabled={saving || !canSubmit} size="sm" type="submit">
            {c.save}
          </Button>
        </div>
      )}
    </form>
  )
}

interface ProviderConfigPanelProps {
  owner: OwnerScope
  provider: string
  active: boolean
  onSaved?: () => void
}

export function ProviderConfigPanel(props: ProviderConfigPanelProps) {
  return <ProviderConfigPanelInner key={`${ownerKey(props.owner)}:${props.provider}`} {...props} />
}

function ProviderConfigPanelInner({ owner, provider, active, onSaved }: ProviderConfigPanelProps) {
  const { t } = useI18n()
  const c = t.memoryProviders
  const [showModal, setShowModal] = useState(false)

  const {
    data: config,
    isError,
    refetch
  } = useQuery({
    queryKey: providerConfigQueryKey(owner, provider),
    queryFn: () => getMemoryProviderConfig(provider, owner),
    staleTime: 0
  })

  if (!config) {
    return isError ? (
      <ErrorState title={c.configLoadFailed}>
        <Button onClick={() => void refetch()} size="sm">
          {c.retry}
        </Button>
      </ErrorState>
    ) : (
      <PageLoader className="min-h-24" label={c.configLoading} />
    )
  }

  // A backend without save-only writes would select an inactive provider by saving it.
  const blocked = !config.fields.length
    ? c.noSettings
    : !active && config.capabilities?.save_without_activation !== true
      ? c.upgrade
      : null

  if (blocked) {
    return <p className="py-3 text-sm text-muted-foreground">{blocked}</p>
  }

  const saved = () => {
    void refetch()
    onSaved?.()
  }

  return (
    <section className="py-3">
      <h3 className="text-sm font-medium">{config.label}</h3>
      <ProviderConfigForm config={config} onSaved={saved} owner={owner} provider={provider} />
      {(config.capabilities?.requires_full_form || config.fields.some(field => !field.inline)) && (
        <>
          <Button onClick={() => setShowModal(true)} size="sm" variant="secondary">
            {c.fullConfig}
          </Button>
          <Dialog onOpenChange={setShowModal} open={showModal}>
            <DialogContent bodyClassName="dt-portal-scrollbar" className="max-w-2xl">
              <DialogHeader>
                <DialogTitle>
                  {config.label} — {c.fullConfig}
                </DialogTitle>
                <DialogDescription>{c.configDescription}</DialogDescription>
                {config.capabilities?.requires_full_form && (
                  <p className="text-sm text-muted-foreground">{c.nativeSetup}</p>
                )}
                {config.docs_url && (
                  <a
                    className="text-sm underline"
                    href={config.docs_url}
                    onClick={event => {
                      event.preventDefault()
                      void window.hermesDesktop?.openExternal?.(config.docs_url)
                    }}
                  >
                    {c.reference}
                  </a>
                )}
              </DialogHeader>
              <ProviderConfigForm
                config={config}
                full
                onSaved={() => {
                  saved()
                  setShowModal(false)
                }}
                owner={owner}
                provider={provider}
              />
              <DialogFooter>
                <DialogClose asChild>
                  <Button size="sm" variant="ghost">
                    {t.common.cancel}
                  </Button>
                </DialogClose>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </>
      )}
    </section>
  )
}
