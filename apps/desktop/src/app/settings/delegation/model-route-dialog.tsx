import { REASONING_EFFORTS } from '@hermes/shared'
import { type FormEvent, useState } from 'react'

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
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import type { DelegationModelRoute, ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { findCatalogProvider } from '@/lib/model-options'

import { displayProviders, useOwnerCatalog } from '../delegation-model-provider-field'
import { ModelSelect } from '../model-select'

import {
  draftFromRoute,
  emptyRouteDraft,
  type ModelRouteDraft,
  type RouteField,
  routeFromDraft,
  SERVICE_TIERS,
  validateRouteDraft
} from './routes'

// Radix Select cannot hold an empty value; this item stands for "no override".
const INHERIT = '__inherit__'

interface ModelRouteDialogProps {
  /** The route being edited; null creates a new one. */
  route: DelegationModelRoute | null
  /** Names of the other routes (a new name must not collide with them). */
  taken: readonly string[]
  scope: ProfileScope
  onClose: () => void
  onSave: (route: DelegationModelRoute) => Promise<boolean>
}

export function ModelRouteDialog({ route, taken, scope, onClose, onSave }: ModelRouteDialogProps) {
  const { t } = useI18n()
  const r = t.settings.model.subagents.routing
  const { providers } = useOwnerCatalog(scope)
  const [draft, setDraft] = useState<ModelRouteDraft>(() => (route ? draftFromRoute(route) : emptyRouteDraft()))
  const [submitted, setSubmitted] = useState(false)
  const [touched, setTouched] = useState<Partial<Record<RouteField, boolean>>>({})
  const [busy, setBusy] = useState(false)
  const errors = validateRouteDraft(draft, taken, Boolean(route?.base_url))
  const selected = findCatalogProvider(providers, draft.provider)

  const set = (field: RouteField, value: string) => setDraft(current => ({ ...current, [field]: value }))
  const touch = (field: RouteField) => setTouched(current => ({ ...current, [field]: true }))

  const errorFor = (field: RouteField) =>
    (submitted || touched[field]) && errors[field] ? r.errors[errors[field]] : null

  const chooseProvider = (slug: string) => {
    const models = findCatalogProvider(providers, slug)?.models ?? []

    // Never carry a model id over to a provider that does not list it.
    setDraft(current => ({ ...current, provider: slug, model: models.includes(current.model) ? current.model : '' }))
    touch('provider')
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault()
    setSubmitted(true)

    if (Object.keys(errors).length > 0) {
      return
    }

    setBusy(true)
    const saved = await onSave(routeFromDraft(draft, route ?? undefined))
    setBusy(false)

    if (saved) {
      onClose()
    }
  }

  const tierOptions = [
    { id: INHERIT, label: r.effortInherit },
    { id: SERVICE_TIERS[0], label: r.tierNormal },
    { id: SERVICE_TIERS[1], label: r.tierFast },
    { id: SERVICE_TIERS[2], label: r.tierUltrafast }
  ]

  return (
    <Dialog onOpenChange={open => !open && !busy && onClose()} open>
      <DialogContent className="max-w-lg" data-slot="delegation-route-dialog">
        <DialogHeader>
          <DialogTitle>{route ? r.dialogEdit : r.dialogNew}</DialogTitle>
          <DialogDescription>{r.dialogDesc}</DialogDescription>
        </DialogHeader>

        <form className="grid gap-4" noValidate onSubmit={handleSubmit}>
          <Field htmlFor="delegation-route-name" label={r.nameLabel}>
            <Input
              aria-invalid={Boolean(errorFor('name'))}
              autoFocus={!route}
              className="font-mono sm:max-w-56"
              id="delegation-route-name"
              onBlur={() => touch('name')}
              onChange={event => set('name', event.target.value)}
              placeholder={r.namePlaceholder}
              value={draft.name}
            />
            <FieldHint error={Boolean(errorFor('name'))}>{errorFor('name') ?? r.nameHint}</FieldHint>
          </Field>
          <Field htmlFor="delegation-route-description" label={r.descriptionLabel}>
            <Textarea
              aria-invalid={Boolean(errorFor('description'))}
              className="min-h-14 resize-none"
              id="delegation-route-description"
              onBlur={() => touch('description')}
              onChange={event => set('description', event.target.value)}
              placeholder={r.descriptionPlaceholder}
              rows={2}
              value={draft.description}
            />
            <FieldHint error={Boolean(errorFor('description'))}>
              {errorFor('description') ?? r.descriptionHint}
            </FieldHint>
          </Field>

          <div className="grid items-start gap-4 sm:grid-cols-2">
            <Field htmlFor="delegation-route-provider" label={r.providerLabel}>
              <Select onValueChange={chooseProvider} value={draft.provider || undefined}>
                <SelectTrigger
                  aria-invalid={Boolean(errorFor('provider'))}
                  className="[&_[data-slot=select-value]]:truncate"
                  id="delegation-route-provider"
                >
                  <SelectValue placeholder={r.providerPlaceholder} />
                </SelectTrigger>
                <SelectContent>
                  {displayProviders(providers, draft.provider).map(provider => (
                    <SelectItem key={provider.slug} value={provider.slug}>
                      {provider.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {errorFor('provider') && <FieldHint error>{errorFor('provider')}</FieldHint>}
              {route?.base_url && !draft.provider && <FieldHint>{r.baseUrlRoute(route.base_url)}</FieldHint>}
            </Field>
            <Field htmlFor="delegation-route-model" label={r.modelLabel}>
              <div className="min-w-0" id="delegation-route-model">
                <ModelSelect
                  aria-invalid={Boolean(errorFor('model'))}
                  aria-label={r.modelLabel}
                  className="w-full [&_[data-slot=select-value]]:truncate"
                  models={selected?.models ?? []}
                  onValueChange={model => {
                    set('model', model)
                    touch('model')
                  }}
                  provider={selected}
                  providerSlug={draft.provider}
                  value={draft.model}
                />
              </div>
              {errorFor('model') && <FieldHint error>{errorFor('model')}</FieldHint>}
            </Field>
          </div>

          <div className="grid items-start gap-4 sm:grid-cols-2">
            <Field htmlFor="delegation-route-effort" label={r.effortLabel} optional optionalLabel={r.optional}>
              <Select
                onValueChange={value => set('reasoning_effort', value === INHERIT ? '' : value)}
                value={draft.reasoning_effort || INHERIT}
              >
                <SelectTrigger id="delegation-route-effort">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={INHERIT}>{r.effortInherit}</SelectItem>
                  {['none', ...REASONING_EFFORTS].map(effort => (
                    <SelectItem key={effort} value={effort}>
                      {effort}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
            <Field htmlFor="delegation-route-tier" label={r.tierLabel} optional optionalLabel={r.optional}>
              <Select
                onValueChange={value => set('service_tier', value === INHERIT ? '' : value)}
                value={draft.service_tier || INHERIT}
              >
                <SelectTrigger id="delegation-route-tier">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {tierOptions.map(option => (
                    <SelectItem key={option.id} value={option.id}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <FieldHint>{r.tierHint}</FieldHint>
            </Field>
          </div>

          <DialogFooter>
            <Button disabled={busy} onClick={onClose} type="button" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button disabled={busy} type="submit">
              {route ? r.saveRoute : r.createRoute}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
