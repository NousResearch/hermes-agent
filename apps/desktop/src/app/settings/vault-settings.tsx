import { useStore } from '@nanostores/react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { EmptyState } from '@/components/ui/empty-state'
import { Field } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Plus, ShieldLock, Trash2 } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'
import { $gatewayState } from '@/store/session'

import { CONTROL_TEXT } from './constants'
import { ListRow, Pill, SectionHeading, SettingsContent } from './primitives'

const VAULT_QUERY_KEY = ['vault-items'] as const

export type VaultKind = 'address' | 'login' | 'payment'
const VAULT_KINDS: readonly VaultKind[] = ['login', 'payment', 'address']
const IDENTIFIER_TYPES = ['email', 'phone', 'username'] as const
type IdentifierType = (typeof IDENTIFIER_TYPES)[number]

interface VaultItem {
  id: string
  kind: string
  label: string
  origin: null | string
  created_at: string
  identifier?: null | string
  identifier_type?: null | string
}

/** Add-dialog prefill from a deep link (`/settings?tab=vault&kind=…`). NEVER secrets. */
export interface VaultPrefill {
  kind?: string
  label?: string
  origin?: string
}

function isVaultKind(value: string | undefined): value is VaultKind {
  return !!value && (VAULT_KINDS as readonly string[]).includes(value)
}

function isValidOrigin(value: string): boolean {
  try {
    const url = new URL(value)

    return (url.protocol === 'https:' || url.protocol === 'http:') && !!url.hostname
  } catch {
    return false
  }
}

const EMPTY_FORM = {
  kind: 'login' as VaultKind,
  label: '',
  origin: '',
  identifierType: 'email' as IdentifierType,
  identifier: '',
  password: '',
  cardNumber: '',
  cardName: '',
  expMonth: '',
  expYear: '',
  cvc: '',
  postal: '',
  line1: '',
  line2: '',
  city: '',
  state: '',
  country: ''
}

type VaultForm = typeof EMPTY_FORM

function buildSecret(form: VaultForm): Record<string, string> {
  if (form.kind === 'login') {
    // identifier_type/identifier are stored as agent-visible metadata by the
    // vault store; only the password stays in the encrypted secret payload.
    return {
      identifier_type: form.identifierType,
      identifier: form.identifier.trim(),
      password: form.password
    }
  }

  if (form.kind === 'payment') {
    return {
      card_number: form.cardNumber.replace(/\s+/g, ''),
      cardholder_name: form.cardName.trim(),
      exp_month: form.expMonth.trim(),
      exp_year: form.expYear.trim(),
      cvc: form.cvc,
      billing_postal_code: form.postal.trim()
    }
  }

  const secret: Record<string, string> = {
    address_line1: form.line1.trim(),
    city: form.city.trim(),
    postal_code: form.postal.trim(),
    country: form.country.trim()
  }

  if (form.line2.trim()) {
    secret.address_line2 = form.line2.trim()
  }

  if (form.state.trim()) {
    secret.state = form.state.trim()
  }

  return secret
}

export function VaultSettings() {
  const { t } = useI18n()
  const v = t.settings.vault
  const { requestGateway } = useGatewayRequest()
  const gatewayState = useStore($gatewayState)
  const queryClient = useQueryClient()
  const [searchParams, setSearchParams] = useSearchParams()

  const [addOpen, setAddOpen] = useState(false)
  const [form, setForm] = useState<VaultForm>(EMPTY_FORM)
  const [formError, setFormError] = useState<null | string>(null)
  const [pendingDelete, setPendingDelete] = useState<null | VaultItem>(null)

  const { data, error, isPending } = useQuery({
    enabled: gatewayState === 'open',
    queryKey: VAULT_QUERY_KEY,
    queryFn: async () => {
      const result = await requestGateway<{ items: VaultItem[] }>('vault.list', {})

      return result.items
    }
  })

  useEffect(() => {
    if (error) {
      notifyError(error, v.loadFailed)
    }
  }, [error, v.loadFailed])

  const items = useMemo(() => data ?? [], [data])

  // Clears the secret fields with the rest of the form — the password/CVC
  // never outlive the dialog.
  const closeAdd = useCallback(() => {
    setAddOpen(false)
    setForm(EMPTY_FORM)
    setFormError(null)
  }, [])

  const openAdd = useCallback((prefill?: VaultPrefill) => {
    setForm({
      ...EMPTY_FORM,
      kind: isVaultKind(prefill?.kind) ? prefill.kind : 'login',
      label: prefill?.label ?? '',
      origin: prefill?.origin ?? ''
    })
    setFormError(null)
    setAddOpen(true)
  }, [])

  // Deep link (`hermes://open/settings?tab=vault&kind=login&label=…&origin=…`,
  // e.g. relayed by the agent when a login is missing): open the Add dialog
  // pre-filled from the query params — metadata only, never a secret — then
  // drop the params so a refresh doesn't re-open it.
  useEffect(() => {
    const kind = searchParams.get('kind') ?? undefined
    const label = searchParams.get('label') ?? undefined
    const origin = searchParams.get('origin') ?? undefined

    if (!kind && !label && !origin) {
      return
    }

    openAdd({ kind, label, origin })
    const next = new URLSearchParams(searchParams)
    next.delete('kind')
    next.delete('label')
    next.delete('origin')
    setSearchParams(next, { replace: true })
  }, [openAdd, searchParams, setSearchParams])

  const invalidate = useCallback(
    () => queryClient.invalidateQueries({ queryKey: VAULT_QUERY_KEY }),
    [queryClient]
  )

  const addMutation = useMutation({
    mutationFn: async (payload: { kind: VaultKind; label: string; origin?: string; secret: Record<string, string> }) =>
      requestGateway<{ id: string }>('vault.add', payload),
    onSuccess: () => {
      triggerHaptic('success')
      notify({ kind: 'info', message: v.added })
      closeAdd()
      void invalidate()
    },
    onError: err => {
      setFormError(String(err instanceof Error ? err.message : err))
    }
  })

  const submitAdd = useCallback(() => {
    setFormError(null)

    if (!form.label.trim()) {
      setFormError(v.labelRequired)

      return
    }

    const needsOrigin = form.kind === 'login'
    const origin = form.origin.trim()

    if (needsOrigin && !isValidOrigin(origin)) {
      setFormError(v.originInvalid)

      return
    }

    if (!needsOrigin && origin && !isValidOrigin(origin)) {
      setFormError(v.originInvalid)

      return
    }

    if (form.kind === 'login' && (!form.identifier.trim() || !form.password)) {
      setFormError(v.loginFieldsRequired)

      return
    }

    addMutation.mutate({
      kind: form.kind,
      label: form.label.trim(),
      ...(origin ? { origin } : {}),
      secret: buildSecret(form)
    })
  }, [addMutation, form, v.labelRequired, v.loginFieldsRequired, v.originInvalid])

  const deleteItem = useCallback(
    async (item: VaultItem) => {
      await requestGateway<{ removed: boolean }>('vault.remove', { id: item.id })
      triggerHaptic('success')
      void invalidate()
    },
    [invalidate, requestGateway]
  )

  const kindLabel = useCallback((kind: string) => v.kinds[kind as VaultKind] ?? kind, [v.kinds])

  const formatCreated = useCallback((iso: string) => {
    const parsed = new Date(iso)

    return Number.isNaN(parsed.getTime()) ? iso : parsed.toLocaleDateString()
  }, [])

  return (
    <SettingsContent>
      <SectionHeading
        aside={
          <Button className="gap-1.5" onClick={() => openAdd()} size="sm" type="button" variant="outline">
            <Plus className="size-3.5" />
            {v.add}
          </Button>
        }
        icon={ShieldLock}
        meta={items.length > 0 ? v.count(items.length) : undefined}
        title={v.title}
      />
      <p className="mb-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {v.blurb}
      </p>

      {!isPending && items.length === 0 && <EmptyState description={v.emptyDesc} title={v.empty} />}

      {items.map(item => (
        <ListRow
          action={
            <Button
              aria-label={v.deleteAction}
              className="text-(--ui-text-tertiary) hover:text-destructive"
              onClick={() => setPendingDelete(item)}
              size="icon-sm"
              type="button"
              variant="ghost"
            >
              <Trash2 className="size-3.5" />
            </Button>
          }
          description={
            <span className="flex flex-wrap items-center gap-2">
              {item.identifier && <span className="truncate">{v.identifierShown(item.identifier)}</span>}
              {item.origin && <span className="truncate">{item.origin}</span>}
              <span>{v.createdOn(formatCreated(item.created_at))}</span>
            </span>
          }
          key={item.id}
          title={
            <span className="flex items-center gap-2">
              <span className="truncate">{item.label}</span>
              <Pill tone={item.kind === 'login' ? 'primary' : 'muted'}>{kindLabel(item.kind)}</Pill>
            </span>
          }
        />
      ))}

      {/* Add dialog */}
      <Dialog onOpenChange={open => !open && closeAdd()} open={addOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>{v.addTitle}</DialogTitle>
            <DialogDescription>{v.addDescription}</DialogDescription>
          </DialogHeader>

          <form
            className="grid gap-4"
            onSubmit={e => {
              e.preventDefault()
              submitAdd()
            }}
          >
            <div className="grid items-start gap-4 sm:grid-cols-2">
              <Field htmlFor="vault-kind" label={v.kindField}>
                <Select
                  onValueChange={value => setForm(f => ({ ...f, kind: value as VaultKind }))}
                  value={form.kind}
                >
                  <SelectTrigger className={CONTROL_TEXT} id="vault-kind">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {VAULT_KINDS.map(kind => (
                      <SelectItem key={kind} value={kind}>
                        {v.kinds[kind]}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
              <Field htmlFor="vault-label" label={v.labelField}>
                <Input
                  autoFocus
                  id="vault-label"
                  onChange={e => setForm(f => ({ ...f, label: e.target.value }))}
                  placeholder={v.labelPlaceholder}
                  value={form.label}
                />
              </Field>
            </div>

            {form.kind === 'login' && (
              <>
                <Field htmlFor="vault-origin" label={v.originField}>
                  <Input
                    id="vault-origin"
                    inputMode="url"
                    onChange={e => setForm(f => ({ ...f, origin: e.target.value }))}
                    placeholder={v.originPlaceholder}
                    value={form.origin}
                  />
                </Field>
                <div className="grid items-start gap-4 sm:grid-cols-2">
                  <Field htmlFor="vault-id-type" label={v.identifierTypeField}>
                    <Select
                      onValueChange={value => setForm(f => ({ ...f, identifierType: value as IdentifierType }))}
                      value={form.identifierType}
                    >
                      <SelectTrigger className={CONTROL_TEXT} id="vault-id-type">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {IDENTIFIER_TYPES.map(type => (
                          <SelectItem key={type} value={type}>
                            {v.identifierTypes[type]}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </Field>
                  <Field htmlFor="vault-identifier" label={v.identifierField}>
                    <Input
                      autoComplete="off"
                      id="vault-identifier"
                      onChange={e => setForm(f => ({ ...f, identifier: e.target.value }))}
                      value={form.identifier}
                    />
                  </Field>
                </div>
                <Field htmlFor="vault-password" label={v.passwordField}>
                  <Input
                    autoComplete="new-password"
                    id="vault-password"
                    onChange={e => setForm(f => ({ ...f, password: e.target.value }))}
                    type="password"
                    value={form.password}
                  />
                </Field>
              </>
            )}

            {form.kind === 'payment' && (
              <>
                <Field htmlFor="vault-card-number" label={v.cardNumberField}>
                  <Input
                    autoComplete="off"
                    id="vault-card-number"
                    inputMode="numeric"
                    onChange={e => setForm(f => ({ ...f, cardNumber: e.target.value }))}
                    type="password"
                    value={form.cardNumber}
                  />
                </Field>
                <Field htmlFor="vault-card-name" label={v.cardNameField}>
                  <Input
                    autoComplete="off"
                    id="vault-card-name"
                    onChange={e => setForm(f => ({ ...f, cardName: e.target.value }))}
                    value={form.cardName}
                  />
                </Field>
                <div className="grid items-start gap-4 sm:grid-cols-4">
                  <Field htmlFor="vault-exp-month" label={v.expMonthField}>
                    <Input
                      id="vault-exp-month"
                      inputMode="numeric"
                      maxLength={2}
                      onChange={e => setForm(f => ({ ...f, expMonth: e.target.value }))}
                      placeholder="MM"
                      value={form.expMonth}
                    />
                  </Field>
                  <Field htmlFor="vault-exp-year" label={v.expYearField}>
                    <Input
                      id="vault-exp-year"
                      inputMode="numeric"
                      maxLength={4}
                      onChange={e => setForm(f => ({ ...f, expYear: e.target.value }))}
                      placeholder="YYYY"
                      value={form.expYear}
                    />
                  </Field>
                  <Field htmlFor="vault-cvc" label={v.cvcField}>
                    <Input
                      autoComplete="off"
                      id="vault-cvc"
                      inputMode="numeric"
                      maxLength={4}
                      onChange={e => setForm(f => ({ ...f, cvc: e.target.value }))}
                      type="password"
                      value={form.cvc}
                    />
                  </Field>
                  <Field htmlFor="vault-postal" label={v.postalField}>
                    <Input
                      id="vault-postal"
                      onChange={e => setForm(f => ({ ...f, postal: e.target.value }))}
                      value={form.postal}
                    />
                  </Field>
                </div>
              </>
            )}

            {form.kind === 'address' && (
              <>
                <Field htmlFor="vault-line1" label={v.addressLine1Field}>
                  <Input
                    id="vault-line1"
                    onChange={e => setForm(f => ({ ...f, line1: e.target.value }))}
                    value={form.line1}
                  />
                </Field>
                <Field htmlFor="vault-line2" label={v.addressLine2Field} optional optionalLabel={v.optional}>
                  <Input
                    id="vault-line2"
                    onChange={e => setForm(f => ({ ...f, line2: e.target.value }))}
                    value={form.line2}
                  />
                </Field>
                <div className="grid items-start gap-4 sm:grid-cols-2">
                  <Field htmlFor="vault-city" label={v.cityField}>
                    <Input
                      id="vault-city"
                      onChange={e => setForm(f => ({ ...f, city: e.target.value }))}
                      value={form.city}
                    />
                  </Field>
                  <Field htmlFor="vault-state" label={v.stateField} optional optionalLabel={v.optional}>
                    <Input
                      id="vault-state"
                      onChange={e => setForm(f => ({ ...f, state: e.target.value }))}
                      value={form.state}
                    />
                  </Field>
                </div>
                <div className="grid items-start gap-4 sm:grid-cols-2">
                  <Field htmlFor="vault-address-postal" label={v.postalField}>
                    <Input
                      id="vault-address-postal"
                      onChange={e => setForm(f => ({ ...f, postal: e.target.value }))}
                      value={form.postal}
                    />
                  </Field>
                  <Field htmlFor="vault-country" label={v.countryField}>
                    <Input
                      id="vault-country"
                      onChange={e => setForm(f => ({ ...f, country: e.target.value }))}
                      value={form.country}
                    />
                  </Field>
                </div>
              </>
            )}

            {formError && <p className="text-xs text-destructive">{formError}</p>}

            <DialogFooter>
              <Button onClick={closeAdd} size="sm" type="button" variant="ghost">
                {t.common.cancel}
              </Button>
              <Button disabled={addMutation.isPending} size="sm" type="submit">
                {addMutation.isPending ? v.adding : v.addConfirm}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* Delete confirmation */}
      <ConfirmDialog
        confirmLabel={v.deleteConfirm}
        description={pendingDelete ? v.deleteDescription(pendingDelete.label) : undefined}
        destructive
        onClose={() => setPendingDelete(null)}
        onConfirm={async () => {
          if (pendingDelete) {
            await deleteItem(pendingDelete)
          }
        }}
        open={pendingDelete !== null}
        title={v.deleteTitle}
      />
    </SettingsContent>
  )
}
