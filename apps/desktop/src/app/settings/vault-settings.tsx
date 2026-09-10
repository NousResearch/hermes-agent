import { useStore } from '@nanostores/react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router'

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
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { KeyRound, Lock, Plus, ShieldLock, Trash2 } from '@/lib/icons'
import { $activeConnectionId } from '@/store/connections'
import { requestGatewayForProfile } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import { $gatewayState } from '@/store/session'
import { $settingsScopeProfile } from '@/store/settings-scope'

import { CONTROL_TEXT } from './constants'
import { ListRow, Pill, SectionHeading, SettingsContent } from './primitives'

// Vault data is private to one (connection, profile); the cache key carries that owner so a
// late response from profile A can never paint under profile B.
export const vaultOwnerKey = (connectionId: null | string, profile: string) => `${connectionId ?? ''}::${profile}`
const vaultQueryKey = (owner: string) => ['vault-items', owner] as const
const vaultSourcesQueryKey = (owner: string) => ['vault-sources', owner] as const

export type VaultSourceName = 'bitwarden' | 'local' | 'onepassword'

/** One login source as reported by `vault.sources` — the backend is authoritative for enabled/unlocked. */
export interface VaultSource {
  name: VaultSourceName
  display_name: string
  enabled: boolean
  needs_unlock: boolean
  unlocked: boolean
  installed: boolean
}

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
  backend?: VaultSourceName
  has_otp?: boolean
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
  otpSecret: '',
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
      password: form.password,
      ...(form.otpSecret.trim() ? { otp_secret: form.otpSecret.trim() } : {})
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
  const gatewayState = useStore($gatewayState)
  const queryClient = useQueryClient()
  // The owner this panel edits: every RPC below goes through the owner's socket with an explicit
  // profile — never the ambient foreground gateway. The mount site keys the panel by this same
  // owner, so a profile switch / connection swap remounts it: dialogs close and drafts (including a
  // typed master password) are gone by construction rather than by cleanup code.
  const scopeProfile = useStore($settingsScopeProfile)
  const connectionId = useStore($activeConnectionId)
  const owner = vaultOwnerKey(connectionId, scopeProfile)

  const requestGateway = useCallback(
    <T,>(method: string, params: Record<string, unknown> = {}) =>
      requestGatewayForProfile<T>(scopeProfile, method, params),
    [scopeProfile]
  )

  const VAULT_QUERY_KEY = useMemo(() => vaultQueryKey(owner), [owner])
  const VAULT_SOURCES_QUERY_KEY = useMemo(() => vaultSourcesQueryKey(owner), [owner])
  const [searchParams, setSearchParams] = useSearchParams()

  const [addOpen, setAddOpen] = useState(false)
  const [form, setForm] = useState<VaultForm>(EMPTY_FORM)
  const [formError, setFormError] = useState<null | string>(null)
  const [pendingDelete, setPendingDelete] = useState<null | VaultItem>(null)
  const [unlockTarget, setUnlockTarget] = useState<null | VaultSource>(null)
  const [masterPassword, setMasterPassword] = useState('')
  const [unlockError, setUnlockError] = useState<null | string>(null)
  // Secrets never become mutation variables (react-query retains those after settle); they live
  // in refs the mutationFn consumes and wipes.
  const pendingMasterPassword = useRef('')
  const pendingSecret = useRef<null | Record<string, string>>(null)

  const { data: sourcesData } = useQuery({
    enabled: gatewayState === 'open',
    queryKey: VAULT_SOURCES_QUERY_KEY,
    queryFn: async () => {
      const result = await requestGateway<{ sources: VaultSource[] }>('vault.sources', {})

      return result.sources
    }
  })

  const externalSources = useMemo(() => (sourcesData ?? []).filter(s => s.needs_unlock), [sourcesData])

  const invalidateVault = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: VAULT_QUERY_KEY })
    void queryClient.invalidateQueries({ queryKey: VAULT_SOURCES_QUERY_KEY })
  }, [queryClient])

  const setSourceEnabled = useMutation({
    mutationFn: ({ name, enabled }: { name: VaultSourceName; enabled: boolean }) =>
      requestGateway<{ enabled: boolean }>('vault.source.set', { name, enabled }),
    onSuccess: invalidateVault,
    onError: err => notifyError(err, v.sources.toggleFailed)
  })

  const lockSource = useMutation({
    mutationFn: (name: VaultSourceName) => requestGateway<{ locked: boolean }>('vault.lock', { name }),
    onSuccess: invalidateVault
  })

  // The master password lives only in this dialog's state; it is cleared the moment the request
  // returns (success or failure) and never touches a store or the transcript.
  const closeUnlock = useCallback(() => {
    setUnlockTarget(null)
    setMasterPassword('')
    setUnlockError(null)
  }, [])

  const unlockSource = useMutation({
    mutationFn: ({ name }: { name: VaultSourceName }) => {
      const password = pendingMasterPassword.current
      pendingMasterPassword.current = ''

      return requestGateway<{ unlocked: boolean }>('vault.unlock', { name, password })
    },
    onSuccess: (_result, { name }) => {
      triggerHaptic('submit')
      const source = externalSources.find(s => s.name === name)
      notify({ kind: 'success', message: v.sources.unlocked(source?.display_name ?? name) })
      closeUnlock()
      invalidateVault()
    },
    onError: err => {
      setMasterPassword('')
      setUnlockError(err instanceof Error ? err.message : String(err))
    }
  })

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

  const invalidate = useCallback(() => queryClient.invalidateQueries({ queryKey: VAULT_QUERY_KEY }), [queryClient])

  const addMutation = useMutation({
    mutationFn: async (payload: { kind: VaultKind; label: string; origin?: string }) => {
      const secret = pendingSecret.current
      pendingSecret.current = null

      return requestGateway<{ id: string }>('vault.add', { ...payload, secret: secret ?? {} })
    },
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

    // Every kind is filled only on the origin it was saved for; a card without an origin is unfillable.
    const origin = form.origin.trim()

    if (!isValidOrigin(origin)) {
      setFormError(v.originInvalid)

      return
    }

    if (form.kind === 'login' && (!form.identifier.trim() || !form.password)) {
      setFormError(v.loginFieldsRequired)

      return
    }

    pendingSecret.current = buildSecret(form)
    addMutation.mutate({
      kind: form.kind,
      label: form.label.trim(),
      ...(origin ? { origin } : {})
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

  const sourceLabel = useCallback(
    (name: VaultSourceName) => externalSources.find(s => s.name === name)?.display_name ?? name,
    [externalSources]
  )

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
            item.backend && item.backend !== 'local' ? (
              <Pill tone="muted">{sourceLabel(item.backend)}</Pill>
            ) : (
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
            )
          }
          description={
            // identifier · origin · date, separated so the row scans as three facts; the origin is
            // omitted when the label already IS the host (save-on-page items are labelled by host).
            <span className="flex flex-wrap items-center gap-x-2">
              {item.identifier && <span className="truncate">{v.identifierShown(item.identifier)}</span>}
              {item.origin && item.origin.replace(/^https?:\/\//, '') !== item.label && (
                <>
                  {item.identifier && (
                    <span aria-hidden className="text-(--ui-text-tertiary)">
                      ·
                    </span>
                  )}
                  <span className="truncate">{item.origin}</span>
                </>
              )}
              <span aria-hidden className="text-(--ui-text-tertiary)">
                ·
              </span>
              <span>{v.createdOn(formatCreated(item.created_at))}</span>
            </span>
          }
          key={item.id}
          title={
            <span className="flex items-center gap-2">
              <span className="truncate">{item.label}</span>
              <Pill tone={item.kind === 'login' ? 'primary' : 'muted'}>{kindLabel(item.kind)}</Pill>
              {item.has_otp && <Pill tone="muted">{v.twoFactorBadge}</Pill>}
            </span>
          }
        />
      ))}

      {/* Password managers */}
      <div className="mt-6">
        <SectionHeading icon={KeyRound} title={v.sources.title} />
      </div>
      <p className="mb-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {v.sources.blurb}
      </p>
      {externalSources.map(source => (
        <ListRow
          action={
            <span className="flex items-center justify-end gap-2">
              {source.enabled &&
                (source.unlocked ? (
                  <Button
                    className="gap-1.5"
                    disabled={lockSource.isPending}
                    onClick={() => lockSource.mutate(source.name)}
                    size="sm"
                    type="button"
                    variant="ghost"
                  >
                    <Lock className="size-3.5" />
                    {v.sources.lock}
                  </Button>
                ) : (
                  <Button
                    className="gap-1.5"
                    onClick={() => setUnlockTarget(source)}
                    size="sm"
                    type="button"
                    variant="outline"
                  >
                    <KeyRound className="size-3.5" />
                    {v.sources.unlock}
                  </Button>
                ))}
              {source.installed && (
                <Switch
                  aria-label={source.display_name}
                  checked={source.enabled}
                  disabled={setSourceEnabled.isPending}
                  onCheckedChange={enabled => {
                    triggerHaptic('selection')
                    setSourceEnabled.mutate({ name: source.name, enabled })
                  }}
                />
              )}
            </span>
          }
          description={
            !source.installed
              ? v.sources.notInstalled(source.display_name)
              : source.enabled
                ? source.unlocked
                  ? v.sources.unlockedDesc
                  : v.sources.lockedDesc
                : v.sources.disabledDesc
          }
          key={source.name}
          title={
            <span className="flex items-center gap-2">
              <span>{source.display_name}</span>
              <Pill tone={source.enabled && source.unlocked ? 'primary' : 'muted'}>
                {!source.installed
                  ? v.sources.statusNotDetected
                  : !source.enabled
                    ? v.sources.statusOff
                    : source.unlocked
                      ? v.sources.statusUnlocked
                      : v.sources.statusLocked}
              </Pill>
            </span>
          }
        />
      ))}

      {/* Unlock dialog */}
      <Dialog onOpenChange={open => !open && closeUnlock()} open={unlockTarget !== null}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle icon={KeyRound}>{v.sources.unlockTitle(unlockTarget?.display_name ?? '')}</DialogTitle>
            <DialogDescription>{v.sources.unlockDescription}</DialogDescription>
          </DialogHeader>
          <form
            className="grid gap-3"
            onSubmit={e => {
              e.preventDefault()

              if (unlockTarget && masterPassword) {
                pendingMasterPassword.current = masterPassword
                setMasterPassword('')
                unlockSource.mutate({ name: unlockTarget.name })
              }
            }}
          >
            <Input
              autoComplete="current-password"
              autoFocus
              disabled={unlockSource.isPending}
              onChange={e => setMasterPassword(e.target.value)}
              placeholder={v.sources.masterPasswordPlaceholder}
              type="password"
              value={masterPassword}
            />
            {unlockError && <p className="text-xs text-destructive">{unlockError}</p>}
            <DialogFooter>
              <Button onClick={closeUnlock} type="button" variant="ghost">
                {t.common.cancel}
              </Button>
              <Button disabled={unlockSource.isPending || !masterPassword} type="submit">
                {unlockSource.isPending ? v.sources.unlocking : v.sources.unlock}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

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
                <Select onValueChange={value => setForm(f => ({ ...f, kind: value as VaultKind }))} value={form.kind}>
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

            <Field htmlFor="vault-origin" label={v.originField}>
              <Input
                id="vault-origin"
                inputMode="url"
                onChange={e => setForm(f => ({ ...f, origin: e.target.value }))}
                placeholder={form.kind === 'login' ? v.originPlaceholder : v.originPlaceholderCheckout}
                value={form.origin}
              />
            </Field>

            {form.kind === 'login' && (
              <>
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
                <Field htmlFor="vault-otp" label={v.otpField} optional optionalLabel={v.optional}>
                  <Input
                    autoComplete="off"
                    id="vault-otp"
                    onChange={e => setForm(f => ({ ...f, otpSecret: e.target.value }))}
                    placeholder={v.otpPlaceholder}
                    type="password"
                    value={form.otpSecret}
                  />
                  <p className="text-xs text-muted-foreground">{v.otpHint}</p>
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
