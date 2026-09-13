import { useMutation } from '@tanstack/react-query'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router'

import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { notify } from '@/store/notifications'

import type { VaultData } from './vault-settings-data'
import {
  buildSecret,
  EMPTY_FORM,
  isValidOrigin,
  isVaultKind,
  type VaultForm,
  type VaultKind,
  type VaultPrefill
} from './vault-settings-form'

export function useVaultAdd({ requestGateway, invalidate }: VaultData) {
  const { t } = useI18n()
  const v = t.settings.vault
  const [searchParams, setSearchParams] = useSearchParams()
  const [addOpen, setAddOpen] = useState(false)
  const [form, setForm] = useState<VaultForm>(EMPTY_FORM)
  const [formError, setFormError] = useState<null | string>(null)
  // React Query retains mutation variables; the secret is consumed and wiped separately.
  const pendingSecret = useRef<null | Record<string, string>>(null)

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

  return { addOpen, form, formError, setForm, closeAdd, openAdd, submitAdd, addMutation }
}
