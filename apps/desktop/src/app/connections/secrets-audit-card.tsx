import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import { fixSecretsPermissions, getSecretsAudit } from '@/api/secrets'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { ShieldLock } from '@/lib/icons'
import { notifyError } from '@/store/notifications'

const COPY = {
  en: {
    body: (n: number) =>
      `Your keys and sign-ins are stored in files on this computer (${n} secret entries in .env). They are kept out of notes and never shown here.`,
    fix: 'Make them private',
    fixed: 'Only you can read these files.',
    open: (names: string) => `Other users of this computer could read: ${names}.`,
    title: 'Where your secrets live',
    unchecked: 'Permission check is not available on this system.'
  },
  pl: {
    body: (n: number) =>
      `Twoje klucze i logowania leżą w plikach na tym komputerze (wpisów z sekretami w .env: ${n}). Nie trafiają do notatek i nigdy nie są tu pokazywane.`,
    fix: 'Zrób je prywatnymi',
    fixed: 'Tylko Ty możesz czytać te pliki.',
    open: (names: string) => `Inni użytkownicy tego komputera mogliby przeczytać: ${names}.`,
    title: 'Gdzie leżą Twoje sekrety',
    unchecked: 'Sprawdzanie uprawnień nie jest dostępne w tym systemie.'
  }
} as const

/** Presence-only: file names and permissions, never a value. One click tightens open files to owner-only. */
export function SecretsAuditCard() {
  const { locale } = useI18n()
  const copy = locale === 'pl' ? COPY.pl : COPY.en
  const client = useQueryClient()
  const { data } = useQuery({ queryFn: () => getSecretsAudit(), queryKey: ['secrets-audit'] })
  const [busy, setBusy] = useState(false)

  if (!data) {
    return null
  }

  const open = data.items.filter(item => item.too_open).map(item => item.name)

  const fix = async () => {
    setBusy(true)

    try {
      client.setQueryData(['secrets-audit'], await fixSecretsPermissions())
    } catch (error) {
      notifyError(error, copy.title)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="jarvis-well flex items-start gap-3 p-4">
      <ShieldLock className="mt-0.5 size-5 shrink-0 text-(--ui-accent)" />
      <div className="grid gap-1.5">
        <h3 className="text-sm font-semibold text-(--ui-text-primary)">{copy.title}</h3>
        <p className="text-xs text-(--ui-text-secondary)">{copy.body(data.env_secrets.length)}</p>
        {!data.permissions_checked ? (
          <p className="text-xs text-(--ui-text-tertiary)">{copy.unchecked}</p>
        ) : open.length ? (
          <>
            <p className="text-xs text-destructive">{copy.open(open.join(', '))}</p>
            <div>
              <Button disabled={busy} onClick={() => void fix()} size="sm" type="button">
                {copy.fix}
              </Button>
            </div>
          </>
        ) : (
          <p className="text-xs text-(--ui-text-tertiary)">{copy.fixed}</p>
        )}
      </div>
    </section>
  )
}
