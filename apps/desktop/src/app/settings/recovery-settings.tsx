import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'

export function RecoverySettings() {
  const [status, setStatus] = useState<{
    supported: boolean
    available: boolean
    createdAt?: string
    version?: string
  } | null>(null)

  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  useEffect(() => {
    let active = true
    void window.hermesDesktop
      ?.recoveryStatus?.()
      .then(value => {
        if (active) {
          setStatus(value)
        }
      })
      .catch(() => {
        if (active) {
          setMessage('Nie udało się odczytać kopii.')
        }
      })

    return () => {
      active = false
    }
  }, [])

  const run = async (restore: boolean) => {
    setBusy(true)
    setMessage('')

    try {
      const result = await (restore ? window.hermesDesktop.restoreRecovery() : window.hermesDesktop.prepareRecovery())

      if (result.ok) {
        setMessage(
          restore
            ? 'Przywracanie — aplikacja zostanie ponownie uruchomiona.'
            : 'Kopia aplikacji, ustawień i pamięci została zapisana na tym komputerze.'
        )
      }

      setStatus(await window.hermesDesktop.recoveryStatus())
    } catch {
      setMessage(
        'Operacja nie powiodła się. Aktualizacja nie powinna być kontynuowana bez kompletnej kopii. Sprawdź wolne miejsce i zakończ aktywne zadania.'
      )
    } finally {
      setBusy(false)
    }
  }

  if (!status?.supported) {
    return null
  }

  return (
    <section className="my-5 space-y-3 rounded-xl border border-(--stroke-nous) p-4 text-sm">
      <h3 className="text-base font-semibold">Kopia przed aktualizacją</h3>
      <p className="text-muted-foreground">
        Kopia obejmuje aplikację, klucze, ustawienia i pamięć tego komputera. Wymaga miejsca na drugi egzemplarz
        aplikacji. Przed uruchomieniem instalatora zapisz kopię; zakończ najpierw zadania.
      </p>
      {status.available && (
        <p>
          Ostatnia kopia: {status.createdAt ? new Date(status.createdAt).toLocaleString() : '—'} · wersja{' '}
          {status.version}
        </p>
      )}
      <div className="flex flex-wrap gap-2">
        <Button disabled={busy} onClick={() => void run(false)} variant="secondary">
          {busy ? 'Pracuję…' : 'Zapisz kopię'}
        </Button>
        <Button disabled={busy || !status.available} onClick={() => void run(true)} variant="secondary">
          Przywróć poprzednią wersję i dane
        </Button>
      </div>
      {message && <p role="status">{message}</p>}
    </section>
  )
}
