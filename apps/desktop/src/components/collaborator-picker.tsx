import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'

export function CollaboratorPicker() {
  const { locale } = useI18n()
  const pl = locale === 'pl'
  const [items, setItems] = useState<Array<{ root: string }>>([])
  const [busy, setBusy] = useState(true)
  const [error, setError] = useState('')
  useEffect(() => {
    let active = true
    const detect = window.hermesDesktop?.detectCollaborators

    if (!detect) {
      setBusy(false)

      return
    }

    void detect()
      .then(result => {
        if (active) {setItems(result)}
      })
      .catch(err => {
        if (active) {setError(String(err.message || err))}
      })
      .finally(() => {
        if (active) {setBusy(false)}
      })

    return () => {
      active = false
    }
  }, [])

  const browse = async () => {
    setBusy(true)
    setError('')

    try {
      setItems(await window.hermesDesktop.detectCollaborators(true))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const select = async (root: string) => {
    setBusy(true)
    setError('')

    try {
      await window.hermesDesktop.selectCollaborator(root)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setBusy(false)
    }
  }

  return (
    <section className="mt-6 space-y-3 border-t border-(--stroke-nous) pt-5">
      <h3 className="text-sm font-semibold">
        {pl
          ? 'Mam już Hermesa — użyj istniejącego współpracownika'
          : 'I already have Hermes — use an existing collaborator'}
      </h3>
      <p className="text-sm text-muted-foreground">
        {pl
          ? 'Sprawdzamy gotowy silnik. Nie instalujemy go ponownie. Rozmowy i ustawienia Cześka pozostaną w jego własnym profilu.'
          : 'Reuse a verified engine without reinstalling it. Czesiek keeps conversations and settings in its own profile.'}
      </p>
      <div aria-live="polite" className="text-sm text-muted-foreground">
        {busy
          ? pl
            ? 'Sprawdzam środowisko…'
            : 'Checking the environment…'
          : !items.length
            ? pl
              ? 'Nie znaleziono instalacji zgodnej z rozmową Live. Użyj silnika dołączonego do Cześka — bez pobierania.'
              : 'No Live-compatible installation found. Use the engine bundled with Czesiek — no download needed.'
            : null}
      </div>
      {items.map(item => (
        <Button
          className="h-auto w-full flex-col items-start whitespace-normal text-left"
          disabled={busy}
          key={item.root}
          onClick={() => void select(item.root)}
          variant="secondary"
        >
          <span>{pl ? 'Połącz z tym Hermesem' : 'Use this Hermes'}</span>
          <span className="break-all text-xs text-muted-foreground">{item.root}</span>
        </Button>
      ))}
      <Button
        disabled={busy || !window.hermesDesktop?.detectCollaborators}
        onClick={() => void browse()}
        variant="ghost"
      >
        {pl ? 'Wskaż folder Hermesa' : 'Choose Hermes folder'}
      </Button>
      {error && (
        <p className="text-sm text-destructive" role="alert">
          {error}
        </p>
      )}
    </section>
  )
}
