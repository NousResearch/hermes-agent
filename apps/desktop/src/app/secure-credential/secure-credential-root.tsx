import { type FormEvent, StrictMode, useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Loader } from '@/components/ui/loader'
import { TRANSLATIONS } from '@/i18n'
import { normalizeLocale } from '@/i18n/languages'
import { Lock } from '@/lib/icons'

interface CredentialRequest {
  envVar: string
  locale: string
  prompt: string
}

function SecureCredentialEntry() {
  const bridge = window.hermesCredential
  const [request, setRequest] = useState<CredentialRequest | null>(null)
  const [value, setValue] = useState('')
  const [busy, setBusy] = useState(false)
  const [loadFailed, setLoadFailed] = useState(false)
  const [saveError, setSaveError] = useState('')
  const locale = normalizeLocale(request?.locale)
  const copy = useMemo(() => TRANSLATIONS[locale].prompts.secureCredential, [locale])

  useEffect(() => {
    document.documentElement.lang = locale
    document.documentElement.dir = locale === 'ar' ? 'rtl' : 'ltr'
    document.title = copy.title
  }, [copy.title, locale])

  useEffect(() => {
    if (!bridge) {
      setLoadFailed(true)

      return
    }

    void bridge
      .getRequest()
      .then(setRequest)
      .catch(() => setLoadFailed(true))
  }, [bridge])

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    if (!bridge || !value) {
      return
    }

    setBusy(true)
    setSaveError('')

    try {
      const result = await bridge.submit(value)

      if (!result.ok) {
        setSaveError(result.error || copy.saveFailed)
        setBusy(false)
      }
    } catch {
      setSaveError(copy.saveFailed)
      setBusy(false)
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-background p-5 text-foreground">
      <section className="w-full max-w-md border border-(--stroke-nous) bg-background p-5 shadow-nous">
        <header className="mb-5 flex items-start gap-3">
          <Lock aria-hidden="true" className="mt-0.5 shrink-0 text-primary" />
          <div>
            <h1 className="text-base font-semibold">{copy.title}</h1>
            <p className="mt-1 text-sm text-(--ui-text-secondary)">{copy.description}</p>
          </div>
        </header>

        {loadFailed ? (
          <p className="text-sm text-destructive" role="alert">
            {copy.unavailable}
          </p>
        ) : !request ? (
          <div className="flex justify-center py-5">
            <Loader label={copy.loading} type="lemniscate-bloom" />
          </div>
        ) : (
          <>
            <div className="mb-4 border-t border-(--ui-stroke-tertiary) pt-4">
              <div className="text-xs font-medium uppercase tracking-wide text-(--ui-text-tertiary)">
                {copy.savingAs}
              </div>
              <div className="mt-1 break-all font-mono text-sm">{request.envVar}</div>
              {request.prompt ? <p className="mt-2 text-sm text-(--ui-text-secondary)">{request.prompt}</p> : null}
            </div>

            <form className="grid gap-3" onSubmit={submit}>
              <Input
                aria-label={request.envVar}
                autoFocus
                disabled={busy}
                onChange={event => setValue(event.target.value)}
                placeholder={copy.placeholder}
                type="password"
                value={value}
              />
              {saveError ? (
                <p className="text-sm text-destructive" role="alert">
                  {saveError}
                </p>
              ) : null}
              <div className="flex justify-end gap-2 pt-1">
                <Button disabled={busy} onClick={() => void bridge?.cancel()} type="button" variant="text">
                  {copy.cancel}
                </Button>
                <Button disabled={busy || !value} type="submit">
                  {busy ? copy.saving : copy.save}
                </Button>
              </div>
            </form>
          </>
        )}
      </section>
    </main>
  )
}

export function mountSecureCredential(): void {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <SecureCredentialEntry />
    </StrictMode>
  )
}
