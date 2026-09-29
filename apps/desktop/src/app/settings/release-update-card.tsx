import { useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import type { DesktopReleaseCheck } from '@/global'
import { useI18n } from '@/i18n'
import { CheckCircle2, ExternalLink, Loader2, RefreshCw } from '@/lib/icons'
import { cn } from '@/lib/utils'

const COPY = {
  en: {
    check: 'Check for updates',
    checking: 'Checking…',
    download: 'Download installer',
    latest: (v: string) => `You have the latest version (${v}).`,
    notes: 'See what changed',
    noRelease: 'No published release was found. The repository may be private or have no release yet.',
    openPage: 'Open releases page',
    retry: 'Could not check for updates. Check your connection and try again.',
    title: 'Updates for this installed app',
    available: (latest: string, current: string) => `Version ${latest} is available (you have ${current}).`,
    hint: 'Installed builds update by installing a newer installer; your profile and data stay.'
  },
  pl: {
    check: 'Sprawdź aktualizacje',
    checking: 'Sprawdzam…',
    download: 'Pobierz instalator',
    latest: (v: string) => `Masz najnowszą wersję (${v}).`,
    notes: 'Zobacz, co się zmieniło',
    noRelease: 'Nie znaleziono opublikowanego wydania. Repozytorium może być prywatne albo nie ma jeszcze wydania.',
    openPage: 'Otwórz stronę wydań',
    retry: 'Nie udało się sprawdzić aktualizacji. Sprawdź połączenie i spróbuj ponownie.',
    title: 'Aktualizacje zainstalowanej aplikacji',
    available: (latest: string, current: string) => `Dostępna jest wersja ${latest} (masz ${current}).`,
    hint: 'Zainstalowana aplikacja aktualizuje się przez nowszy instalator; profil i dane zostają.'
  }
} as const

const open = (url: string) => void window.hermesDesktop?.openExternal?.(url)

/**
 * The update card for installed builds (no git checkout to pull from): reads
 * the distribution repository's latest GitHub Release and offers the installer
 * for this machine. Checks once when it opens, and on demand.
 */
export function ReleaseUpdateCard() {
  const { locale } = useI18n()
  const copy = locale === 'pl' ? COPY.pl : COPY.en
  const [result, setResult] = useState<DesktopReleaseCheck | null>(null)
  const [checking, setChecking] = useState(false)

  const run = useCallback(async () => {
    const check = window.hermesDesktop?.updates?.releaseCheck

    if (!check) {
      return
    }

    setChecking(true)

    try {
      setResult(await check())
    } finally {
      setChecking(false)
    }
  }, [])

  useEffect(() => {
    void run()
  }, [run])

  const available = result?.ok === true && result.updateAvailable
  const failed = result !== null && !result.ok

  return (
    <div
      className={cn(
        'rounded-xl px-4 py-3 text-sm',
        available ? 'bg-primary/8 text-foreground' : failed ? 'bg-destructive/6 text-foreground' : 'bg-muted/25'
      )}
      data-testid="release-update-card"
    >
      <p className="text-xs font-medium text-muted-foreground">{copy.title}</p>
      <div className="mt-2 flex items-start gap-2">
        {result?.ok === true && !available ? (
          <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-emerald-600 dark:text-emerald-400" />
        ) : null}
        <p className="font-medium">
          {checking && !result
            ? copy.checking
            : result === null
              ? copy.hint
              : result.ok
                ? available
                  ? copy.available(result.latestVersion, result.currentVersion)
                  : copy.latest(result.currentVersion)
                : result.reason === 'no-release'
                  ? copy.noRelease
                  : copy.retry}
        </p>
      </div>

      {result?.ok === true && available && result.notes ? (
        <p className="mt-2 line-clamp-4 whitespace-pre-line text-xs text-muted-foreground">{result.notes}</p>
      ) : null}

      <div className="mt-3 flex flex-wrap items-center gap-4">
        <Button disabled={checking} onClick={() => void run()} size="sm" variant="textStrong">
          {checking ? <Loader2 className="size-3 animate-spin" /> : <RefreshCw className="size-3" />}
          {checking ? copy.checking : copy.check}
        </Button>

        {result?.ok === true && available ? (
          <>
            <Button onClick={() => open(result.asset?.url ?? result.pageUrl)} size="sm">
              {copy.download}
            </Button>
            <Button onClick={() => open(result.pageUrl)} size="sm" variant="textStrong">
              <ExternalLink className="size-3" />
              {copy.notes}
            </Button>
          </>
        ) : null}

        {result && !result.ok && result.reason === 'no-release' ? (
          <Button onClick={() => open(result.pageUrl)} size="sm" variant="textStrong">
            <ExternalLink className="size-3" />
            {copy.openPage}
          </Button>
        ) : null}
      </div>
    </div>
  )
}
