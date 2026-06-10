/**
 * Cmd-K "Install theme…" page.
 *
 * Browses the VS Code Marketplace for color themes: an empty query shows the
 * most-installed themes, typing runs a live (debounced) search against the
 * Marketplace. Selecting a row downloads + converts + installs it via the same
 * pipeline as the settings importer, then activates it — and stays open so the
 * user can grab several.
 */

import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'

import type { DesktopMarketplaceSearchItem } from '@/global'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Check, Download, Loader2, Palette } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { installVscodeThemeFromMarketplace } from '@/themes/install'

const compactNumber = new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 })

function useDebounced<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value)

  useEffect(() => {
    const handle = setTimeout(() => setDebounced(value), delayMs)

    return () => clearTimeout(handle)
  }, [value, delayMs])

  return debounced
}

interface MarketplaceThemePageProps {
  search: string
  /** Activate a freshly installed theme by slug. */
  onPickTheme: (name: string) => void
}

export function MarketplaceThemePage({ search, onPickTheme }: MarketplaceThemePageProps) {
  const { t } = useI18n()
  const copy = t.commandCenter.installTheme
  const debouncedSearch = useDebounced(search.trim(), 300)
  const [installingId, setInstallingId] = useState<string | null>(null)
  const [installed, setInstalled] = useState<Record<string, true>>({})
  const [installError, setInstallError] = useState<string | null>(null)

  const query = useQuery({
    queryKey: ['marketplace-themes', debouncedSearch],
    queryFn: () => window.hermesDesktop?.themes?.searchMarketplace(debouncedSearch) ?? Promise.resolve([]),
    staleTime: 5 * 60 * 1000
  })

  const install = async (item: DesktopMarketplaceSearchItem) => {
    if (installingId) {
      return
    }

    setInstallingId(item.extensionId)
    setInstallError(null)

    try {
      const theme = await installVscodeThemeFromMarketplace(item.extensionId)

      triggerHaptic('crisp')
      setInstalled(prev => ({ ...prev, [item.extensionId]: true }))
      onPickTheme(theme.name)
    } catch (error) {
      setInstallError(error instanceof Error ? error.message : copy.error)
    } finally {
      setInstallingId(null)
    }
  }

  if (query.isLoading) {
    return <Status icon={<Loader2 className="size-4 animate-spin" />} text={copy.loading} />
  }

  if (query.isError) {
    return <Status text={copy.error} tone="error" />
  }

  const results = query.data ?? []

  if (results.length === 0) {
    return <Status text={copy.empty} />
  }

  return (
    <div role="listbox">
      {installError && (
        <p className="px-3 pb-1.5 pt-2 text-[0.75rem] text-(--ui-red)">{installError}</p>
      )}
      {results.map(item => {
        const busy = installingId === item.extensionId
        const done = installed[item.extensionId]

        return (
          <button
            className="flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-left text-sm transition-colors hover:bg-(--chrome-action-hover) disabled:opacity-60 aria-disabled:opacity-60"
            disabled={Boolean(installingId) && !busy}
            key={item.extensionId}
            onClick={() => void install(item)}
            onMouseDown={event => event.preventDefault()}
            role="option"
            type="button"
          >
            <Palette className="size-4 shrink-0 text-muted-foreground" />
            <span className="flex min-w-0 flex-col">
              <span className="truncate font-medium">{item.displayName}</span>
              <span className="truncate text-[0.75rem] text-muted-foreground/80">
                {item.publisher}
                {item.installs > 0 ? ` · ${copy.installs(compactNumber.format(item.installs))}` : ''}
              </span>
            </span>
            <span className="ml-auto flex shrink-0 items-center gap-1 text-[0.75rem] text-muted-foreground">
              {busy ? (
                <>
                  <Loader2 className="size-3.5 animate-spin" />
                  {copy.installing}
                </>
              ) : done ? (
                <>
                  <Check className="size-3.5 text-(--ui-green)" />
                  {copy.installed}
                </>
              ) : (
                <>
                  <Download className="size-3.5" />
                  {copy.install}
                </>
              )}
            </span>
          </button>
        )
      })}
    </div>
  )
}

function Status({ icon, text, tone }: { icon?: React.ReactNode; text: string; tone?: 'error' }) {
  return (
    <div
      className={cn(
        'flex items-center justify-center gap-2 px-3 py-8 text-sm',
        tone === 'error' ? 'text-(--ui-red)' : 'text-muted-foreground'
      )}
    >
      {icon}
      {text}
    </div>
  )
}
