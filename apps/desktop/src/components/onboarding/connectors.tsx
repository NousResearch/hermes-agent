import { useEffect, useMemo, useState } from 'react'

import { ConnectorLogo } from '@/components/ui/connector-logo'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { Check, Loader2, Search } from '@/lib/icons'
import { type Connector, listLocalConnectors, searchConnectors } from '@/lib/mcp-connectors'
import { cn } from '@/lib/utils'
import { type OnboardingFlow, toggleOnboardingConnector } from '@/store/onboarding'

import { DecodedLabel, GlyphText, HackeryButton, useScramble } from './glyph'

/**
 * "Which apps do you use?" — the last onboarding step.
 *
 * Checking an app here connects nothing. It records intent in
 * `mcp.connectors`, which the agent reads at session start, so the first time
 * a task actually needs Linear it raises the inline consent card instead of
 * saying it has no access. That deferral is the whole design: a first-run
 * user should not sit through five OAuth tabs before typing anything, and a
 * sign-in prompt lands far better when there's a concrete reason attached.
 *
 * The grid is local sources only (reviewed catalog + curated vendor
 * directory) because a first-run user should see recognizable apps
 * immediately. The search box widens to the public registry for anyone
 * looking for something specific.
 */
export function ConnectorsPanel({
  flow,
  leaving,
  onBegin
}: {
  flow: Extract<OnboardingFlow, { status: 'choosing_connectors' }>
  leaving: boolean
  onBegin: () => void
}) {
  const { t } = useI18n()
  const copy = t.onboarding.connectors
  const [local, setLocal] = useState<Connector[] | null>(null)
  const [query, setQuery] = useState('')
  const [found, setFound] = useState<Connector[]>([])
  const [searching, setSearching] = useState(false)

  const label = flow.selected.length > 0 ? copy.continueWith(flow.selected.length) : copy.skip
  const scrambledBegin = useScramble(label, leaving)

  useEffect(() => {
    let live = true

    void listLocalConnectors()
      .catch((): Connector[] => [])
      .then(entries => {
        if (live) {
          setLocal(entries)
        }
      })

    return () => {
      live = false
    }
  }, [])

  // Debounced so a typing user doesn't fire a registry query per keystroke.
  useEffect(() => {
    const needle = query.trim()

    if (needle.length < 2) {
      setFound([])
      setSearching(false)

      return
    }

    setSearching(true)
    let live = true

    const timer = window.setTimeout(() => {
      void searchConnectors(needle, 24)
        .catch((): Connector[] => [])
        .then(entries => {
          if (live) {
            setFound(entries)
            setSearching(false)
          }
        })
    }, 300)

    return () => {
      live = false
      window.clearTimeout(timer)
    }
  }, [query])

  const shown = useMemo(() => {
    if (query.trim().length >= 2) {
      return found
    }

    // Selected apps stay visible after the search box clears — otherwise a
    // registry pick made mid-search silently vanishes from the grid.
    const entries = local ?? []
    const missing = flow.selected.filter(name => !entries.some(entry => entry.name === name))

    return [
      ...entries,
      ...missing.map((name): Connector => ({
        auth: 'unknown',
        description: '',
        docs: '',
        homepage: '',
        name,
        needsInstall: false,
        keywords: [],
        publisher: '',
        registryName: '',
        requiredEnv: [],
        setup: [],
        source: 'registry',
        title: name,
        trust: 'community',
        url: null
      }))
    ]
  }, [flow.selected, found, local, query])

  return (
    <div className="grid gap-5 py-2">
      <div className="grid justify-items-center gap-1.5 text-center">
        <DecodedLabel leaving={leaving} text={copy.title} />
        <p className="max-w-sm text-sm text-muted-foreground">{copy.subtitle}</p>
      </div>

      <div className="relative">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
        <Input
          className="h-8 pl-8 text-sm"
          onChange={event => setQuery(event.currentTarget.value)}
          placeholder={copy.searchPlaceholder}
          value={query}
        />
        {searching && <Loader2 className="absolute right-2.5 top-1/2 size-3.5 -translate-y-1/2 animate-spin" />}
      </div>

      <div
        className={cn(
          'grid max-h-64 grid-cols-2 gap-1.5 overflow-y-auto transition duration-[360ms] ease-out sm:grid-cols-3',
          leaving ? 'opacity-0 saturate-0' : 'opacity-100 saturate-100'
        )}
      >
        {local === null ? (
          <div className="col-span-full flex items-center justify-center py-6">
            <Loader2 className="size-4 animate-spin text-muted-foreground" />
          </div>
        ) : shown.length === 0 ? (
          <p className="col-span-full py-6 text-center text-sm text-muted-foreground">
            {query.trim().length >= 2 ? copy.noResults(query.trim()) : copy.noneAvailable}
          </p>
        ) : (
          shown.map(connector => (
            <ConnectorTile
              checked={flow.selected.includes(connector.name)}
              connector={connector}
              key={connector.name}
              onToggle={() => toggleOnboardingConnector(connector.name)}
            />
          ))
        )}
      </div>

      <div className="flex justify-center">
        <HackeryButton
          disabled={flow.saving}
          label={<GlyphText text={scrambledBegin} />}
          loading={flow.saving}
          onClick={onBegin}
        />
      </div>
    </div>
  )
}

function ConnectorTile({
  checked,
  connector,
  onToggle
}: {
  checked: boolean
  connector: Connector
  onToggle: () => void
}) {
  return (
    <button
      aria-pressed={checked}
      className={cn(
        'flex items-center gap-2 rounded-md border px-2 py-1.5 text-left transition-colors',
        checked
          ? 'border-primary/40 bg-primary/10 text-foreground'
          : 'border-border/60 text-muted-foreground hover:border-border hover:text-foreground'
      )}
      onClick={onToggle}
      type="button"
    >
      <ConnectorLogo className="size-5 rounded" connector={connector} />
      <span className="min-w-0 flex-1 truncate text-sm">{connector.title}</span>
      {checked && <Check aria-hidden className="size-3.5 shrink-0 text-primary" />}
    </button>
  )
}
