import { useStore } from '@nanostores/react'
import { Puzzle } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { $chatLayoutPicked, assembleChatOnboarding } from '@/components/onboarding-chat/assembly'
import { CardFrame, type CardProps, useCardCommit } from '@/components/onboarding-chat/cards/frame'
import { Chip } from '@/components/onboarding-chat/chip'
import {
  accentsFor,
  AccentSwatch,
  LayoutPreviewCard,
  LAYOUTS,
  NOUS_ACCENT,
  orderConnectorPicks
} from '@/components/onboarding-chat/options'
import type { LayoutNode } from '@/components/pane-shell/tree/model'
import { ConnectorLogo } from '@/components/ui/connector-logo'
import { SearchField } from '@/components/ui/search-field'
import { registry } from '@/contrib/registry'
import { connectorIconUrl, connectorTitle } from '@/lib/connector-tools'
import { cn } from '@/lib/utils'
import { type ConnectorCatalog, useConnectorCatalog } from '@/store/connector-catalog'
import { $onboardingAnswers, setOnboardingAnswers } from '@/store/onboarding-answers'
import { type OnboardingPlugin, pluginNeedsApp, useOnboardingPluginList } from '@/store/onboarding-plugins'
import { useTheme } from '@/themes'
import { setAccentOverride } from '@/themes/accent-override'
import { normalizeHex } from '@/themes/color'

export function ConnectorsCard({ locked }: CardProps) {
  const view = useSessionView()
  const storedId = useStore(view.$storedId)
  const runtimeId = useStore(view.$runtimeId)
  const { commit, done } = useCardCommit('connectors')
  const catalog = useConnectorCatalog(storedId, runtimeId)
  const plugins = useOnboardingPluginList(storedId) ?? []

  return <ConnectorPicks catalog={catalog} commit={commit} done={done} locked={locked} plugins={plugins} />
}

interface ConnectorPicksProps {
  catalog: ConnectorCatalog
  commit: (summary: string) => boolean
  done: boolean
  locked: boolean
  plugins: OnboardingPlugin[]
}

export function ConnectorPicks({ catalog, commit, done, locked, plugins }: ConnectorPicksProps) {
  const answers = useStore($onboardingAnswers)
  const [query, setQuery] = useState('')

  const rows = useMemo(() => (catalog.status === 'ready' ? orderConnectorPicks(catalog.rows) : []), [catalog])
  const search = query.trim().toLowerCase()

  const shown = search
    ? rows.filter(row => connectorTitle(row.connector).toLowerCase().includes(search))
    : rows.slice(0, Math.max(0, 12 - plugins.length))

  const shownPlugins = search ? plugins.filter(plugin => plugin.title.toLowerCase().includes(search)) : plugins
  const picked = rows.filter(row => answers.connectors.includes(row.connector))
  const pickedPlugins = plugins.filter(plugin => answers.plugins.includes(plugin.name))
  const pickedCount = picked.length + pickedPlugins.length

  const toggle = (id: string) =>
    setOnboardingAnswers({
      connectors: answers.connectors.includes(id)
        ? answers.connectors.filter(item => item !== id)
        : [...answers.connectors, id]
    })

  const togglePlugin = (name: string) =>
    setOnboardingAnswers({
      plugins: answers.plugins.includes(name)
        ? answers.plugins.filter(item => item !== name)
        : [...answers.plugins, name]
    })

  const summary = () => {
    const apps = picked.length > 0 ? picked.map(row => row.connector).join(', ') : 'none for now'
    const tools = pickedPlugins.map(plugin => plugin.name).join(', ')

    return `apps I use, not connected yet: ${apps}${tools ? `; plugins picked, not installed yet: ${tools}` : ''}`
  }

  if (plugins.length === 0 && (catalog.status === 'unavailable' || (catalog.status === 'ready' && rows.length === 0))) {
    return (
      <CardFrame
        continueLabel="Skip this"
        done={done}
        locked={locked}
        onContinue={() => commit('apps I use: none for now')}
      >
        <p className="text-sm text-muted-foreground">
          Connections aren’t available right now — this can be set up later.
        </p>
      </CardFrame>
    )
  }

  return (
    <CardFrame
      continueLabel={pickedCount > 0 ? `Continue with ${pickedCount}` : 'None of these'}
      disabled={catalog.status === 'loading'}
      done={done}
      locked={locked}
      onContinue={() => void commit(summary())}
    >
      {catalog.status === 'loading' ? (
        <div className="grid grid-cols-3 gap-2">
          {Array.from({ length: 9 }, (_, index) => (
            <div className="h-10 animate-pulse rounded-lg bg-muted/40" key={index} />
          ))}
        </div>
      ) : (
        <>
          {rows.length + plugins.length > 12 ? (
            <SearchField onChange={setQuery} placeholder="Find an app" value={query} />
          ) : null}
          <div className="grid max-h-72 grid-cols-3 gap-2 overflow-y-auto">
            {shownPlugins.map(plugin => (
              <Chip
                className={cn(pluginNeedsApp(plugin) && 'opacity-60')}
                icon={
                  <span className="grid size-7 shrink-0 place-items-center rounded-full bg-background text-muted-foreground">
                    <Puzzle className="size-4" />
                  </span>
                }
                key={`plugin:${plugin.name}`}
                label={plugin.title}
                on={answers.plugins.includes(plugin.name)}
                onToggle={() => togglePlugin(plugin.name)}
                sub={pluginNeedsApp(plugin) ? plugin.sentence : 'Plugin'}
              />
            ))}
            {shown.map(row => (
              <Chip
                icon={
                  <ConnectorLogo
                    className="size-7 rounded-full text-sm"
                    connector={{
                      iconUrl: connectorIconUrl(row.connector),
                      name: row.connector,
                      title: connectorTitle(row.connector)
                    }}
                  />
                }
                key={row.connector}
                label={connectorTitle(row.connector)}
                on={answers.connectors.includes(row.connector)}
                onToggle={() => toggle(row.connector)}
              />
            ))}
          </div>
        </>
      )}
      <p className="text-xs text-muted-foreground">
        <strong className="font-medium text-foreground">Nothing connects or installs yet.</strong> Hermes will offer to
        link these, or install a plugin, when a task needs them, and asks first.
      </p>
    </CardFrame>
  )
}

function pickAccent(value: string, receipt?: string): void {
  const hex = normalizeHex(value)

  if (!hex) {
    return
  }

  const accent = hex === NOUS_ACCENT ? null : hex

  setOnboardingAnswers({
    accent,
    ...(receipt ? { committed: [...$onboardingAnswers.get().committed, receipt] } : {})
  })
  setAccentOverride(accent)
}

export function LookCard({ attrs, locked, messageId }: CardProps) {
  const answers = useStore($onboardingAnswers)
  const { renderedMode } = useTheme()
  const { commit, done } = useCardCommit('look')
  const accents = accentsFor(renderedMode === 'dark')
  const accent = answers.accent ?? NOUS_ACCENT
  const picked = accents.find(swatch => swatch.hex === accent.toLowerCase())

  const requested = normalizeHex(attrs.value)
  const receipt = messageId && requested ? `look-color:${JSON.stringify([messageId, requested])}` : null

  useEffect(() => {
    if (locked || done) {
      return
    }

    if (requested && receipt && !answers.committed.includes(receipt)) {
      pickAccent(requested, receipt)
    } else {
      setAccentOverride(answers.accent)
    }
  }, [answers.accent, answers.committed, done, locked, receipt, requested])

  if (attrs.value !== undefined) {
    return null
  }

  return (
    <CardFrame done={done} locked={locked} onContinue={() => commit(`accent color: ${picked?.name ?? accent}`)}>
      <div className="flex flex-wrap gap-2.5">
        {accents.map(swatch => (
          <AccentSwatch
            active={accent.toLowerCase() === swatch.hex}
            hex={swatch.hex}
            key={swatch.name}
            name={swatch.name}
            onPick={() => pickAccent(swatch.hex)}
          />
        ))}
        <AccentSwatch active={!picked} hex={accent} name="Custom color" onColorChange={pickAccent} />
      </div>
    </CardFrame>
  )
}

export function LayoutCard({ locked }: CardProps) {
  const answers = useStore($onboardingAnswers)
  const { commit, done } = useCardCommit('layout')
  const picked = useStore($chatLayoutPicked)

  const pickLayout = (id: string) => {
    $chatLayoutPicked.set(true)
    setOnboardingAnswers({ layout: id })

    const layout = LAYOUTS.find(candidate => candidate.id === id)

    const preset = registry.getArea('layouts').find(contribution => contribution.id === id)

    if (!preset?.data) {
      return
    }

    assembleChatOnboarding(preset.id, preset.data as LayoutNode, layout?.mode)
  }

  return (
    <CardFrame
      disabled={!picked}
      done={done}
      locked={locked}
      onContinue={() => {
        const choice = LAYOUTS.find(layout => layout.id === answers.layout)

        commit(`layout: ${choice?.name ?? answers.layout}`)
      }}
    >
      <div className="grid grid-cols-2 gap-3">
        {LAYOUTS.map(layout => (
          <LayoutPreviewCard
            active={picked && answers.layout === layout.id}
            description={layout.description}
            key={layout.id}
            name={layout.name}
            onSelect={() => pickLayout(layout.id)}
            tree={layout.tree}
          />
        ))}
      </div>
    </CardFrame>
  )
}
