/**
 * The three setup picks — accent, connectors, layout.
 *
 * Picks apply live. The shared catalog keeps cards and previews in agreement
 * without asking the model to enumerate the options.
 */

import { useStore } from '@nanostores/react'
import { useMemo, useState } from 'react'

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
import { connectorTitle } from '@/lib/connector-tools'
import { useConnectorCatalog } from '@/store/connector-catalog'
import { $onboardingAnswers, setOnboardingAnswers } from '@/store/onboarding-answers'
import { useTheme } from '@/themes'
import { setAccentOverride } from '@/themes/accent-override'

export function ConnectorsCard({ locked }: CardProps) {
  const view = useSessionView()
  const storedId = useStore(view.$storedId)
  const runtimeId = useStore(view.$runtimeId)
  const answers = useStore($onboardingAnswers)
  const { commit, done } = useCardCommit('connectors')
  const catalog = useConnectorCatalog(storedId, runtimeId)
  const [query, setQuery] = useState('')

  // Only what the gateway actually carries. A pick is a slug the build chat
  // can hand straight to manage_connections; a name with nothing behind it
  // is a promise it has to walk back.
  const rows = useMemo(() => (catalog.status === 'ready' ? orderConnectorPicks(catalog.rows) : []), [catalog])
  const shown = rows.filter(row => connectorTitle(row.connector).toLowerCase().includes(query.toLowerCase()))
  const picked = rows.filter(row => answers.connectors.includes(row.connector))

  const toggle = (id: string) =>
    setOnboardingAnswers({
      connectors: answers.connectors.includes(id)
        ? answers.connectors.filter(item => item !== id)
        : [...answers.connectors, id]
    })

  // Nothing to pick from: the toolset is off or the gateway is unreachable.
  // The step still has to end, so it ends honestly.
  if (catalog.status === 'unavailable' || (catalog.status === 'ready' && rows.length === 0)) {
    return (
      <CardFrame continueLabel="Skip this" done={done} locked={locked} onContinue={() => commit('apps I use: none for now')}>
        <p className="text-sm text-muted-foreground">Connections aren’t available right now — this can be set up later.</p>
      </CardFrame>
    )
  }

  return (
    <CardFrame
      continueLabel={picked.length > 0 ? `Continue with ${picked.length}` : 'None of these'}
      disabled={catalog.status === 'loading'}
      done={done}
      locked={locked}
      onContinue={() => {
        commit(
          `apps I use, not connected yet: ${picked.length > 0 ? picked.map(row => row.connector).join(', ') : 'none for now'}`
        )
      }}
    >
      {catalog.status === 'loading' ? (
        <div className="grid grid-cols-3 gap-2">
          {Array.from({ length: 9 }, (_, index) => (
            <div className="h-10 animate-pulse rounded-lg bg-muted/40" key={index} />
          ))}
        </div>
      ) : (
        <>
          {rows.length > 12 ? <SearchField onChange={setQuery} placeholder="Find an app" value={query} /> : null}
          <div className="grid max-h-72 grid-cols-3 gap-2 overflow-y-auto">
            {shown.map(row => (
              <Chip
                icon={
                  <ConnectorLogo
                    className="size-7 rounded-full text-sm"
                    connector={{ name: row.connector, title: row.name || connectorTitle(row.connector) }}
                  />
                }
                key={row.connector}
                label={row.name || connectorTitle(row.connector)}
                on={answers.connectors.includes(row.connector)}
                onToggle={() => toggle(row.connector)}
              />
            ))}
          </div>
        </>
      )}
      {/* Picking is a preference, not an authorization: nothing *** signed into
          here. Saying so is what keeps the Connect cards later from reading as
          a second ask for the same thing. */}
      <p className="text-xs text-muted-foreground">
        Nothing connects yet. Hermes will offer to link these when a task needs them, and asks before reading
        anything.
      </p>
    </CardFrame>
  )
}

export function LookCard({ locked }: CardProps) {
  const answers = useStore($onboardingAnswers)
  const { renderedMode } = useTheme()
  const { commit, done } = useCardCommit('look')
  const accents = accentsFor(renderedMode === 'dark')
  const accent = answers.accent ?? NOUS_ACCENT
  const picked = accents.find(swatch => swatch.hex === accent.toLowerCase())

  const pickAccent = (hex: string) => {
    const seed = hex === NOUS_ACCENT ? null : hex

    setOnboardingAnswers({ accent: seed })
    setAccentOverride(seed)
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
      </div>
    </CardFrame>
  )
}

export function LayoutCard({ locked }: CardProps) {
  const answers = useStore($onboardingAnswers)
  const { commit, done } = useCardCommit('layout')
  // The stored answer defaults to 'basic', but the CHOICE is the point of this
  // step — nothing renders selected (and Continue stays off) until they click.
  // Store-backed: the pick's own layout apply remounts this card (the pane
  // tree is replaced), so local state would drop the highlight instantly.
  const picked = useStore($chatLayoutPicked)

  const pickLayout = (id: string) => {
    $chatLayoutPicked.set(true)
    setOnboardingAnswers({ layout: id })

    // Live, behind the chat — the panes rearrange as the option is clicked.
    const preset = registry.getArea('layouts').find(contribution => contribution.id === id)

    if (!preset?.data) {
      return
    }

    // Every pick goes through assembly, including re-picks. The first grows
    // the window and places the panes, keeping the chat (and the cursor over
    // this card) pixel-fixed; later ones re-arrange in place. Swapping just the
    // preset tree on a re-pick left the previous layout's dismissals and dock
    // records in force, and the two layouts came up mixed together.
    // SAFETY: Layout presets declare data: LayoutNode (pane-shell/tree/presets.ts).
    assembleChatOnboarding(preset.id, preset.data as LayoutNode)
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
