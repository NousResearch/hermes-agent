/**
 * The three setup picks — accent, connectors, layout.
 *
 * Picks apply live. The shared catalog keeps cards and previews in agreement
 * without asking the model to enumerate the options.
 */

import { useStore } from '@nanostores/react'

import { $chatLayoutPicked, assembleChatOnboarding } from '@/components/onboarding-chat/assembly'
import { CardFrame, type CardProps, useCardCommit } from '@/components/onboarding-chat/cards/frame'
import { Chip } from '@/components/onboarding-chat/chip'
import {
  accentsFor,
  AccentSwatch,
  CONNECTORS,
  LayoutPreviewCard,
  LAYOUTS,
  NOUS_ACCENT
} from '@/components/onboarding-chat/options'
import type { LayoutNode } from '@/components/pane-shell/tree/model'
import { ConnectorLogo } from '@/components/ui/connector-logo'
import { registry } from '@/contrib/registry'
import { $onboardingAnswers, setOnboardingAnswers } from '@/store/onboarding-answers'
import { useTheme } from '@/themes'
import { setAccentOverride } from '@/themes/accent-override'

export function ConnectorsCard({ locked }: CardProps) {
  const answers = useStore($onboardingAnswers)
  const { commit, done } = useCardCommit()

  const toggle = (id: string) =>
    setOnboardingAnswers({
      connectors: answers.connectors.includes(id)
        ? answers.connectors.filter(item => item !== id)
        : [...answers.connectors, id]
    })

  return (
    <CardFrame
      done={done}
      locked={locked}
      onContinue={() => {
        const picked = CONNECTORS.filter(connector => answers.connectors.includes(connector.id))

        commit(
          `apps I use, not connected yet: ${picked.length > 0 ? picked.map(c => c.name).join(', ') : 'none for now'}`
        )
      }}
    >
      <div className="grid grid-cols-3 gap-2">
        {CONNECTORS.map(connector => (
          <Chip
            icon={
              <ConnectorLogo
                className="size-7 rounded-full text-sm"
                connector={{ homepage: connector.homepage, name: connector.id, title: connector.name }}
              />
            }
            key={connector.id}
            label={connector.name}
            on={answers.connectors.includes(connector.id)}
            onToggle={() => toggle(connector.id)}
          />
        ))}
      </div>
    </CardFrame>
  )
}

export function LookCard({ locked }: CardProps) {
  const answers = useStore($onboardingAnswers)
  const { renderedMode } = useTheme()
  const { commit, done } = useCardCommit()
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
  const { commit, done } = useCardCommit()
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
