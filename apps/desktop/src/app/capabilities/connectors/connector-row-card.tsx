import { Button } from '@/components/ui/button'
import { ConnectorLogo } from '@/components/ui/connector-logo'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n/types'
import { cn } from '@/lib/utils'

import { CatalogMark } from './catalog-mark'
import { connectorKindWord, showsCatalogMark } from './connector-kind'
import type { ConnectorCardModel, ConnectorFact, ConnectorState } from './types'

const STATE_DOT = {
  available: 'bg-(--ui-text-quaternary)',
  broken: 'bg-(--ui-red)',
  connected: 'bg-(--ui-green)',
  connecting: 'bg-(--ui-yellow)',
  expired: 'bg-(--ui-orange)',
  off: 'bg-(--ui-text-quaternary)',
  unknown: 'bg-(--ui-text-quaternary)'
} satisfies Record<ConnectorState, string>

const REASON_TONE = {
  available: '',
  broken: 'text-(--ui-red)',
  connected: '',
  connecting: 'text-(--ui-text-secondary)',
  expired: 'text-(--ui-orange)',
  off: '',
  unknown: ''
} satisfies Record<ConnectorState, string>

function factText(copy: Translations['connectorsPage']['card'], fact: ConnectorFact): string {
  switch (fact.key) {
    case 'tools':
      return copy.fact.tools(fact.count)

    case 'toolsOff':
      return copy.fact.toolsOff(fact.count)

    case 'toolsOn':
      return copy.fact.toolsOn(fact.count)

    default:
      return copy.fact.toolsSomeOn(fact.count, fact.on ?? 0)
  }
}

export interface ConnectorRowCardProps {
  busy?: boolean
  card: ConnectorCardModel
  onOpen: () => void
  onServerToggle?: (next: boolean) => void
  onVerb?: () => void
  selected?: boolean
}

export function ConnectorRowCard({
  busy = false,
  card,
  onOpen,
  onServerToggle,
  onVerb,
  selected = false
}: ConnectorRowCardProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.card
  const unset = card.state === 'available'

  const stateLabel = card.fact ? factText(copy, card.fact) : copy.state[card.stateWord]

  const reason = card.reason ? (card.reason.text ?? copy.reason[card.reason.key]) : undefined

  return (
    <div
      className={cn(
        'relative flex items-center gap-3 rounded-lg border p-3 transition-colors duration-100',
        selected
          ? 'border-(--theme-primary) bg-(--ui-row-active-background)'
          : 'border-(--ui-stroke-quaternary) bg-(--ui-bg-elevated) hover:bg-(--chrome-action-hover)'
      )}
      data-connector={card.slug}
      data-slot="connector-row-card"
    >
      <ConnectorLogo
        className="size-9 shrink-0 rounded-[9px]"
        connector={{ name: card.slug, title: card.name }}
      />

      <div className="grid min-w-0 flex-1 gap-0.5">
        <div className="flex min-w-0 flex-wrap items-center gap-x-1.5 gap-y-0.5">
          <button
            className="min-w-0 truncate text-[0.8125rem] font-semibold text-(--ui-text-primary) outline-none after:absolute after:inset-0 after:rounded-lg focus-visible:after:ring-[0.1875rem] focus-visible:after:ring-ring/50"
            onClick={onOpen}
            type="button"
          >
            {card.name}
            <span className="sr-only">{` — ${copy.open(card.name)}`}</span>
          </button>

          <span className="shrink-0 text-[0.6875rem] text-(--ui-text-tertiary)">{connectorKindWord(card, copy)}</span>

          {showsCatalogMark(card) ? <CatalogMark /> : null}
        </div>

        <SecondLine card={card} reason={reason} />
      </div>

      <CardLane
        busy={busy}
        card={card}
        onServerToggle={card.plugin === undefined ? onServerToggle : undefined}
        onVerb={onVerb}
        stateLabel={stateLabel}
        withDot={!unset}
      />
    </div>
  )
}

function CardLane({
  busy,
  card,
  onServerToggle,
  onVerb,
  stateLabel,
  withDot
}: {
  busy: boolean
  card: ConnectorCardModel
  onServerToggle?: (next: boolean) => void
  onVerb?: () => void
  stateLabel: string
  withDot: boolean
}) {
  const { t } = useI18n()
  const copy = t.connectorsPage.card
  const server = card.ways

  return (
    <div className="relative z-10 flex w-[7.75rem] shrink-0 flex-col items-end gap-1">
      <span className="flex items-center gap-1.5 text-[0.6875rem] text-(--ui-text-secondary)">
        {withDot ? (
          <span aria-hidden className={cn('size-[5px] shrink-0 rounded-full', STATE_DOT[card.state])} />
        ) : null}
        <span className="truncate">{stateLabel}</span>
      </span>

      {server?.installed === true && onServerToggle ? (
        <Switch
          aria-label={server.serverEnabled ? copy.turnServerOff(card.name) : copy.turnServerOn(card.name)}
          checked={server.serverEnabled ?? false}
          onCheckedChange={onServerToggle}
          size="xs"
        />
      ) : null}

      {card.verb && onVerb ? (
        <Button
          disabled={busy}
          loading={busy}
          onClick={onVerb}
          size="xs"
          variant={card.state === 'available' ? 'outline' : 'secondary'}
        >
          {copy.verb[card.verb]}
        </Button>
      ) : null}

    </div>
  )
}

function SecondLine({ card, reason }: { card: ConnectorCardModel; reason?: string }) {
  if (reason) {
    return <p className={cn('truncate text-[0.72rem]', REASON_TONE[card.state])}>{reason}</p>
  }

  return card.description ? (
    <p className="line-clamp-2 text-[0.72rem] leading-snug text-(--ui-text-secondary)">{card.description}</p>
  ) : null
}
