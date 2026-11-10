import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ConnectorLogo } from '@/components/ui/connector-logo'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n/types'
import { connectorIconUrl } from '@/lib/connector-tools'
import { cn } from '@/lib/utils'

import { CatalogMark } from './catalog-mark'
import { connectorKindWord, showsCatalogMark } from './connector-kind'
import { twinPillOf } from './derive'
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
  onPrefetch?: () => void
  onServerToggle?: (next: boolean) => void
  onVerb?: () => void
  selected?: boolean
}

export function ConnectorRowCard({
  busy = false,
  card,
  onOpen,
  onPrefetch,
  onServerToggle,
  onVerb,
  selected = false
}: ConnectorRowCardProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.card
  const local = card.residency === 'local'
  const unset = card.state === 'available'

  const stateLabel = local && card.fact ? factText(copy, card.fact) : copy.state[card.stateWord]

  const reason = card.reason ? (card.reason.text ?? copy.reason[card.reason.key]) : undefined
  const twin = twinPillOf(card)

  return (
    <div
      className="relative flex items-center justify-between gap-3 p-4 transition-colors duration-100"
      data-connector={card.slug}
      data-connectors-card
      data-selected={selected ? '' : undefined}
      data-slot="connector-row-card"
      onFocus={onPrefetch}
      onPointerEnter={onPrefetch}
    >
      <div className="flex min-w-0 flex-1 items-center gap-3.5">
        <ConnectorLogo
          className="size-11 shrink-0 rounded-xl"
          connector={{ iconUrl: local ? undefined : connectorIconUrl(card.slug), name: card.slug, title: card.name }}
        />

        <div className="grid min-w-0 flex-1 gap-1.5">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <button
              className="min-w-0 truncate text-[15px] font-semibold text-[#0f172a] outline-none after:absolute after:inset-0 after:rounded-2xl focus-visible:after:ring-[0.1875rem] focus-visible:after:ring-ring/50 dark:text-foreground"
              onClick={onOpen}
              type="button"
            >
              {card.name}
              <span className="sr-only">{` — ${copy.open(card.name)}`}</span>
            </button>

            <span className="shrink-0 rounded-lg border border-[#e2e8f0] bg-[#f1f5f9] px-2 py-[3px] text-[11px] font-medium text-[#64748b]">
              {connectorKindWord(card, copy)}
            </span>

            {showsCatalogMark(card) ? <CatalogMark /> : null}

            {twin ? (
              <Badge className="shrink-0" size="xs" variant="muted">
                {twin === 'hostedTwin' ? copy.hostedTwin : copy.alsoLocal}
              </Badge>
            ) : null}
          </div>

          <SecondLine card={card} reason={reason} />
        </div>
      </div>

      <CardLane
        busy={busy}
        card={card}
        onServerToggle={local && card.plugin === undefined ? onServerToggle : undefined}
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
  const server = card.ways.local
  const fact = card.residency === 'hosted' && card.fact ? factText(copy, card.fact) : null
  const showSwitch = server?.installed === true && onServerToggle !== undefined
  const showVerb = Boolean(card.verb && onVerb)

  return (
    <div className="relative z-10 flex shrink-0 flex-col items-end gap-2">
      <span className="flex items-center gap-1.5 text-[12px] font-medium text-[#64748b]">
        {withDot ? (
          <span aria-hidden className={cn('size-[5px] shrink-0 rounded-full', STATE_DOT[card.state])} />
        ) : null}
        <span className="truncate">{stateLabel}</span>
      </span>

      {showSwitch && showVerb ? (
        <div className="flex flex-nowrap items-center gap-2">
          <VerbButton busy={busy} card={card} label={copy.verb[card.verb!]} onVerb={onVerb!} />
          <ServerSwitch card={card} copy={copy} onServerToggle={onServerToggle} server={server} />
        </div>
      ) : (
        <>
          {showSwitch ? <ServerSwitch card={card} copy={copy} onServerToggle={onServerToggle} server={server} /> : null}
          {showVerb ? <VerbButton busy={busy} card={card} label={copy.verb[card.verb!]} onVerb={onVerb!} /> : null}
        </>
      )}

      {card.verb === undefined && fact !== null ? (
        <span className="truncate text-[12px] text-[#94a3b8]">{fact}</span>
      ) : null}
    </div>
  )
}

function ServerSwitch({
  card,
  copy,
  onServerToggle,
  server
}: {
  card: ConnectorCardModel
  copy: Translations['connectorsPage']['card']
  onServerToggle: (next: boolean) => void
  server: ConnectorCardModel['ways']['local']
}) {
  return (
    <Switch
      aria-label={server?.serverEnabled ? copy.turnServerOff(card.name) : copy.turnServerOn(card.name)}
      checked={server?.serverEnabled ?? false}
      className="data-[state=checked]:border-transparent data-[state=checked]:bg-[#7c3aed]"
      onCheckedChange={onServerToggle}
      size="xs"
    />
  )
}

function VerbButton({
  busy,
  card,
  label,
  onVerb
}: {
  busy: boolean
  card: ConnectorCardModel
  label: string
  onVerb: () => void
}) {
  return (
    <Button
      className={cn(
        card.state === 'available' &&
          'rounded-full border border-[#e2e8f0] bg-white px-3.5 py-2 text-[13px] font-semibold text-[#0f172a] shadow-none hover:bg-[#f8fafc]'
      )}
      disabled={busy}
      loading={busy}
      onClick={onVerb}
      size="xs"
      variant={card.state === 'available' ? 'outline' : 'secondary'}
    >
      {label}
    </Button>
  )
}

function SecondLine({ card, reason }: { card: ConnectorCardModel; reason?: string }) {
  if (reason) {
    return <p className={cn('line-clamp-2 text-[13px] leading-normal', REASON_TONE[card.state])}>{reason}</p>
  }

  return card.description ? (
    <p className="line-clamp-2 text-[13px] leading-normal text-[#475569]">{card.description}</p>
  ) : null
}
