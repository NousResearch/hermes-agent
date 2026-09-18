// One app, one wide row-card, the same anatomy in every group: a mark, a body
// column that says what the app is, and a trailing lane of fixed width so the
// state words and the actions line up down the whole page.
//
// A card never says why twice. The state lane carries one word (or, for a server
// on this Mac, the one count a person can act on), and the second line carries
// either the description, the reason it broke, or the endpoint — never two.

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ConnectorLogo } from '@/components/ui/connector-logo'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n/types'
import { connectorIconUrl } from '@/lib/connector-tools'
import { cn } from '@/lib/utils'

import { CatalogMark } from './catalog-mark'
import type { ConnectorCardModel, ConnectorFact, ConnectorState } from './types'

/** Available and off both read as "nothing is happening here", so they share the
 *  quietest mark on the page. Only the three states that cost the person
 *  something are coloured. */
const STATE_DOT = {
  available: 'bg-(--ui-text-quaternary)',
  broken: 'bg-(--ui-red)',
  connected: 'bg-(--ui-green)',
  connecting: 'bg-(--ui-yellow)',
  expired: 'bg-(--ui-orange)',
  off: 'bg-(--ui-text-quaternary)'
} satisfies Record<ConnectorState, string>

const REASON_TONE = {
  available: '',
  broken: 'text-(--ui-red)',
  connected: '',
  connecting: 'text-(--ui-text-secondary)',
  expired: 'text-(--ui-orange)',
  off: ''
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
  /** True while this card's own write is in flight. The card paints nothing
   *  before the backend answers, so the verb says it is busy instead. */
  busy?: boolean
  card: ConnectorCardModel
  onOpen: () => void
  /** Local cards only. The switch is on the card, so the dialog is never needed
   *  to turn a server off. */
  onServerToggle?: (next: boolean) => void
  onVerb?: () => void
  /** The card whose dialog is open, highlighted behind it. */
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
  const local = card.residency === 'local'
  // A working server says how much of itself is live where a hosted app says
  // "Connected": the count is the more useful of the two, and both are one line.
  const stateLabel = local && card.fact ? factText(copy, card.fact) : copy.state[card.stateWord]
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
        connector={{ iconUrl: local ? undefined : connectorIconUrl(card.slug), name: card.slug, title: card.name }}
      />

      <div className="grid min-w-0 flex-1 gap-0.5">
        <div className="flex min-w-0 items-center gap-1.5">
          {/* The name is the one control that opens the app, and its pseudo
              element covers the card so the whole row is clickable without
              nesting a second button inside a button. */}
          <button
            className="truncate text-[0.8125rem] font-semibold text-(--ui-text-primary) outline-none after:absolute after:inset-0 after:rounded-lg focus-visible:after:ring-[0.1875rem] focus-visible:after:ring-ring/50"
            onClick={onOpen}
            type="button"
          >
            {card.name}
            <span className="sr-only">{` — ${copy.open(card.name)}`}</span>
          </button>

          {local ? null : <span className="shrink-0 text-[0.6875rem] text-(--ui-text-tertiary)">{copy.hosted}</span>}

          {card.inCatalog ? <CatalogMark /> : null}

          {card.hostedTwinAvailable ? (
            <Badge className="shrink-0" size="xs" variant="muted">
              {copy.hostedTwin}
            </Badge>
          ) : null}
        </div>

        <SecondLine card={card} reason={reason} />
      </div>

      <div className="relative z-10 flex w-[7.75rem] shrink-0 flex-col items-end gap-1">
        <span className="flex items-center gap-1.5 text-[0.6875rem] text-(--ui-text-secondary)">
          <span aria-hidden className={cn('size-[5px] shrink-0 rounded-full', STATE_DOT[card.state])} />
          <span className="truncate">{stateLabel}</span>
        </span>

        {/* The lane is a column, not a slot: a server on this Mac that needs
            signing in carries BOTH its switch and its repair. Hanging the verb
            off the same branch as the switch is what made `Authenticate` and
            `Open logs` unpressable on the only cards that can ask for them. */}
        {local && onServerToggle ? (
          <Switch
            aria-label={card.serverEnabled ? copy.turnServerOff(card.name) : copy.turnServerOn(card.name)}
            checked={card.serverEnabled ?? false}
            onCheckedChange={onServerToggle}
            size="xs"
          />
        ) : null}

        {card.verb && onVerb ? (
          <Button
            disabled={busy}
            onClick={onVerb}
            size="xs"
            variant={card.state === 'available' ? 'outline' : 'secondary'}
          >
            {copy.verb[card.verb]}
          </Button>
        ) : !local && card.fact ? (
          <span className="truncate text-[0.6875rem] text-(--ui-text-tertiary)">{factText(copy, card.fact)}</span>
        ) : null}
      </div>
    </div>
  )
}

/** One line under the name, and only one: the reason it broke if it broke, the
 *  endpoint if it runs here, otherwise what the app is for. */
function SecondLine({ card, reason }: { card: ConnectorCardModel; reason?: string }) {
  if (reason) {
    return <p className={cn('truncate text-[0.72rem]', REASON_TONE[card.state])}>{reason}</p>
  }

  if (card.residency === 'local' && card.target) {
    return <p className="truncate font-mono text-[0.65rem] text-(--ui-text-tertiary)">{card.target}</p>
  }

  return <p className="line-clamp-2 text-[0.72rem] leading-snug text-(--ui-text-secondary)">{card.description}</p>
}
