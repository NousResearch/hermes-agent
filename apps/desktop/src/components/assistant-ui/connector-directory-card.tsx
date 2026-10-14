import { useStore } from '@nanostores/react'
import { useMemo, useState } from 'react'

import {
  $abandonedConnects,
  $accountOperations,
  type AccountOperation,
  accountOperationFor,
  retryable
} from '@/app/capabilities/connectors/data/account-operations'
import { useHostedActions } from '@/app/capabilities/connectors/data/hosted-actions'
import { useHostedConnectors } from '@/app/capabilities/connectors/data/queries'
import { deriveCards, forgetAbandoned } from '@/app/capabilities/connectors/derive'
import { HostedNotice } from '@/app/capabilities/connectors/hosted-notice'
import type { ConnectorCardModel } from '@/app/capabilities/connectors/types'
import { useSessionView } from '@/app/chat/session-view'
import { sessionRoute } from '@/app/routes'
import { orderConnectorPicks } from '@/components/assistant-ui/connector-order'
import { CARD_ACTION_CLASS, CARD_CHIP_CLASS, CARD_LABEL_CLASS } from '@/components/assistant-ui/setup-beat/card-type'
import { useConnectionOwner } from '@/components/assistant-ui/use-connection-owner'
import { WIDGET_SHELL_CLASS } from '@/components/chat/widget-shell'
import { Button } from '@/components/ui/button'
import { Chip } from '@/components/ui/chip'
import { ConnectorLogo } from '@/components/ui/connector-logo'
import { FadeScroll } from '@/components/ui/fade-scroll'
import { SearchField } from '@/components/ui/search-field'
import { useI18n } from '@/i18n'
import { connectorIconUrl } from '@/lib/connector-tools'
import { Plug } from '@/lib/icons'
import { cn } from '@/lib/utils'
import type { ConnectionOwner } from '@/store/connection-request'
import { $freeTierStatus } from '@/store/free-tier'

/** Same shape as the `manage_connections` display result. */
export interface ConnectorDirectoryResult {
  connected: string[]
  newly_connected: string[]
  suggested: string[]
}

export interface ConnectorDirectoryCardProps {
  /** Slugs pinned at the top, in this order; unknown slugs are left out. */
  suggested?: readonly string[]
  onContinue: (result: ConnectorDirectoryResult) => void
}

export interface DirectorySections {
  rest: ConnectorCardModel[]
  suggested: ConnectorCardModel[]
}

// The onboarding apps step showed search from this many apps on.
const SEARCH_THRESHOLD = 12

const matches = (card: ConnectorCardModel, needle: string): boolean =>
  !needle || card.name.toLowerCase().includes(needle) || card.slug.includes(needle)

/** Suggested apps first in the caller's order, then every other app in the onboarding picker's order (the most
 *  used apps lead; apps turned off and the picker's hidden few are left out). */
export function directorySections(
  cards: readonly ConnectorCardModel[],
  suggested: readonly string[],
  query: string
): DirectorySections {
  const needle = query.trim().toLowerCase()
  const pinned = new Set(suggested)

  const ordered = orderConnectorPicks(
    cards.map(card => ({ card, connector: card.slug, enabled: card.state !== 'off' }))
  ).map(row => row.card)

  // A suggestion cannot bring back an app the picker leaves out.
  const bySlug = new Map(ordered.map(card => [card.slug, card]))

  return {
    rest: ordered.filter(card => !pinned.has(card.slug) && matches(card, needle)),
    suggested: suggested.flatMap(slug => {
      const card = bySlug.get(slug)

      return card && matches(card, needle) ? [card] : []
    })
  }
}

const NO_SUGGESTIONS: readonly string[] = []

const connectedSlugs = (cards: readonly ConnectorCardModel[]): string[] =>
  cards.filter(card => card.state === 'connected').map(card => card.slug)

/** Every hosted connector in one chat card, laid out like the onboarding apps step; a tile runs the Connectors
 *  page's connect flow. It resolves the chat's own profile and connection, so it must render inside a session view. */
export function ConnectorDirectoryCard(props: ConnectorDirectoryCardProps) {
  const storedId = useStore(useSessionView().$storedId)
  const owner = useConnectionOwner(storedId, true)

  // The browser's return from a sign-in brings the user back to this chat, where the card waits.
  return owner && storedId ? <DirectoryBody {...props} owner={owner} returnRoute={sessionRoute(storedId)} /> : null
}

interface DirectoryBodyProps extends ConnectorDirectoryCardProps {
  owner: ConnectionOwner
  returnRoute: string
}

function DirectoryBody({ onContinue, owner, returnRoute, suggested = NO_SUGGESTIONS }: DirectoryBodyProps) {
  const { t } = useI18n()
  const copy = t.connectors
  const hosted = useHostedConnectors(owner)
  const abandoned = useStore($abandonedConnects)
  const operations = useStore($accountOperations)
  const freeTier = useStore($freeTierStatus)
  const actions = useHostedActions(owner, hosted.accounts, { returnRoute })
  const [query, setQuery] = useState('')

  const cards = useMemo(
    () => deriveCards({ hosted: forgetAbandoned(hosted.rows, new Set(abandoned)), local: [], titles: hosted.titles }),
    [abandoned, hosted.rows, hosted.titles]
  )

  const sections = useMemo(() => directorySections(cards, suggested, query), [cards, query, suggested])

  // What was connected when the list first answered; `newly_connected` is measured against it.
  const [baseline, setBaseline] = useState<null | ReadonlySet<string>>(null)

  if (baseline === null && hosted.phase === 'ready') {
    setBaseline(new Set(connectedSlugs(cards)))
  }

  const finish = () => {
    const connected = connectedSlugs(cards)

    onContinue({
      connected,
      newly_connected: connected.filter(slug => !baseline?.has(slug)),
      suggested: [...suggested]
    })
  }

  // An open sign-in decides both the tile's label and its click, so the two never disagree with a stale row.
  const openSignIn = (card: ConnectorCardModel) => {
    const operation = accountOperationFor(operations, card.slug, owner)

    return operation && !operation.settled ? operation : null
  }

  const tileSub = (card: ConnectorCardModel, busy: boolean, open: AccountOperation | null): string | undefined => {
    if (busy) {
      return copy.opening
    }

    if (open) {
      return retryable(open) ? copy.directory.signInFailed : copy.directory.finishInBrowser
    }

    return card.state === 'available' ? undefined : t.connectorsPage.card.state[card.stateWord]
  }

  // Failed: the same request again. Waiting: the link again. Otherwise the app's own verb.
  const press = (card: ConnectorCardModel, open: AccountOperation | null) => {
    if (!open) {
      actions.runHostedVerb(card)

      return
    }

    // Read before `retryable`: its type guard narrows `open` to never on the false branch.
    const url = open.targets.find(target => target.connectUrl)?.connectUrl

    if (retryable(open)) {
      actions.retry(card, open)
    } else if (url) {
      void window.hermesDesktop?.openExternal?.(url)
    }
  }

  const tile = (card: ConnectorCardModel) => {
    const busy = actions.busySlug === card.slug
    const open = openSignIn(card)

    return (
      <Chip
        className={cn('w-full', CARD_CHIP_CLASS)}
        icon={
          <ConnectorLogo
            className="size-7 rounded-full text-sm"
            connector={{ iconUrl: connectorIconUrl(card.slug), name: card.slug, title: card.name }}
          />
        }
        key={card.slug}
        label={card.name}
        on={card.state === 'connected'}
        onToggle={() => {
          if (!busy) {
            press(card, open)
          }
        }}
        sub={tileSub(card, busy, open)}
      />
    )
  }

  const pinned = sections.suggested.length > 0
  const nothingMatches = cards.length > 0 && !pinned && sections.rest.length === 0

  return (
    <div className="my-1.5 grid min-w-0 gap-2" data-connector-directory>
      <div
        className={cn(
          WIDGET_SHELL_CLASS,
          'grid gap-3 text-[length:var(--conversation-text-font-size)] text-(--ui-text-primary)'
        )}
      >
        <div className="flex items-start gap-2">
          <span className={cn('flex-1', CARD_LABEL_CLASS)}>{copy.title}</span>
          <Plug aria-hidden className="mt-px size-4 shrink-0 text-(--ui-text-tertiary)" />
        </div>

        <HostedNotice hasGuest={freeTier?.has_guest === true} phase={hosted.phase} />

        {hosted.phase === 'failed' ? (
          <p className={cn('flex items-center gap-1.5', CARD_LABEL_CLASS)}>
            {t.connectorsPage.page.hostedFailedTitle}
            <Button onClick={hosted.refetch} size="xs" variant="text">
              {t.connectorsPage.page.retry}
            </Button>
          </p>
        ) : null}

        {cards.length > SEARCH_THRESHOLD ? (
          <SearchField onChange={setQuery} placeholder={copy.search} value={query} />
        ) : null}

        {hosted.phase === 'loading' && cards.length === 0 ? (
          <div className="grid grid-cols-3 gap-2" role="status">
            <span className="sr-only">{t.connectorsPage.page.loading}</span>
            {Array.from({ length: 6 }, (_, index) => (
              <div className="h-10 animate-pulse rounded-lg bg-muted/40" key={index} />
            ))}
          </div>
        ) : (
          <>
            {pinned ? (
              <section className="grid gap-1.5">
                <span className={CARD_LABEL_CLASS}>{copy.directory.suggested}</span>
                <div className="grid grid-cols-3 gap-2 p-1">{sections.suggested.map(tile)}</div>
              </section>
            ) : null}
            {sections.rest.length > 0 ? (
              <section className="grid gap-1.5">
                {pinned ? <span className={CARD_LABEL_CLASS}>{copy.directory.allApps}</span> : null}
                <FadeScroll className="grid grid-cols-3 gap-2 p-1" fade="1.5rem" maxHeight="18rem" pad="1.5rem">
                  {sections.rest.map(tile)}
                </FadeScroll>
              </section>
            ) : null}
            {nothingMatches ? <p className={CARD_LABEL_CLASS}>{copy.empty}</p> : null}
          </>
        )}
      </div>
      <div className="px-3.5">
        {/* Continue reports the apps connected; while the list loads it would report none. */}
        <Button className={CARD_ACTION_CLASS} disabled={hosted.phase === 'loading'} onClick={finish} size="sm">
          {t.common.continue}
        </Button>
      </div>
    </div>
  )
}
