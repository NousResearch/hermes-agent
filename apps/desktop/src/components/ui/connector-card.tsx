import { SCAFFOLD_META_CLASS, ScaffoldRow } from '@/components/chat/scaffold-row'
import { WIDGET_SHELL_CLASS } from '@/components/chat/widget-shell'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ConnectorLogo, type ConnectorLogoSubject } from '@/components/ui/connector-logo'
import { Input } from '@/components/ui/input'
import { Tip } from '@/components/ui/tooltip'
import { MarkdownLinkText } from '@/lib/external-link'
import { Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'

/**
 * The consent card for connecting a thing to Hermes, as pure presentation.
 *
 * Deliberately knows nothing about MCP. It renders a subject, a state, and an
 * outcome, and it calls back — what a connector IS, how one connects, and
 * where the strings come from all belong to the caller. That boundary is the
 * point: the transcript's inline setup card and any other surface that has to
 * ask "connect this?" should look identical without sharing a data layer.
 *
 * Consent vocabulary follows the tool approval bar: primary-tinted action,
 * quiet ghost decline, the same `mt-2` stand-off.
 */

/** Where the subject stands before anything is attempted. */
export type ConnectorCardState = 'connected' | 'disabled' | 'needs_auth' | 'not_configured'

/** How much the source vouches for the subject. */
export type ConnectorCardTrust = 'catalog' | 'community' | 'verified'

/** One credential the subject declares it needs. */
export interface ConnectorCardField {
  name: string
  prompt?: string
  required?: boolean
}

/** What the card renders. Structural, so a caller's richer type satisfies it
 *  without this file importing that type. */
export interface ConnectorCardSubject extends ConnectorLogoSubject {
  description?: string
  /** Named by the publisher and checked against the serving domain. */
  publisher?: string
  requiredEnv?: ConnectorCardField[]
  /** Ordered work only the user can do, elsewhere, before this can connect. */
  setup?: string[]
  title: string
  trust?: ConnectorCardTrust
}

/** How an attempt ended. */
export interface ConnectorCardOutcome {
  detail?: string
  /** The failure was a refusal, not a fault: wrong key, expired grant, or a
   *  grant that never covered the tools. Asking again is the fix, so the card
   *  offers access rather than a retry. */
  needsAuth?: boolean
  status: 'connected' | 'declined' | 'error'
  tools?: unknown[]
}

/** Every string the card shows. Passed in rather than read from i18n so the
 *  card stays a leaf that anything can render, tests included. */
export interface ConnectorCardCopy {
  connectAction: string
  decline: string
  envRequired: string
  grantAction: string
  retryAction: string
  stateConnected: string
  stateDeclined: string
  stateDisabled: string
  stateFailed: string
  stateNeedsAuth: string
  toolCount: (count: number) => string
  trustCommunity: string
  trustCommunityTip: (host: string) => string
  trustVerified: (publisher: string) => string
  trustVerifiedTip: (publisher: string) => string
}

const SHELL_CLASS = `${WIDGET_SHELL_CLASS} text-[length:var(--conversation-text-font-size)] text-(--ui-text-primary)`

const hostOf = (url: null | string | undefined): string => {
  if (!url) {
    return ''
  }

  try {
    return new URL(url).hostname
  } catch {
    return url
  }
}

/** What trails a settled subject's name: how it ended, and the useful number
 *  or reason behind that. */
export function outcomeMeta(outcome: ConnectorCardOutcome, copy: ConnectorCardCopy): string {
  if (outcome.status === 'connected') {
    const toolCount = Array.isArray(outcome.tools) ? outcome.tools.length : 0

    return toolCount > 0 ? `${copy.stateConnected} · ${copy.toolCount(toolCount)}` : copy.stateConnected
  }

  if (outcome.status === 'error') {
    return outcome.detail || copy.stateFailed
  }

  return copy.stateDeclined
}

/**
 * A subject that no longer needs anything: connected, skipped, failed.
 *
 * Not a card. An answered offer is transcript scaffolding — the same kind of
 * line as a settled tool run — so it renders through `ScaffoldRow` and reads
 * like everything else already done. The name keeps full contrast because
 * that's the content; the verdict trails it as meta.
 */
export function ConnectorSummary({
  connector,
  meta,
  tone
}: {
  connector: ConnectorLogoSubject
  meta?: string
  tone?: 'error'
}) {
  // The scaffold mark goes on the row, never on a container holding several:
  // opacity opens a stacking context and would pin every sibling to one level.
  return (
    <div data-conversation-scaffold="" data-slot="connector-card">
      <ScaffoldRow>
        <ConnectorLogo className="size-4 rounded-[0.25rem]" connector={connector} />
        <span className="truncate text-[length:var(--conversation-tool-font-size)] text-(--ui-text-primary)">
          {connector.title || connector.name}
        </span>
        {meta ? <span className={cn(SCAFFOLD_META_CLASS, tone === 'error' && 'text-destructive')}>{meta}</span> : null}
      </ScaffoldRow>
    </div>
  )
}

/**
 * How much the source vouches for this subject.
 *
 * Only the exceptions get a badge. Something we shipped in a reviewed catalog
 * is the ordinary case — badging it "reviewed" spends a word on every card to
 * say "normal", and a label whose meaning nobody can guess teaches the user to
 * ignore the one that matters.
 *
 * So: nothing for vetted sources. A publisher that proved it owns the serving
 * domain says "verified · notion.com" — checkable identity, not an
 * endorsement, which is why it names the domain instead of claiming trust.
 * Everything else gets the amber "unreviewed" with the host in its tooltip:
 * the card doesn't spend a line on an endpoint nobody reads, but for a
 * publisher nobody has vouched for, the host is the whole question.
 */
function TrustBadge({ connector, copy }: { connector: ConnectorCardSubject; copy: ConnectorCardCopy }) {
  if (!connector.trust || connector.trust === 'catalog') {
    return null
  }

  if (connector.trust === 'verified') {
    // No publisher domain means a curated directory of vendor remotes, which
    // is the ordinary case again — nothing to say.
    if (!connector.publisher) {
      return null
    }

    return (
      <Tip label={copy.trustVerifiedTip(connector.publisher)}>
        <span className="text-[0.6875rem] text-(--ui-text-tertiary)">{copy.trustVerified(connector.publisher)}</span>
      </Tip>
    )
  }

  return (
    <Tip label={copy.trustCommunityTip(hostOf(connector.url))}>
      <span className="inline-flex items-center gap-1 text-[0.6875rem] text-amber-500">
        <Codicon name="warning" size="0.6875rem" />
        {copy.trustCommunity}
      </span>
    </Tip>
  )
}

export interface ConnectorCardProps {
  connector: ConnectorCardSubject
  copy: ConnectorCardCopy
  /** Waved off by the user. Collapses to the same settled line as success. */
  dismissed?: boolean
  envDraft?: Record<string, string>
  /** Whether the credential fields are revealed. The caller owns this because
   *  a refused credential should reveal them without a second click. */
  envOpen?: boolean
  onConnect: () => void
  onDismiss: () => void
  onEnvChange?: (key: string, value: string) => void
  /** A sibling card is mid-flight. Two sign-in tabs racing for focus is
   *  hostile, so the action waits — but the decline never does. */
  otherBusy?: boolean
  outcome?: ConnectorCardOutcome
  /** Present only while working; replaces the resting state label. */
  phase?: string
  state: ConnectorCardState
}

/**
 * One subject's consent card.
 *
 * Owns everything about that subject and nothing about its siblings: its own
 * trust badge, credential fields, failure reason, and its own action. A
 * connected card collapses to a single confirmed line, because its offer is
 * spent and the space belongs to the ones still asking.
 */
export function ConnectorCard({
  connector,
  copy,
  dismissed = false,
  envDraft = {},
  envOpen = false,
  onConnect,
  onDismiss,
  onEnvChange,
  otherBusy = false,
  outcome,
  phase,
  state
}: ConnectorCardProps) {
  const working = phase !== undefined
  const connected = outcome?.status === 'connected'
  const failed = outcome?.status === 'error'

  // Answered: the offer is spent, so the card collapses to a scaffold line and
  // gives the space back to whatever is still asking.
  if (connected || dismissed) {
    return <ConnectorSummary connector={connector} meta={outcomeMeta(outcome ?? { status: 'declined' }, copy)} />
  }

  const stateLabel = state === 'disabled' ? copy.stateDisabled : state === 'needs_auth' ? copy.stateNeedsAuth : null

  // Before the first connect, and again after a failed one. Something that
  // connected doesn't need its console instructions repeated — but a failure
  // usually happened BECAUSE of that console: an API left un-enabled, the
  // wrong client type, a secret copied one character short. Withdrawing the
  // steps and the fields at the moment they're finally needed is backwards,
  // and it leaves a wrong credential with nowhere to be corrected.
  const fixable = state === 'not_configured' || failed
  const envFields = fixable ? (connector.requiredEnv ?? []) : []
  const steps = fixable ? (connector.setup ?? []) : []

  // Logo owns the left rail; everything the card says and every control it
  // offers shares the one text column, so the buttons sit on the copy's grid
  // line instead of hanging off the card's edge under the mark.
  return (
    <div className={cn(SHELL_CLASS, 'flex items-start gap-3')} data-slot="connector-card">
      <ConnectorLogo connector={connector} />

      <div className="grid min-w-0 flex-1 gap-0.5">
        <div className="flex flex-wrap items-baseline gap-x-1.5">
          <span className="font-medium">{connector.title}</span>
          {/* While the card is working its phase replaces the resting state —
              "Signing in…" is the one the user needs, because the browser tab
              that just took focus is otherwise unexplained. */}
          {working ? (
            <span className="text-[0.6875rem] text-(--ui-text-tertiary)">{phase}</span>
          ) : (
            stateLabel && <span className="text-[0.6875rem] text-(--ui-text-tertiary)">{stateLabel}</span>
          )}
          <TrustBadge connector={connector} copy={copy} />
        </div>

        {connector.description ? <p className="text-(--ui-text-secondary)">{connector.description}</p> : null}

        {failed && outcome.detail ? <p className="text-[0.6875rem] text-destructive">{outcome.detail}</p> : null}

        {/* The part we cannot do. Numbered because order matters, linked
            because the whole cost of these steps is finding the page. */}
        {steps.length > 0 && (
          <ol className="mt-1.5 grid gap-1" data-slot="connector-card-steps">
            {steps.map((step, index) => (
              <li className="flex gap-1.5 text-[0.6875rem] text-(--ui-text-secondary)" key={step}>
                <span className="tabular-nums text-(--ui-text-tertiary)">{index + 1}.</span>
                <MarkdownLinkText text={step} />
              </li>
            ))}
          </ol>
        )}

        {envOpen && envFields.length > 0 && (
          <div className="mt-1 grid gap-2" data-slot="connector-card-env">
            <p className="text-[0.6875rem] text-(--ui-text-tertiary)">{copy.envRequired}</p>
            {envFields.map(env => (
              <label className="grid gap-1" key={env.name}>
                <span className="text-[0.6875rem] text-(--ui-text-secondary)">
                  {env.prompt || env.name}
                  {env.required ? ' *' : ''}
                </span>
                <Input
                  className="h-7 text-xs"
                  onChange={event => onEnvChange?.(env.name, event.currentTarget.value)}
                  type="password"
                  value={envDraft[env.name] ?? ''}
                />
              </label>
            ))}
          </div>
        )}

        {/* Same strip as the tool approval bar (tool/approval.tsx), down to its
            `mt-2` stand-off: a bordered primary-tinted action plus a quiet
            ghost decline. One consent vocabulary across the transcript. */}
        <div className="mt-2 flex items-center gap-2.5">
          <div className="inline-flex h-6 items-stretch overflow-hidden rounded-md border border-primary/25 bg-primary/10 text-primary">
            <Button
              className="h-full gap-1 rounded-none px-2 text-xs font-medium text-primary hover:bg-primary/15 hover:text-primary"
              disabled={working || otherBusy}
              onClick={onConnect}
              size="xs"
              variant="ghost"
            >
              {working ? (
                <Loader2 className="size-3 animate-spin" />
              ) : outcome?.needsAuth ? (
                copy.grantAction
              ) : failed ? (
                copy.retryAction
              ) : (
                copy.connectAction
              )}
            </Button>
          </div>
          {/* Never disabled: while a connect is in flight this is the way out
              of a stuck sign-in tab or a hung install. */}
          <Button
            className="h-6 gap-1.5 rounded-md px-1.5 text-xs font-normal text-(--ui-text-tertiary) hover:text-foreground"
            onClick={onDismiss}
            size="xs"
            variant="ghost"
          >
            {copy.decline}
          </Button>
        </div>
      </div>
    </div>
  )
}
