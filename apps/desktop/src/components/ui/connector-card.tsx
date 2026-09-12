import { SCAFFOLD_META_CLASS, ScaffoldRow } from '@/components/chat/scaffold-row'
import { WIDGET_SHELL_CLASS } from '@/components/chat/widget-shell'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ConnectorLogo, type ConnectorLogoSubject } from '@/components/ui/connector-logo'
import { Input } from '@/components/ui/input'
import { Tip } from '@/components/ui/tooltip'
import { MarkdownLinkText } from '@/lib/external-link'
import { CheckCircle2 } from '@/lib/icons'
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
  /** The live offer's heading as a question — "Connect Gmail?" — the same
   *  shape the MCP setup card asks in. Absent, the card leads with the name. */
  connectTitle?: (title: string) => string
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

/** What connecting actually means — the endpoint that will be contacted,
 *  or the catalog it came from. VS Code's trust dialog links the config it
 *  is about to trust; same idea. Shown under the description in tertiary. */
export interface ConnectorCardSource {
  text: string
}

const SHELL_CLASS = `${WIDGET_SHELL_CLASS} text-[length:var(--conversation-text-font-size)] text-(--ui-text-primary)`

// Same platform sniff the approval bar uses for its accelerator hint.
const isMac = typeof navigator !== 'undefined' && /Mac|iP(hone|ad|od)/.test(navigator.platform)

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
  /** `ok` is the settled tool row's emerald; `error` its destructive. Absent
   *  is the neutral grey a skip or a no-answer reads in. */
  tone?: 'error' | 'ok'
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
        {meta ? (
          <span
            className={cn(
              SCAFFOLD_META_CLASS,
              tone === 'error' && 'text-destructive',
              tone === 'ok' && 'text-emerald-600/85 dark:text-emerald-400/85'
            )}
          >
            {meta}
          </span>
        ) : null}
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
  /** Show ⌘⏎ / Esc beside the actions. Only the card that also LISTENS for
   *  those keys should claim them; a hint on a card that ignores the key is
   *  a lie the user finds out about by pressing it. */
  accelerators?: boolean
  /** Settled cards fold to a scaffold line (the MCP setup default). The
   *  connector offer keeps the card standing with a green Connected in the
   *  action slot: a row of identical cards where one collapses reads as a
   *  row where one broke. */
  collapseWhenSettled?: boolean
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
  actionDisabled?: boolean
  outcome?: ConnectorCardOutcome
  /** Present only while working; replaces the resting state label. */
  phase?: string
  source?: ConnectorCardSource
  state: ConnectorCardState
  /** `avatar` leads with the mark at identity scale in a left gutter, the way
   *  the MCP and Messaging headers introduce a service. `compact` (default)
   *  trails a small mark on the right like the setup card's tool row. */
  variant?: 'avatar' | 'compact'
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
  accelerators = false,
  collapseWhenSettled = true,
  connector,
  copy,
  dismissed = false,
  envDraft = {},
  envOpen = false,
  onConnect,
  onDismiss,
  onEnvChange,
  otherBusy = false,
  actionDisabled = false,
  outcome,
  phase,
  source,
  state,
  variant = 'compact'
}: ConnectorCardProps) {
  const working = phase !== undefined
  const connected = outcome?.status === 'connected'
  const failed = outcome?.status === 'error'

  // Answered: the offer is spent, so the card collapses to a scaffold line and
  // gives the space back to whatever is still asking.
  if ((connected || dismissed) && collapseWhenSettled) {
    return (
      <ConnectorSummary
        connector={connector}
        meta={outcomeMeta(outcome ?? { status: 'declined' }, copy)}
        tone={connected ? 'ok' : undefined}
      />
    )
  }

  const settled = connected || dismissed

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

  // Same shape as the MCP setup card, which is the consent widget users have
  // already met: the ask as a heading, one line of what it means, and the
  // action strip on the shell's own left edge. The mark either trails small
  // on the right (compact) or leads at identity scale in a left gutter
  // (avatar) — the MCP tab and Messaging headers' `items-start gap-3` row,
  // sized up so a first-time user sees WHOSE sign-in is about to open.
  const avatar = variant === 'avatar'

  return (
    <div className={cn(SHELL_CLASS, 'my-1.5 grid gap-1.5')} data-slot="connector-card">
      <div className={cn('flex items-start', avatar ? 'gap-3' : 'gap-2')}>
        {avatar ? <ConnectorLogo className="size-10 rounded-xl text-base" connector={connector} /> : null}
        <div className="grid min-w-0 flex-1 gap-0.5">
          <div className="flex flex-wrap items-baseline gap-x-1.5">
            <span className="font-medium leading-(--conversation-line-height)">
              {copy.connectTitle ? copy.connectTitle(connector.title) : connector.title}
            </span>
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

          {source ? <p className="truncate text-[0.6875rem] text-(--ui-text-tertiary)">{source.text}</p> : null}

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
        </div>
        {avatar ? null : (
          <ConnectorLogo className="mt-px size-5 rounded-[0.3rem] text-[0.6875rem]" connector={connector} />
        )}
      </div>

      {/* Same strip as the tool approval bar (tool/approval.tsx), down to its
          stand-off: a bordered primary-tinted action plus a quiet ghost
          decline. One consent vocabulary across the transcript. In the avatar
          layout the strip sits on the text column, so the gutter stays the
          mark's alone. Settled (and kept standing), the action slot holds the
          verdict in the same box — green Connected, grey Skipped — so the row
          of cards keeps its rhythm and nothing jumps. */}
      <div className={cn('flex items-center gap-2.5', avatar && 'pl-13')}>
        {settled ? (
          <div
            className={cn(
              'inline-flex h-6 items-stretch overflow-hidden rounded-md border',
              connected
                ? 'border-emerald-600/25 bg-emerald-600/10 text-emerald-700 dark:border-emerald-400/25 dark:bg-emerald-400/10 dark:text-emerald-300'
                : 'border-(--ui-stroke-tertiary) bg-(--ui-bg-quaternary) text-(--ui-text-tertiary)'
            )}
            data-slot="connector-card-verdict"
          >
            <span className="inline-flex h-full items-center gap-1 px-2 text-xs font-medium">
              {connected ? <CheckCircle2 aria-hidden className="size-3" /> : null}
              {outcomeMeta(outcome ?? { status: 'declined' }, copy)}
            </span>
          </div>
        ) : (
          <>
            <div className="inline-flex h-6 items-stretch overflow-hidden rounded-md border border-primary/25 bg-primary/10 text-primary">
              <Button
                className="h-full gap-1 rounded-none px-2 text-xs font-medium text-primary hover:bg-primary/15 hover:text-primary"
                disabled={otherBusy || actionDisabled}
                loading={working}
                onClick={onConnect}
                size="xs"
                variant="ghost"
              >
                {outcome?.needsAuth ? copy.grantAction : failed ? copy.retryAction : copy.connectAction}
                {accelerators ? (
                  <span className="text-[0.625rem] text-primary/60">{isMac ? '⌘⏎' : 'Ctrl⏎'}</span>
                ) : null}
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
              {accelerators ? <span className="text-[0.625rem] opacity-55">Esc</span> : null}
            </Button>
          </>
        )}
      </div>
    </div>
  )
}
