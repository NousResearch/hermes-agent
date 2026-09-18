// One dialog for one app: what it is on the left, every tool on the right.
//
// The shell is the same for a hosted app and for a server on this Mac; only the
// left column changes, because the two answer different questions. A hosted app
// is asked "who am I here and may Hermes use it"; a server is asked "where does
// it run, what does it cost, and where is its config".
//
// Nothing here fetches. The right column is a slot the caller fills with
// `ToolsList`, and the connect flow is the shipped connector element, cloned —
// never redrawn — so the page and the chat card can never disagree.

import { type ReactNode, type RefObject, useRef } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ConnectorLogo } from '@/components/ui/connector-logo'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog'
import { Separator } from '@/components/ui/separator'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import { connectorIconUrl } from '@/lib/connector-tools'
import { cn } from '@/lib/utils'

import { CatalogMark } from './catalog-mark'
import type { ConnectorCardModel, ConnectorState } from './types'

type BadgeVariant = 'default' | 'destructive' | 'muted' | 'success' | 'warn'

const STATE_BADGE = {
  available: 'muted',
  broken: 'destructive',
  connected: 'success',
  connecting: 'warn',
  expired: 'warn',
  off: 'muted'
} satisfies Record<ConnectorState, BadgeVariant>

export interface ConnectorDialogProps {
  /** A local server's mcp.json entry, logs and Remove. Lives here and only here. */
  advanced?: ReactNode
  /** The signed-in identity Hermes acts as. */
  accountLabel?: string
  card: ConnectorCardModel
  /** Already formatted for the reader's locale by the caller. */
  connectedOn?: string
  /** The shipped connector element, cloned, while a connect is in flight. */
  connectElement?: ReactNode
  cost?: { tokensPerCall?: string; usesPerMonth?: string }
  /** The kebab's menu. Absent means no kebab. */
  menu?: ReactNode
  onDisconnect?: () => void
  onOpenAdmin?: () => void
  onOpenChange: (open: boolean) => void
  onServerToggle?: (next: boolean) => void
  onToggleForMe?: (next: boolean) => void
  open: boolean
  /** How many tools the organisation took away. Zero means no note. */
  orgDisabledCount?: number
  profileName?: string
  /** True while `On for me` is being written. The switch is never optimistic —
   *  the policy write can take the full connector deadline — so it says it is
   *  busy instead of moving before the backend agreed. */
  togglePending?: boolean
  /** The right column: `ToolsList`. */
  tools: ReactNode
}

export function ConnectorDialog({ card, onOpenChange, open, tools, ...rest }: ConnectorDialogProps) {
  const { t } = useI18n()
  const local = card.residency === 'local'
  const titleRef = useRef<HTMLHeadingElement>(null)

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent
        bodyClassName="gap-0 overflow-hidden p-0"
        className="h-[min(55rem,85vh)] min-w-[min(62.5rem,92vw)]"
        fitContent
        // Radix focuses the first tabbable node, which here is the server switch
        // or Disconnect: opening a dialog must not put the keyboard on a control
        // that turns something off. The title takes the focus instead, so the
        // dialog still answers keys and Tab starts at the top of the content.
        onOpenAutoFocus={event => {
          event.preventDefault()
          titleRef.current?.focus()
        }}
      >
        <Header card={card} menu={rest.menu} onServerToggle={rest.onServerToggle} titleRef={titleRef} />

        <div className="grid min-h-0 flex-1 grid-cols-[18.75rem_minmax(0,1fr)]">
          <div className="flex min-h-0 flex-col gap-3 overflow-y-auto border-r border-(--ui-stroke-tertiary) p-4">
            {local ? <LocalColumn card={card} {...rest} /> : <HostedColumn card={card} {...rest} />}
          </div>

          <div className="flex min-h-0 flex-col">{tools}</div>
        </div>

        <span className="sr-only">{t.connectorsPage.title}</span>
      </DialogContent>
    </Dialog>
  )
}

function Header({
  card,
  menu,
  onServerToggle,
  titleRef
}: {
  card: ConnectorCardModel
  menu?: ReactNode
  onServerToggle?: (next: boolean) => void
  titleRef: RefObject<HTMLHeadingElement | null>
}) {
  const { t } = useI18n()
  const copy = t.connectorsPage.card
  const local = card.residency === 'local'

  return (
    <header className="flex shrink-0 items-center gap-2.5 border-b border-(--ui-stroke-tertiary) px-5 py-3">
      <ConnectorLogo
        className="size-9 shrink-0 rounded-[9px]"
        connector={{ iconUrl: local ? undefined : connectorIconUrl(card.slug), name: card.slug, title: card.name }}
      />

      <div className="grid min-w-0 flex-1 gap-0.5">
        <div className="flex min-w-0 items-center gap-2">
          <DialogTitle className="truncate text-base font-semibold outline-none" ref={titleRef} tabIndex={-1}>
            {card.name}
          </DialogTitle>

          <span className="shrink-0 text-[0.6875rem] text-(--ui-text-tertiary)">
            {local
              ? t.connectorsPage.dialog[card.target?.startsWith('http') ? 'transportUrl' : 'transportProgram']
              : copy.hosted}
          </span>

          {card.inCatalog ? <CatalogMark /> : null}

          <Badge className="shrink-0" size="xs" variant={STATE_BADGE[card.state]}>
            {copy.state[card.stateWord]}
          </Badge>
        </div>

        <DialogDescription className="truncate text-[0.72rem] text-(--ui-text-secondary)">
          {card.description ?? card.target ?? ''}
        </DialogDescription>
      </div>

      {local && onServerToggle ? (
        <Switch
          aria-label={t.connectorsPage.dialog.serverSwitch(card.name)}
          checked={card.serverEnabled ?? false}
          onCheckedChange={onServerToggle}
          size="xs"
        />
      ) : null}

      {/* The close button sits at the shell's top right; the kebab keeps clear of it. */}
      {menu ? <div className="mr-7 shrink-0">{menu}</div> : null}
    </header>
  )
}

type ColumnProps = Omit<ConnectorDialogProps, 'onOpenChange' | 'open' | 'tools'>

function HostedColumn({
  accountLabel,
  card,
  connectedOn,
  connectElement,
  onDisconnect,
  onOpenAdmin,
  onToggleForMe,
  orgDisabledCount = 0,
  togglePending = false
}: ColumnProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.dialog

  // A connect in flight owns the whole column: there is no account yet to
  // describe and no rule worth editing until the browser hands the person back.
  if (connectElement) {
    return <>{connectElement}</>
  }

  return (
    <>
      {accountLabel ? <p className="text-[0.78rem] text-(--ui-text-primary)">{copy.actsAs(accountLabel)}</p> : null}

      {connectedOn ? (
        <div className="flex items-center justify-between gap-2">
          <span className="text-[0.7rem] text-(--ui-text-tertiary)">{copy.connectedOn(connectedOn)}</span>
          {onDisconnect ? (
            <Button className="text-destructive hover:text-destructive" onClick={onDisconnect} size="xs" variant="text">
              {copy.disconnect}
            </Button>
          ) : null}
        </div>
      ) : null}

      <Separator />

      {onToggleForMe ? (
        <div className="grid gap-1">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-(--ui-text-primary)">{copy.onForMe}</span>
            <Switch
              aria-label={copy.onForMe}
              checked={card.state !== 'off' || card.offBy !== 'me'}
              disabled={card.offBy === 'org' || togglePending}
              onCheckedChange={onToggleForMe}
              size="xs"
            />
          </div>
          <p className="text-[0.7rem] text-(--ui-text-tertiary)">{copy.onForMeHint}</p>
        </div>
      ) : null}

      {orgDisabledCount > 0 ? (
        <div className="grid gap-1 rounded-md bg-(--ui-orange)/8 p-2.5">
          <p className="text-[0.7rem] text-(--ui-text-secondary)">{copy.orgNote(orgDisabledCount)}</p>
          {onOpenAdmin ? (
            <Button className="justify-self-start" onClick={onOpenAdmin} size="inline" variant="textStrong">
              {copy.orgLink}
            </Button>
          ) : null}
        </div>
      ) : null}

      <div className="mt-auto grid gap-2">
        <Separator />
        <p className="text-[0.7rem] text-(--ui-text-tertiary)">{copy.residencyHosted}</p>
        <p className="text-[0.7rem] text-(--ui-text-tertiary)">{t.connectorsPage.accountNote}</p>
      </div>
    </>
  )
}

function LocalColumn({ advanced, card, cost, profileName }: ColumnProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.dialog
  const overHttp = card.target?.startsWith('http') ?? false

  return (
    <>
      <div className="grid gap-1.5">
        <h3 className="text-xs font-medium text-(--ui-text-primary)">{copy.whereItLives}</h3>
        <p className="text-[0.7rem] leading-relaxed text-(--ui-text-secondary)">
          {overHttp ? copy.localUrlBody : copy.localProgramBody}
        </p>
        {card.target ? (
          <code className="break-all font-mono text-[0.65rem] text-(--ui-text-tertiary)">{card.target}</code>
        ) : null}
        {profileName ? (
          <p className="text-[0.7rem] text-(--ui-text-tertiary)">{copy.configuring(profileName)}</p>
        ) : null}
      </div>

      {cost && (cost.tokensPerCall || cost.usesPerMonth) ? (
        <>
          <Separator />
          <div className="grid gap-1.5">
            <h3 className="text-xs font-medium text-(--ui-text-primary)">{copy.whatItCosts}</h3>
            <div className="flex gap-6">
              <Metric label={copy.tokensPerCall} value={cost.tokensPerCall} />
              <Metric label={copy.usesPerMonth} value={cost.usesPerMonth} />
            </div>
          </div>
        </>
      ) : null}

      {advanced ? (
        <>
          <Separator />
          <details className="group grid gap-2">
            <summary className="flex cursor-pointer list-none items-center gap-1.5 text-xs font-medium text-(--ui-text-primary)">
              <Codicon
                className={cn('shrink-0 transition-transform duration-100 group-open:rotate-90')}
                name="chevron-right"
                size="0.75rem"
              />
              {copy.advanced}
              <span className="font-normal text-(--ui-text-quaternary)">{copy.advancedHint}</span>
            </summary>
            <div className="pt-2">{advanced}</div>
          </details>
        </>
      ) : null}
    </>
  )
}

function Metric({ label, value }: { label: string; value?: string }) {
  if (!value) {
    return null
  }

  return (
    <div className="grid gap-0.5">
      <span className="text-sm font-semibold tabular-nums text-(--ui-text-primary)">{value}</span>
      <span className="text-[0.65rem] text-(--ui-text-quaternary)">{label}</span>
    </div>
  )
}
