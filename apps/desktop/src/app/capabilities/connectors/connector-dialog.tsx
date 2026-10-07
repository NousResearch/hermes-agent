import { type ReactNode, type RefObject, useRef } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ConnectorLogo } from '@/components/ui/connector-logo'
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import { X } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { CatalogMark } from './catalog-mark'
import { connectorKindWord, showsCatalogMark } from './connector-kind'
import { type InstallField, LocalInstall } from './local-server-control'
import type { ConnectorCardModel, ConnectorState, ConnectorVerb, ConnectorWayLocal } from './types'

type BadgeVariant = 'default' | 'destructive' | 'muted' | 'success' | 'warn'

const STATE_BADGE = {
  available: 'muted',
  broken: 'destructive',
  connected: 'success',
  connecting: 'warn',
  expired: 'warn',
  off: 'muted',
  unknown: 'muted'
} satisfies Record<ConnectorState, BadgeVariant>

export interface ConnectorDialogProps {
  advanced?: ReactNode
  card: ConnectorCardModel
  cost?: { tokensPerCall?: string; usesPerMonth?: string }
  installFields?: readonly InstallField[]
  installing?: boolean
  menu?: ReactNode
  onAuthenticate?: () => void
  onInstall?: (env: Record<string, string>) => void
  onOpenChange: (open: boolean) => void
  onServerToggle?: (next: boolean) => void
  open: boolean
  tools: ReactNode
}

const localTarget = (card: ConnectorCardModel): string | undefined => card.ways.target

export function ConnectorDialog({ card, onOpenChange, open, tools, ...rest }: ConnectorDialogProps) {
  const { t } = useI18n()
  const titleRef = useRef<HTMLHeadingElement>(null)

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent
        bodyClassName="gap-0 overflow-hidden p-0"
        className="max-h-[85vh] max-w-lg"
        onOpenAutoFocus={event => {
          event.preventDefault()
          titleRef.current?.focus()
        }}
        showCloseButton={false}
      >
        <Header card={card} titleRef={titleRef} {...rest} />

        <div className="flex min-h-0 flex-1 flex-col">
          <LocalLead card={card} {...rest} />

          {tools}

          <LocalFoot card={card} {...rest} />
        </div>

        <span className="sr-only">{t.connectorsPage.title}</span>
      </DialogContent>
    </Dialog>
  )
}

type PartProps = Omit<ConnectorDialogProps, 'onOpenChange' | 'open' | 'tools'>

function Header({ card, titleRef, ...rest }: PartProps & { titleRef: RefObject<HTMLHeadingElement | null> }) {
  const { t } = useI18n()
  const copy = t.connectorsPage.card

  return (
    <header className="flex shrink-0 items-center gap-2.5 border-b border-(--ui-stroke-tertiary) px-5 py-3">
      <ConnectorLogo
        className="size-9 shrink-0 rounded-[9px]"
        connector={{ name: card.slug, title: card.name }}
      />

      <div className="grid min-w-0 flex-1 gap-0.5">
        <div className="flex min-w-0 items-center gap-2">
          <DialogTitle className="truncate text-base font-semibold outline-none" ref={titleRef} tabIndex={-1}>
            {card.name}
          </DialogTitle>

          <span className="shrink-0 text-[0.6875rem] text-(--ui-text-tertiary)">{connectorKindWord(card, copy)}</span>

          {showsCatalogMark(card) ? <CatalogMark /> : null}

          <Badge className="shrink-0" size="xs" variant={STATE_BADGE[card.state]}>
            {copy.state[card.stateWord]}
          </Badge>
        </div>

        <DialogDescription className="truncate text-[0.72rem] text-(--ui-text-secondary)">
          {card.description ?? localTarget(card) ?? copy.state[card.stateWord]}
        </DialogDescription>
      </div>

      <HeaderActions card={card} {...rest} />
    </header>
  )
}

function HeaderActions({
  card,
  installFields,
  installing,
  menu,
  onAuthenticate,
  onInstall,
  onServerToggle
}: PartProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.card
  const localWay = card.ways
  const action = localAction({ installFields, installing, onAuthenticate, onInstall, way: localWay })

  return (
    <div className="flex shrink-0 items-center gap-2">
      {action ? (
        <Button disabled={action.busy} onClick={action.run} size="xs">
          {copy.verb[action.verb]}
        </Button>
      ) : null}

      {localWay.installed === true && card.plugin === undefined && onServerToggle ? (
        <Switch
          aria-label={localWay.serverEnabled === true ? copy.turnServerOff(card.name) : copy.turnServerOn(card.name)}
          checked={localWay.serverEnabled ?? false}
          onCheckedChange={onServerToggle}
          size="xs"
        />
      ) : null}

      {menu}

      <DialogClose asChild>
        <Button aria-label={t.common.close} size="icon-xs" variant="ghost">
          <X className="size-3" />
        </Button>
      </DialogClose>
    </div>
  )
}

interface LeadVerb {
  busy?: boolean
  run: () => void
  verb: ConnectorVerb
}

function localAction({
  installFields,
  installing = false,
  onAuthenticate,
  onInstall,
  way
}: {
  installFields?: readonly InstallField[]
  installing?: boolean
  onAuthenticate?: () => void
  onInstall?: (env: Record<string, string>) => void
  way: ConnectorWayLocal
}): LeadVerb | undefined {
  if (way.installed === false) {
    return onInstall && (installFields ?? []).length === 0
      ? { busy: installing, run: () => onInstall({}), verb: 'install' }
      : undefined
  }

  return way.verb === 'authenticate' && onAuthenticate ? { run: onAuthenticate, verb: 'authenticate' } : undefined
}


function LocalLead({ card, installFields = [], installing, onInstall }: PartProps) {
  const local = card.ways
  const install = local.installed === false && installFields.length > 0 ? onInstall : undefined

  if (!install) {
    return null
  }

  return (
    <div className="shrink-0 border-b border-(--ui-stroke-tertiary) px-3.5 py-2.5">
      <LocalInstall installFields={installFields} installing={installing} onInstall={install} />
    </div>
  )
}

function LocalFoot({ advanced, cost }: PartProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.dialog
  const metrics = cost && (cost.tokensPerCall || cost.usesPerMonth)

  if (!metrics && !advanced) {
    return null
  }

  return (
    <div className="grid shrink-0 gap-3 border-t border-(--ui-stroke-tertiary) px-3.5 py-2.5">
      {metrics ? (
        <p className="text-[0.7rem] text-(--ui-text-tertiary)">
          {[
            cost?.usesPerMonth ? `${cost.usesPerMonth} ${copy.usesPerMonth}` : null,
            cost?.tokensPerCall ? `${cost.tokensPerCall} ${copy.tokensPerCall}` : null
          ]
            .filter(Boolean)
            .join(' · ')}
        </p>
      ) : null}

      {advanced ? (
        <details className="group grid gap-2">
          <summary className="flex cursor-pointer list-none items-center gap-1.5 text-xs font-medium text-(--ui-text-primary)">
            <Codicon
              className={cn('shrink-0 transition-transform duration-100 group-open:rotate-90')}
              name="chevron-right"
              size="0.75rem"
            />
            <span className="shrink-0">{copy.advanced}</span>
            <span className="min-w-0 truncate font-normal text-(--ui-text-quaternary)">{copy.advancedHint}</span>
          </summary>
          <div className="pt-2">{advanced}</div>
        </details>
      ) : null}
    </div>
  )
}
