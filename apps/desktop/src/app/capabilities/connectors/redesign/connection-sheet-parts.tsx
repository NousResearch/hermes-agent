import { type FormEvent, type ReactNode, type RefObject, useCallback, useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ConnectorLogo } from '@/components/ui/connector-logo'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  dropdownMenuRow,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Tip } from '@/components/ui/tooltip'
import type { ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { isToolEnabled } from '@/lib/mcp-tool-filter'
import { formatDate } from '@/lib/time'
import { cn } from '@/lib/utils'
import { readableError } from '@/store/notifications'

import { okProbe, type Probe } from '../../mcp/mcp-status'
import type { McpServersController } from '../../mcp/use-mcp-servers'
import { type AccountOperation, syncAccountOperation } from '../data/account-operations'
import { aliasProblem, type AliasProblem } from '../data/alias'
import { accountName, type ConnectorPolicyView, connectorToolRows, memberRevision } from '../data/join'
import { useConnectorToolsSave, type WriteOutcome } from '../data/mutations'
import type { ConnectorToolsView } from '../data/queries'
import { localServerName, localWord } from '../derive'
import { facetSummary, toolDisplayName, toolRows } from '../derive-tools'
import type { InstallField } from '../local-server-control'
import type {
  AccountRow,
  ConnectorCardModel,
  ConnectorWayHosted,
  ConnectorWayLocal,
  ToolInput,
  ToolRowModel,
  ToolsEditorStatus
} from '../types'
import type { SaveResult } from '../use-tools-editor'

import { useConnectionsCopy } from './connections-copy'
import { type ToolFacet } from './filter-logic'
import { cardCtaKind, taxonomyAppForCard } from './taxonomy'

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

const NO_DISABLED_TOOLS: readonly string[] = []

const CARD = 'conn-card'

const CARD_TITLE = 'conn-sheet-section-title text-[length:var(--portal-type-section-size)] font-medium text-(--portal-text-primary)'
const CARD_BODY = 'mt-1 text-[length:var(--portal-type-ui-size)] text-(--portal-text-tertiary)'
const PRIMARY_BUTTON =
  'conn-primary w-full bg-(--portal-surface-brand) text-(--portal-text-on-brand) hover:bg-(--portal-surface-brand-hover)'

export interface SheetBannerState {
  kind: 'error' | 'success'
  retry?: () => void
  text: string
}

export interface LocalToolsModel {
  discovered: readonly string[]
  onSave: (disabled: string[]) => Promise<SaveResult>
  rows: readonly ToolRowModel[]
  savedDisabled: readonly string[]
  status: ToolsEditorStatus | null
}

function localToolsStatus(card: ConnectorCardModel, probe: Probe | undefined): ToolsEditorStatus | null {
  const local = card.ways.local

  if (local?.serverEnabled === false) {
    return 'off'
  }

  if (local?.reason?.key === 'serverNeedsAuth') {
    return 'needsAuth'
  }

  if (probe === undefined || probe === 'probing') {
    return 'loading'
  }

  return probe.ok ? null : 'unavailable'
}

/** Reads a local MCP server's discovered tools and edits its per-tool disable list. */
export function useLocalTools(
  controller: McpServersController,
  card: ConnectorCardModel,
  enabled: boolean
): LocalToolsModel {
  const name = localServerName(card)
  const probe = controller.probes[name]
  const entry = controller.servers[name]

  useEffect(() => {
    if (enabled && probe === undefined) {
      void controller.runProbe(name)
    }
  }, [controller, enabled, name, probe])

  const discovered = useMemo(() => okProbe(probe)?.tools.map(tool => tool.name) ?? [], [probe])
  const inputs = useMemo<ToolInput[]>(
    () =>
      (okProbe(probe)?.tools ?? []).map(tool => ({
        categories: [],
        deprecated: false,
        description: tool.description ?? '',
        facet: 'unclassified',
        hints: [],
        name: toolDisplayName(tool.name),
        slug: tool.name
      })),
    [probe]
  )
  const savedDisabled = useMemo(
    () => discovered.filter(tool => entry !== undefined && !isToolEnabled(entry, tool)),
    [discovered, entry]
  )
  const rows = useMemo(() => toolRows(inputs, new Set(savedDisabled)), [inputs, savedDisabled])
  const onSave = useCallback(
    async (disabled: string[]) => ((await controller.setServerTools(name, disabled, discovered)) ? 'saved' : 'failed'),
    [controller, discovered, name]
  )

  return { discovered, onSave, rows, savedDisabled, status: localToolsStatus(card, probe) }
}

/** Keeps Tab inside the sheet and closes it on Escape. Restores the opener's focus on unmount. */
export function useSheetFocusTrap(sheet: RefObject<HTMLDivElement | null>, onEscape: () => void): void {
  useEffect(() => {
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const root = sheet.current

    root?.querySelector<HTMLElement>(FOCUSABLE)?.focus()

    return () => {
      opener?.focus()
    }
  }, [sheet])

  useEffect(() => {
    const root = sheet.current

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || portalAboveSheet()) {
        return
      }

      if (event.key === 'Escape') {
        event.preventDefault()
        onEscape()

        return
      }

      if (event.key !== 'Tab' || !root) {
        return
      }

      const items = [...root.querySelectorAll<HTMLElement>(FOCUSABLE)].filter(item => item.closest('[inert]') === null)

      if (items.length === 0) {
        event.preventDefault()
        root.focus()

        return
      }

      const first = items[0]
      const last = items[items.length - 1]
      const active = document.activeElement

      if (event.shiftKey && (active === first || !root.contains(active))) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && (active === last || !root.contains(active))) {
        event.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', onKeyDown)

    return () => {
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [onEscape, sheet])
}

/** A Radix menu, popover or dialog rendered above the sheet owns Escape and Tab while it is open.
 *  Tooltips are excluded: their popper wrapper wraps a `<Tip>` bubble, which takes no focus. */
function portalAboveSheet(): boolean {
  return (
    document.querySelector(
      '[role="menu"], [role="dialog"]:not([data-slot="connection-sheet"]), [data-slot="popover-content"]'
    ) !== null
  )
}

export function SheetBanner({
  banner,
  busy = false,
  onDismiss
}: {
  banner: SheetBannerState
  busy?: boolean
  onDismiss?: () => void
}) {
  const copy = useConnectionsCopy()
  const error = banner.kind === 'error'

  return (
    <div
      className={cn(
        'conn-banner flex items-start gap-2 rounded-[var(--portal-radius-card)] px-3 py-2.5 text-[length:var(--portal-type-ui-size)]',
        error ? 'conn-banner-error' : 'conn-banner-success',
        error
          ? 'bg-(--portal-surface-danger) text-(--portal-text-danger)'
          : 'bg-[color-mix(in_srgb,var(--portal-status-online)_12%,transparent)] text-(--portal-status-online)'
      )}
      role={error ? 'alert' : 'status'}
    >
      <Codicon
        className={cn('mt-0.5 shrink-0', error ? 'text-(--portal-text-danger)' : 'text-(--portal-status-online)')}
        name={error ? 'warning' : 'check'}
        size="0.875rem"
      />
      <p className="min-w-0 flex-1">
        {banner.text}
        {error && banner.retry ? (
          <button className="conn-retry ms-1" disabled={busy} onClick={banner.retry} type="button">
            {copy.errors.tryAgain}
          </button>
        ) : null}
      </p>
      {onDismiss ? (
        <Button
          aria-label={copy.sheet.close}
          className="size-6 shrink-0 text-current"
          onClick={onDismiss}
          size="icon-xs"
          variant="ghost"
        >
          <Codicon name="close" size="0.875rem" />
        </Button>
      ) : null}
    </div>
  )
}

export interface SheetHeaderProps {
  card: ConnectorCardModel
  connected: boolean
  menu?: ReactNode
  onClose: () => void
}

export function SheetHeader({ card, connected, menu, onClose }: SheetHeaderProps) {
  const copy = useConnectionsCopy()

  return (
    <header className="conn-sheet-header shrink-0 px-6 pt-4 pb-5">
      <div className="flex items-center justify-between">
        <Button
          aria-label={copy.sheet.close}
          className="conn-sheet-close size-[1.875rem]"
          onClick={onClose}
          size="icon-xs"
          variant="ghost"
        >
          <Codicon name="close" size="1rem" />
        </Button>
        {menu}
      </div>

      <div className="mt-2 flex items-center gap-4">
        <span className="conn-sheet-logo">
          <ConnectorLogo className="conn-logo-mark" connector={{ name: card.name }} />
        </span>
        <div className="conn-sheet-heading">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <h2 className="conn-sheet-title [font-family:var(--portal-font-display)] leading-none font-light tracking-[var(--portal-tracking-display)] text-(--portal-text-primary)">
              {card.name}
            </h2>
            {connected ? (
              <span className="conn-pill-connected rounded-full border border-[color-mix(in_srgb,var(--portal-status-online)_45%,transparent)] bg-[color-mix(in_srgb,var(--portal-status-online)_12%,transparent)] px-2 py-0.5 text-[length:var(--portal-type-caption-size)] font-semibold tracking-[var(--portal-tracking-chip)] text-(--portal-status-online) uppercase">
                {copy.sheet.connected}
              </span>
            ) : null}
          </div>
          {card.description ? <p className="conn-sheet-subtitle">{card.description}</p> : null}
        </div>
      </div>
    </header>
  )
}

export interface HeaderMenuProps {
  disabled: boolean
  kind: 'hosted' | 'local'
  onDisconnect: () => void
  onReconnect: () => void
  onRefreshTools: () => void
  onRemoveLocal?: () => void
}

export function HeaderMenu({
  disabled,
  kind,
  onDisconnect,
  onReconnect,
  onRefreshTools,
  onRemoveLocal
}: HeaderMenuProps) {
  const copy = useConnectionsCopy()
  const hosted = kind === 'hosted'

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button aria-label={copy.card.menu} className="conn-sheet-menu" size="icon-xs" variant="ghost">
          <Codicon name="ellipsis" size="0.875rem" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {hosted ? (
          <DropdownMenuItem className={dropdownMenuRow} disabled={disabled} onSelect={onReconnect}>
            {copy.card.reconnect}
          </DropdownMenuItem>
        ) : null}
        <DropdownMenuItem className={dropdownMenuRow} onSelect={onRefreshTools}>
          {copy.card.refreshTools}
        </DropdownMenuItem>
        {hosted || onRemoveLocal ? (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              className={cn(dropdownMenuRow, 'text-(--portal-text-danger) focus:text-(--portal-text-danger)')}
              disabled={disabled}
              onSelect={() => (hosted ? onDisconnect() : onRemoveLocal?.())}
            >
              {hosted ? copy.card.disconnect : copy.card.remove}
            </DropdownMenuItem>
          </>
        ) : null}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export function AvailabilityCard({
  checked,
  disabled,
  onChange
}: {
  checked: boolean
  disabled: boolean
  onChange: (next: boolean) => void
}) {
  const copy = useConnectionsCopy()

  return (
    <section className={cn(CARD, 'conn-card-availability flex items-center gap-4')}>
      <div className="min-w-0 flex-1">
        <h3 className={CARD_TITLE}>{copy.sheet.availableToAgents}</h3>
        <p className={CARD_BODY}>{copy.sheet.availableSubtitle}</p>
      </div>
      <Switch
        aria-label={copy.sheet.availableToAgents}
        checked={checked}
        disabled={disabled}
        onCheckedChange={onChange}
      />
    </section>
  )
}

export interface AliasEditorProps {
  accounts: readonly AccountRow[]
  busy: boolean
  error?: null | string
  initial: string
  label: string
  onCancel: () => void
  onSubmit: (alias: string) => void
  self?: string
  submitLabel: string
}

export function AliasEditor({
  accounts,
  busy,
  error = null,
  initial,
  label,
  onCancel,
  onSubmit,
  self,
  submitLabel
}: AliasEditorProps) {
  const copy = useConnectionsCopy()
  const { t } = useI18n()
  const [draft, setDraft] = useState(initial)
  const [problem, setProblem] = useState<AliasProblem | null>(null)

  const submit = (event: FormEvent) => {
    event.preventDefault()
    const alias = draft.trim()

    if (alias.length === 0) {
      if (self) {
        setProblem('invalid')

        return
      }

      onSubmit('')

      return
    }

    const next = aliasProblem(alias, accounts, self)

    if (next) {
      setProblem(next)

      return
    }

    setProblem(null)
    onSubmit(alias)
  }

  const message =
    problem === null
      ? error
      : problem === 'taken'
        ? t.connectorsPage.accounts.aliasTaken
        : t.connectorsPage.accounts.aliasInvalid

  return (
    <form className="grid gap-2 py-2" onSubmit={submit}>
      <label className="grid gap-1">
        <span className="text-[length:var(--portal-type-meta-size)] text-(--portal-text-secondary)">{label}</span>
        <Input
          autoFocus
          className="h-8 text-xs"
          disabled={busy}
          onChange={event => {
            setDraft(event.currentTarget.value)
            setProblem(null)
          }}
          placeholder={copy.sheet.renamePlaceholder}
          value={draft}
        />
      </label>
      {message ? (
        <p className="text-[length:var(--portal-type-meta-size)] text-(--portal-text-danger)" role="alert">
          {message}
        </p>
      ) : null}
      <div className="conn-sheet-actions">
        <Button disabled={busy} loading={busy} size="xs" type="submit">
          {submitLabel}
        </Button>
        <Button disabled={busy} onClick={onCancel} size="xs" type="button" variant="text">
          {t.common.cancel}
        </Button>
      </div>
    </form>
  )
}

export interface AvailabilitySectionProps {
  agentControls: boolean
  hostedWay: ConnectorWayHosted | null
  local: ConnectorWayLocal | null
  onServerEnabledChange: (next: boolean) => void
  onToggleForMe?: (next: boolean) => void
  plugin?: string
  togglePending: boolean
}

export function AvailabilitySection({
  agentControls,
  hostedWay,
  local,
  onServerEnabledChange,
  onToggleForMe,
  plugin,
  togglePending
}: AvailabilitySectionProps) {
  if (hostedWay?.connected === true) {
    return (
      <AvailabilityCard
        checked={hostedWay.state !== 'off'}
        disabled={hostedWay.offBy === 'org' || !agentControls || togglePending || !onToggleForMe}
        onChange={next => onToggleForMe?.(next)}
      />
    )
  }

  // A plugin owns its server's enabled flag, so the sheet must never offer to pause it.
  if (local?.installed === true && plugin === undefined) {
    return (
      <AvailabilityCard
        checked={local.serverEnabled === true}
        disabled={!agentControls || togglePending}
        onChange={onServerEnabledChange}
      />
    )
  }

  return null
}

function AccountRowView({
  account,
  locale,
  onRemove,
  onRename
}: {
  account: AccountRow
  locale: string
  onRemove: () => void
  onRename: () => void
}) {
  const copy = useConnectionsCopy()
  const { t } = useI18n()

  return (
    <li className="conn-account-row flex items-center gap-3 border-t border-(--portal-stroke-divider) py-2 first:border-t-0">
      <div className="min-w-0 flex-1">
        <p className="truncate text-[length:var(--portal-type-body-size)] text-(--portal-text-primary)">
          {accountName(account)}
        </p>
        <p className="text-[length:var(--portal-type-meta-size)] text-(--portal-text-tertiary)">
          {copy.sheet.connectedOn(formatDate(new Date(account.created_at), locale))}
        </p>
      </div>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button aria-label={copy.card.menu} size="icon-xs" variant="ghost">
            <Codicon name="ellipsis" size="0.875rem" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem className={dropdownMenuRow} onSelect={onRename}>
            {t.connectorsPage.accounts.rename}
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            className={cn(dropdownMenuRow, 'text-(--portal-text-danger) focus:text-(--portal-text-danger)')}
            onSelect={onRemove}
          >
            {t.connectorsPage.accounts.remove}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </li>
  )
}

/** An inline account editor hands focus back to the sheet, never to `document.body`. */
function restoreSheetFocus(): void {
  requestAnimationFrame(() => {
    document.querySelector<HTMLElement>('[data-slot="connection-sheet"]')?.focus()
  })
}

function RemoveAccountCard({
  alias,
  appName,
  busy,
  onCancel,
  onConfirm
}: {
  alias: string
  appName: string
  busy: boolean
  onCancel: () => void
  onConfirm: () => void
}) {
  const copy = useConnectionsCopy()
  const { t } = useI18n()

  return (
    <section
      aria-label={copy.sheet.removeAccountTitle(alias)}
      className="conn-inline-danger"
      role="alertdialog"
    >
      <h3 className={CARD_TITLE}>{copy.sheet.removeAccountTitle(alias)}</h3>
      <p className={CARD_BODY}>{copy.sheet.removeAccountBody(appName)}</p>
      <div className="conn-sheet-actions mt-3">
        <Button
          className="bg-(--portal-surface-danger-solid) text-(--portal-text-on-danger) hover:opacity-90"
          disabled={busy}
          loading={busy}
          onClick={onConfirm}
          size="xs"
        >
          {t.connectorsPage.accounts.remove}
        </Button>
        <Button autoFocus disabled={busy} onClick={onCancel} size="xs" variant="outline">
          {t.common.cancel}
        </Button>
      </div>
    </section>
  )
}

export interface HostedAccountsCardProps {
  accounts: readonly AccountRow[]
  appName: string
  busy: boolean
  onAdd: (alias?: string) => Promise<WriteOutcome | void>
  onRemove: (account: AccountRow) => void
  onRename: (connectionId: string, alias: string) => Promise<WriteOutcome>
  onRenamed: (alias: string) => void
  renameAccountId: null | string
}

export function HostedAccountsCard({
  accounts,
  appName,
  busy,
  onAdd,
  onRemove,
  onRename,
  onRenamed,
  renameAccountId
}: HostedAccountsCardProps) {
  const copy = useConnectionsCopy()
  const { locale, t } = useI18n()
  const [removing, setRemoving] = useState<AccountRow | null>(null)
  const [edit, setEdit] = useState<{ connectionId: string; draft: string; error: null | string } | null>(null)
  const [adding, setAdding] = useState(false)
  const [addError, setAddError] = useState<null | string>(null)
  const [saving, setSaving] = useState(false)
  const [handledRename, setHandledRename] = useState<null | string>(null)

  useEffect(() => {
    if (!renameAccountId || handledRename === renameAccountId) {
      return
    }

    const account = accounts.find(row => row.connection_id === renameAccountId)

    if (!account) {
      return
    }

    setHandledRename(renameAccountId)
    setEdit({ connectionId: account.connection_id, draft: account.alias ?? '', error: null })
  }, [accounts, handledRename, renameAccountId])

  const rename = async (connectionId: string, alias: string) => {
    setSaving(true)

    try {
      const outcome = await onRename(connectionId, alias)

      if (outcome.ok) {
        onRenamed(alias)
        setEdit(null)

        return
      }

      setEdit(current =>
        current?.connectionId === connectionId
          ? {
              ...current,
              error: readableError(outcome.error, t.connectorsPage.page.writeFailed).message
            }
          : current
      )
    } finally {
      setSaving(false)
    }
  }

  const add = async (alias: string) => {
    const trimmed = alias.trim()
    setAddError(null)
    setSaving(true)

    try {
      const outcome = await onAdd(trimmed.length === 0 ? undefined : trimmed)

      if (outcome && !outcome.ok) {
        setAddError(readableError(outcome.error, t.connectorsPage.page.writeFailed).message)

        return
      }

      setAdding(false)
    } finally {
      setSaving(false)
    }
  }

  if (removing) {
    return (
      <RemoveAccountCard
        alias={accountName(removing)}
        appName={appName}
        busy={busy}
        onCancel={() => {
          setRemoving(null)
          restoreSheetFocus()
        }}
        onConfirm={() => {
          setRemoving(null)
          onRemove(removing)
        }}
      />
    )
  }

  const title = accounts.length === 1 ? copy.sheet.account : copy.sheet.accounts

  return (
    <section className={cn(CARD, 'conn-card-accounts')}>
      <h3 className={CARD_TITLE}>{title}</h3>
      <ul className="mt-1">
        {accounts.map(account =>
          edit?.connectionId === account.connection_id ? (
            <li className="border-t border-(--portal-stroke-divider) first:border-t-0" key={account.connection_id}>
              <AliasEditor
                accounts={accounts}
                busy={busy || saving}
                error={edit.error}
                initial={edit.draft}
                label={t.connectorsPage.accounts.aliasLabel}
                onCancel={() => {
                  setEdit(null)
                  restoreSheetFocus()
                }}
                onSubmit={alias => void rename(account.connection_id, alias)}
                self={account.connection_id}
                submitLabel={t.common.save}
              />
            </li>
          ) : (
            <AccountRowView
              account={account}
              key={account.connection_id}
              locale={locale}
              onRemove={() => {
                setEdit(null)
                setRemoving(account)
              }}
              onRename={() => setEdit({ connectionId: account.connection_id, draft: account.alias ?? '', error: null })}
            />
          )
        )}
      </ul>

      {adding ? (
        <AliasEditor
          accounts={accounts}
          busy={busy || saving}
          error={addError}
          initial=""
          label={t.connectorsPage.accounts.aliasLabel}
          onCancel={() => {
            setAdding(false)
            setAddError(null)
            restoreSheetFocus()
          }}
          onSubmit={alias => void add(alias)}
          submitLabel={copy.card.connect}
        />
      ) : (
        <Button
          className="conn-section-link mt-2"
          disabled={busy}
          onClick={() => {
            setAddError(null)
            setAdding(true)
          }}
          size="inline"
          variant="textStrong"
        >
          {accounts.length > 0 ? copy.sheet.addAnotherAccount : copy.sheet.addAnAccount}
        </Button>
      )}
    </section>
  )
}

export function SignInCard({
  appName,
  busy,
  cta,
  noSignIn,
  onConnect
}: {
  appName: string
  busy: boolean
  cta: string
  noSignIn: boolean
  onConnect: () => void
}) {
  const copy = useConnectionsCopy()

  return (
    <section className={cn(CARD, 'conn-card-signin grid gap-3')}>
      <div>
        <h3 className={CARD_TITLE}>{noSignIn ? `${copy.card.add} ${appName}` : copy.sheet.signInWith(appName)}</h3>
        {noSignIn ? null : <p className={CARD_BODY}>{copy.sheet.signInBody(appName)}</p>}
      </div>
      <Button className={PRIMARY_BUTTON} disabled={busy} loading={busy} onClick={onConnect}>
        {cta}
      </Button>
    </section>
  )
}

export function WaitingCard({
  appName,
  card,
  operation,
  onGiveUp,
  onReopen,
  onStopWaiting
}: {
  appName: string
  card: ConnectorCardModel
  operation: AccountOperation | null
  onGiveUp: (opId: string) => void
  onReopen: () => void
  onStopWaiting?: (card: ConnectorCardModel) => void
}) {
  const copy = useConnectionsCopy()
  const link = operation?.targets.find(target => target.connectUrl)?.connectUrl ?? null

  useEffect(() => {
    if (operation && link === null && !operation.settled) {
      void syncAccountOperation(operation.opId)
    }
  }, [link, operation?.opId, operation?.settled])

  const reopen = () => {
    if (link) {
      void window.hermesDesktop?.openExternal?.(link)

      return
    }

    onReopen()
  }

  const cancel = () => {
    if (operation) {
      onGiveUp(operation.opId)

      return
    }

    onStopWaiting?.(card)
  }

  return (
    <section className={cn(CARD, 'conn-card-waiting grid gap-3')}>
      <div>
        <h3 className={CARD_TITLE}>{copy.sheet.finishSignIn}</h3>
        <p className={CARD_BODY}>{copy.sheet.finishSignInBody(appName)}</p>
        <Button className="mt-1 h-auto px-0" onClick={reopen} size="inline" variant="textStrong">
          {copy.sheet.reopenSignIn}
        </Button>
      </div>
      <div className="conn-sheet-actions">
        <Button className={cn(PRIMARY_BUTTON, 'w-auto')} disabled>
          {copy.sheet.waitingFor(appName)}
        </Button>
        {operation !== null || onStopWaiting !== undefined ? (
          <Button onClick={cancel} size="inline" variant="text">
            {copy.sheet.cancelConnection}
          </Button>
        ) : null}
      </div>
    </section>
  )
}

function InstallForm({
  fields,
  installing,
  onInstall
}: {
  fields: readonly InstallField[]
  installing: boolean
  onInstall: (env: Record<string, string>) => void
}) {
  const copy = useConnectionsCopy()
  const { t } = useI18n()
  const [draft, setDraft] = useState<Record<string, string>>({})
  const missing = fields.some(field => field.required && !(draft[field.name] ?? '').trim())

  return (
    <div className="grid gap-2">
      {fields.length > 0 ? (
        <p className="text-[length:var(--portal-type-meta-size)] text-(--portal-text-tertiary)">
          {t.settings.mcp.catalogEnvRequired}
        </p>
      ) : null}
      {fields.map(field => (
        <label className="grid gap-1" key={field.name}>
          <span className="text-[length:var(--portal-type-meta-size)] text-(--portal-text-secondary)">
            {field.prompt || field.name}
            {field.required ? ' *' : ''}
          </span>
          <Input
            className="h-8 text-xs"
            onChange={event => setDraft({ ...draft, [field.name]: event.currentTarget.value })}
            type="password"
            value={draft[field.name] ?? ''}
          />
        </label>
      ))}
      <Button
        className={PRIMARY_BUTTON}
        disabled={installing || missing}
        loading={installing}
        onClick={() => onInstall(draft)}
      >
        {copy.card.add}
      </Button>
    </div>
  )
}

export function LocalServerCard({
  card,
  installing,
  installFields = [],
  local,
  onAuthenticate,
  onInstall
}: {
  card: ConnectorCardModel
  installing: boolean
  installFields?: readonly InstallField[]
  local: ConnectorWayLocal
  onAuthenticate: () => void
  onInstall: (env: Record<string, string>) => void
}) {
  const copy = useConnectionsCopy()
  const { t } = useI18n()

  if (local.installed !== true) {
    const oauth = local.authType === 'oauth'

    return (
      <section className={cn(CARD, 'conn-card-local grid gap-3')}>
        <h3 className={CARD_TITLE}>{oauth ? copy.sheet.signInWith(card.name) : `${copy.card.add} ${card.name}`}</h3>
        <InstallForm fields={installFields} installing={installing} onInstall={onInstall} />
      </section>
    )
  }

  return (
    <section className={cn(CARD, 'conn-card-local grid gap-3')}>
      <div>
        <h3 className={CARD_TITLE}>{t.connectorsPage.residencyLocal}</h3>
        <p className="mt-1 truncate font-mono text-[length:var(--portal-type-meta-size)] text-(--portal-text-tertiary)">
          {local.target ?? card.slug}
        </p>
        <p className="mt-1 text-[length:var(--portal-type-ui-size)] text-(--portal-text-secondary)">
          {t.connectorsPage.card.state[localWord(local)]}
        </p>
      </div>
      {local.verb === 'authenticate' ? (
        <Button className={PRIMARY_BUTTON} onClick={onAuthenticate}>
          {t.connectorsPage.card.verb.authenticate}
        </Button>
      ) : null}
    </section>
  )
}

export interface AccountSectionProps {
  accounts: readonly AccountRow[]
  appName: string
  busy: boolean
  card: ConnectorCardModel
  hostedConnected: boolean
  installFields?: readonly InstallField[]
  installing: boolean
  local: ConnectorWayLocal | null
  localInstalled: boolean
  onAdd: (alias?: string) => Promise<WriteOutcome | void>
  onAuthenticate: () => void
  onGiveUp: (opId: string) => void
  onInstall: (env: Record<string, string>) => void
  onRemove: (account: AccountRow) => void
  onRename: (connectionId: string, alias: string) => Promise<WriteOutcome>
  onRenamed: (alias: string) => void
  onStopWaiting?: (card: ConnectorCardModel) => void
  operation: AccountOperation | null
  renameAccountId: null | string
  waiting: boolean
}

export function AccountSection({
  accounts,
  appName,
  busy,
  card,
  hostedConnected,
  installFields,
  installing,
  local,
  localInstalled,
  onAdd,
  onAuthenticate,
  onGiveUp,
  onInstall,
  onRemove,
  onRename,
  onRenamed,
  onStopWaiting,
  operation,
  renameAccountId,
  waiting
}: AccountSectionProps) {
  const copy = useConnectionsCopy()
  const ctaKind = cardCtaKind(card)
  const noSignIn = taxonomyAppForCard(card)?.noSignIn === true || ctaKind === 'add'
  const localCard =
    local && (localInstalled || card.ways.hosted === null) ? (
      <LocalServerCard
        card={card}
        installFields={installFields}
        installing={installing}
        local={local}
        onAuthenticate={onAuthenticate}
        onInstall={onInstall}
      />
    ) : null

  if (hostedConnected) {
    return (
      <HostedAccountsCard
        accounts={accounts}
        appName={appName}
        busy={busy}
        onAdd={onAdd}
        onRemove={onRemove}
        onRename={onRename}
        onRenamed={onRenamed}
        renameAccountId={renameAccountId}
      />
    )
  }

  if (card.ways.hosted === null) {
    return localCard
  }

  return (
    <>
      {localCard}
      {waiting ? (
        <WaitingCard
          appName={appName}
          card={card}
          onGiveUp={onGiveUp}
          onReopen={() => void onAdd()}
          onStopWaiting={onStopWaiting}
          operation={operation}
        />
      ) : (
        <SignInCard
          appName={appName}
          busy={busy}
          cta={ctaKind === 'add' ? copy.card.add : copy.card.connect}
          noSignIn={noSignIn}
          onConnect={() => void onAdd()}
        />
      )}
    </>
  )
}

export interface ToolPermissionsCardProps {
  card: ConnectorCardModel
  connected: boolean
  disabledTools?: readonly string[]
  editable: boolean
  localTools?: LocalToolsModel
  onError: (message: string) => void
  onOpen: () => void
  policy: ConnectorPolicyView
  profile: ProfileScope
  tools: ConnectorToolsView
}

function isKnownFacet(facet: string): facet is ToolFacet {
  return facet === 'read' || facet === 'write' || facet === 'destructive'
}

export function ToolPermissionsCard({
  card,
  connected,
  disabledTools = NO_DISABLED_TOOLS,
  editable,
  localTools,
  onError,
  onOpen,
  policy,
  profile,
  tools
}: ToolPermissionsCardProps) {
  const copy = useConnectionsCopy()
  const { t } = useI18n()
  const saver = useConnectorToolsSave(profile, card.slug, memberRevision(policy))
  const [saving, setSaving] = useState<null | string>(null)

  const rows = useMemo(() => {
    if (localTools) {
      return localTools.rows
    }

    return editable
      ? connectorToolRows(policy, card.slug, tools.tools)
      : toolRows(tools.tools, new Set(disabledTools))
  }, [card.slug, disabledTools, editable, localTools, policy, tools.tools])

  const onBySlug = useMemo(() => new Map(rows.map(row => [row.slug, row.on])), [rows])

  const summary = useMemo(
    () => facetSummary(rows, slug => onBySlug.get(slug) === true),
    [onBySlug, rows]
  )

  const total = rows.filter(row => !row.deprecated).length
  // Local probes carry no facet metadata: show their one bucket so the card still counts them.
  const facets = localTools ? summary : summary.filter(row => isKnownFacet(row.facet))

  const toggle = async (facet: string, next: boolean) => {
    const targets = new Set(
      rows.filter(row => row.facet === facet && row.lockedBy === null && !row.deprecated).map(row => row.slug)
    )
    const disabled = new Set(
      rows.filter(row => row.lockedBy === null && !row.deprecated && !row.on).map(row => row.slug)
    )

    for (const slug of targets) {
      if (next) {
        disabled.delete(slug)
      } else {
        disabled.add(slug)
      }
    }

    setSaving(facet)
    const result = localTools ? await localTools.onSave([...disabled]) : await saver.onSave([...disabled])
    setSaving(null)

    if (result === 'conflict') {
      saver.reload()
      onError(t.connectorsPage.tools.conflictTitle)
    } else if (result === 'failed') {
      onError(t.connectorsPage.tools.saveFailed)
    }
  }

  return (
    <section className={cn(CARD, 'conn-card-tools')}>
      <div className="flex items-center gap-2">
        <h3 className={CARD_TITLE}>{copy.sheet.toolPermissions}</h3>
        <Tip label={copy.sheet.permissionsTooltip(card.name)}>
          <span
            aria-label={copy.sheet.permissionsTooltip(card.name)}
            className="inline-flex text-(--portal-text-tertiary)"
            role="img"
            tabIndex={0}
          >
            <Codicon name="question" size="0.875rem" />
          </span>
        </Tip>
        <Button className="conn-section-link ms-auto" onClick={onOpen} size="inline" variant="textStrong">
          {connected ? copy.sheet.reviewAll(total) : copy.sheet.previewAll(total)}
        </Button>
      </div>

      <ul className="mt-1">
        {facets.map(row => {
          const facet = row.facet
          const danger = facet === 'destructive'
          const known = isKnownFacet(facet)
          const label = known ? copy.badges[facet] : copy.tools.title
          const description = known ? copy.sheet.facetDescriptions[facet] : null

          return (
            <li
              className="flex items-center gap-3 border-t border-(--portal-stroke-divider) py-2 first:border-t-0"
              key={facet}
            >
              <div className="min-w-0 flex-1">
                <p
                  className={cn(
                    'text-[length:var(--portal-type-body-size)] font-medium',
                    danger ? 'text-(--portal-text-danger)' : 'text-(--portal-text-primary)'
                  )}
                >
                  {label}
                </p>
                <p className="text-[length:var(--portal-type-meta-size)] text-(--portal-text-tertiary)">
                  {copy.sheet.toolsCount(row.total)}
                  {description === null ? null : ` · ${description}`}
                </p>
              </div>
              {connected ? (
                <Switch
                  aria-label={label}
                  checked={row.switchState !== 'off'}
                  disabled={!editable || row.locked || saving !== null}
                  onCheckedChange={next => void toggle(facet, next)}
                  size="xs"
                />
              ) : (
                <span className="shrink-0 text-[length:var(--portal-type-meta-size)] text-(--portal-text-tertiary)">
                  {danger ? copy.sheet.startsOff : copy.sheet.startsOn}
                </span>
              )}
            </li>
          )
        })}
      </ul>

      {connected ? null : (
        <p className="mt-2 text-[length:var(--portal-type-meta-size)] text-(--portal-text-tertiary)">
          {copy.sheet.changeAfterConnect}
        </p>
      )}
    </section>
  )
}

export function DisconnectModal({
  appName,
  busy,
  onCancel,
  onConfirm
}: {
  appName: string
  busy: boolean
  onCancel: () => void
  onConfirm: () => void
}) {
  const copy = useConnectionsCopy()

  return (
    <div className="conn-sheet-modal-layer absolute inset-0 z-10 flex justify-end">
      <button
        aria-label={copy.sheet.close}
        className="conn-sheet-modal-scrim absolute inset-0"
        onClick={onCancel}
        type="button"
      />
      <div
        aria-label={copy.disconnect.title(appName)}
        aria-modal="true"
        className="conn-disconnect-modal relative flex h-full w-[25rem] max-w-full flex-col border-r-[3px] border-r-(--portal-surface-brand) bg-(--portal-surface-elevated) p-6 shadow-2xl"
        role="alertdialog"
      >
        <h2 className="[font-family:var(--portal-font-display)] text-[length:var(--portal-type-display-lg-size)] leading-tight font-light tracking-[var(--portal-tracking-display)] text-(--portal-text-primary)">
          {copy.disconnect.title(appName)}
        </h2>
        <p className="mt-2 text-[length:var(--portal-type-ui-size)] text-(--portal-text-tertiary)">
          {copy.disconnect.body(appName)}
        </p>
        <div className="conn-sheet-actions mt-auto pt-6">
          <Button
            className="bg-(--portal-surface-danger-solid) text-(--portal-text-on-danger) hover:opacity-90"
            disabled={busy}
            loading={busy}
            onClick={onConfirm}
          >
            {copy.disconnect.confirm}
          </Button>
          <Button autoFocus disabled={busy} onClick={onCancel} variant="outline">
            {copy.disconnect.cancel}
          </Button>
        </div>
      </div>
    </div>
  )
}
