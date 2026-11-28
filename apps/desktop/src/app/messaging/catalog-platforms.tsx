import type * as React from 'react'
import { useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { ErrorBanner } from '@/components/ui/error-state'
import { installMessagingCatalogPlatform, type MessagingCatalogPlatform, type ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { Download } from '@/lib/icons'
import { normalize } from '@/lib/text'
import { cn } from '@/lib/utils'
import { notify } from '@/store/notifications'

import { PlatformAvatar } from './platform-icon'

/** Selection id of a catalog row; installed platforms keep their bare id. */
export const catalogRowId = (row: MessagingCatalogPlatform) => `catalog:${row.name}`

/** The listed rows plus the one this page is installing (kept while the refreshed list has dropped it). */
export function catalogRowsWithKept(rows: MessagingCatalogPlatform[], kept: MessagingCatalogPlatform | null) {
  return kept ? [...rows, kept] : rows
}

/** Rows the list shows for a search and status filter. Not installed has no status yet: All only. */
export function visibleCatalogRows(rows: MessagingCatalogPlatform[], query: string, statusFilter: string) {
  if (statusFilter !== 'all') {
    return []
  }

  const needle = normalize(query)

  return rows.filter(row => !needle || normalize(`${row.title} ${row.name}`).includes(needle))
}

const CAPTION =
  'text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)'

function BadgePill({ badge }: { badge: MessagingCatalogPlatform['badge'] }) {
  const { t } = useI18n()

  return (
    <span className="inline-flex h-5 shrink-0 items-center rounded-full bg-primary/10 px-2 text-[0.68rem] font-medium text-primary">
      {t.messaging.catalog.badges[badge]}
    </span>
  )
}

export function CatalogPlatformRow({
  active,
  onSelect,
  row
}: {
  active: boolean
  onSelect: () => void
  row: MessagingCatalogPlatform
}) {
  return (
    <button
      className={cn(
        'row-hover flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left hover:text-foreground',
        active ? 'bg-(--ui-row-active-background) text-foreground' : 'text-(--ui-text-secondary)'
      )}
      onClick={onSelect}
      type="button"
    >
      <PlatformAvatar platformId={row.name} platformName={row.title} />
      <span className="flex min-w-0 flex-1 items-center justify-between gap-2">
        <span className="truncate text-[length:var(--conversation-text-font-size)] font-normal">{row.title}</span>
        <BadgePill badge={row.badge} />
      </span>
    </button>
  )
}

/** The "Available to install" section of the platform list; nothing when no row matches. */
export function CatalogPlatformList({
  activeName,
  onSelect,
  rows,
  title
}: {
  activeName: string | undefined
  onSelect: (row: MessagingCatalogPlatform) => void
  rows: MessagingCatalogPlatform[]
  title: React.ReactNode
}) {
  if (rows.length === 0) {
    return null
  }

  return (
    <section className="mt-3">
      {title}
      <ul className="mt-1 space-y-1">
        {rows.map(row => (
          <li key={row.name}>
            <CatalogPlatformRow active={activeName === row.name} onSelect={() => onSelect(row)} row={row} />
          </li>
        ))}
      </ul>
    </section>
  )
}

export interface CatalogInstallHandlers {
  /** The install request for `row` is about to go out (the page keeps the row while it runs). */
  onInstallStart: (row: MessagingCatalogPlatform) => void
  /** It settled, either way. */
  onInstallEnd: (row: MessagingCatalogPlatform) => void
  /** It succeeded under `profile`; `platformIds` are the ids the plugin registered. */
  onInstalled: (row: MessagingCatalogPlatform, platformIds: string[], profile: ProfileScope | undefined) => Promise<void>
}

/** A catalog platform's pane: what it is, then Install, or why it cannot install
 *  here and the command that does. The pane is reused across rows: an install still running for
 *  a row the user has left reports nothing here. */
export function CatalogPlatformDetail({
  onInstallEnd,
  onInstallStart,
  onInstalled,
  row,
  scopeProfile
}: CatalogInstallHandlers & {
  row: MessagingCatalogPlatform
  scopeProfile: ProfileScope | undefined
}) {
  const { t } = useI18n()
  const c = t.messaging.catalog
  const [installing, setInstalling] = useState(false)
  const [failure, setFailure] = useState<null | { command: string; error: string }>(null)
  const [prevName, setPrevName] = useState(row.name)
  const shownName = useRef(row.name)
  shownName.current = row.name

  if (prevName !== row.name) {
    setPrevName(row.name)
    setFailure(null)
    setInstalling(false)
  }

  async function install() {
    const target = row
    const profile = scopeProfile
    const stillShown = () => shownName.current === target.name
    setInstalling(true)
    setFailure(null)
    onInstallStart(target)

    try {
      const result = await installMessagingCatalogPlatform(target.name, profile)

      if (!result.ok) {
        if (stillShown()) {
          setFailure({
            command: result.install_command || target.install_command,
            error: result.error || c.installFailed(target.title)
          })
        }

        return
      }

      notify({ kind: 'success', title: c.installed(target.title), message: c.installedDetail })
      await onInstalled(target, result.platforms ?? [], profile)
    } catch (err) {
      if (stillShown()) {
        setFailure({ command: target.install_command, error: err instanceof Error ? err.message : String(err) })
      }
    } finally {
      onInstallEnd(target)

      if (stillShown()) {
        setInstalling(false)
      }
    }
  }

  const blockedReason = row.blocked === 'lazy_installs_off' ? c.lazyOff : row.blocked_reason

  return (
    <>
      <header className="flex items-start gap-3">
        <PlatformAvatar platformId={row.name} platformName={row.title} />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="min-w-0 truncate text-[0.9375rem] font-semibold tracking-tight">{row.title}</h3>
            <BadgePill badge={row.badge} />
            <span className={CAPTION}>{c.notInstalled}</span>
          </div>
          <p className={cn('mt-1', CAPTION)}>{row.summary}</p>
          {row.disclosure && (
            <p className={cn('mt-1', CAPTION)}>
              <span className="font-medium">{c.disclosure}</span> {row.disclosure}
            </p>
          )}
        </div>
      </header>

      {row.blocked ? (
        <section className={cn('grid gap-1', CAPTION)} data-testid="catalog-blocked">
          <p>{blockedReason}</p>
          <p>
            {c.runCommand} <code className="rounded bg-muted px-1 py-0.5 text-foreground">{row.install_command}</code>
          </p>
        </section>
      ) : (
        <div>
          <Button disabled={installing} onClick={() => void install()} size="sm">
            <Download />
            {installing ? c.installing : c.install}
          </Button>
          <p className={cn('mt-2', CAPTION)}>{c.intro(row.title)}</p>
        </div>
      )}

      {failure && (
        <ErrorBanner>
          {failure.error}
          {failure.command && (
            <span className="block">
              {c.runCommand} <code>{failure.command}</code>
            </span>
          )}
        </ErrorBanner>
      )}
    </>
  )
}
