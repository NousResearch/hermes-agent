import { type ChangeEvent, type KeyboardEvent } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { translateNow, useI18n } from '@/i18n'
import { ChevronDown, ExternalLink, Loader2, Save, Trash2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import type { CredentialPoolEntry, EnvVarInfo } from '@/types/hermes'

import { CONTROL_TEXT } from './constants'
import { prettyName, withoutKey } from './helpers'
import { ListRow } from './primitives'
import type { EnvRowProps } from './types'

export type KeyRowProps = Omit<EnvRowProps, 'info' | 'varKey'>

// Redacted rotation-pool readout: label + active/exhausted status per stored
// credential, so a user juggling e.g. a personal + a shared Copilot account
// can see — and manually switch — which one requests are actually using,
// without touching the CLI.
export function CredentialPoolStatus({
  activating,
  entries,
  onActivate
}: {
  activating?: number
  entries: CredentialPoolEntry[]
  onActivate?: (index: number) => void
}) {
  const { t } = useI18n()
  const sorted = [...entries].sort((a, b) => a.priority - b.priority)
  const current = sorted[0]?.index

  return (
    <ul className="mb-1 ml-3 grid gap-0.5 border-l border-(--ui-border) pl-3">
      {sorted.map(entry => {
        const active = entry.last_status !== 'exhausted' && entry.last_status !== 'error'
        const isCurrent = entry.index === current
        const busy = activating === entry.index

        return (
          <li className="flex items-center gap-2 py-0.5 text-xs text-muted-foreground" key={entry.index}>
            <span
              className={cn('inline-block size-1.5 rounded-full', active ? 'bg-primary' : 'bg-muted-foreground/40')}
            />
            <span className="truncate">{entry.label || entry.token_preview}</span>
            <span className="text-muted-foreground/60">{entry.last_status ?? (active ? 'active' : 'idle')}</span>
            {onActivate && !isCurrent && (
              <Button
                className="ml-auto h-6 px-1.5 text-[11px]"
                disabled={busy}
                onClick={e => {
                  e.stopPropagation()
                  onActivate(entry.index)
                }}
                size="sm"
                type="button"
                variant="ghost"
              >
                {busy ? <Loader2 className="size-3 animate-spin" /> : t.settings.credentials.useThisAccount}
              </Button>
            )}
          </li>
        )
      })}
    </ul>
  )
}


/** Matches Advanced / config field controls (ListRow + Input). */
export const CREDENTIAL_CONTROL_CLASS = cn('h-8', CONTROL_TEXT)

// Resting credential field: chrome stripped so it reads as plain subtext.
// Stacked (<@2xl) it collapses to zero box (flush under its label); at @2xl it
// keeps the full control metrics (h-8 + px-2.5/py-1.5) so it centres on the
// label and nothing shifts when focus/expand adds the border. `!` beats the
// unlayered chrome CSS and the shared control sizing.
const CRED_BARE = 'border-0! bg-transparent! shadow-none! h-auto! p-0! @2xl:h-8! @2xl:px-2.5! @2xl:py-1.5!'

export const isKeyVar = (key: string, info: EnvVarInfo) => info.is_password || /(?:_API_KEY|_TOKEN|_KEY)$/.test(key)

export const friendlyFieldLabel = (key: string, info: EnvVarInfo) =>
  info.description?.trim() ||
  key
    .replace(/_/g, ' ')
    .toLowerCase()
    .replace(/\b\w/g, c => c.toUpperCase())

export const credentialPlaceholder = (key: string, info: EnvVarInfo, label: string): string =>
  isKeyVar(key, info)
    ? translateNow('settings.credentials.pasteLabelKey', label)
    : /URL$/i.test(key)
      ? 'https://…'
      : translateNow('settings.credentials.optional')

// A single credential field: a set key shows as a filled read-only input
// (redacted value) that edits in place on click. Save appears once typed; a set
// key also offers Remove, and Esc cancels without closing the overlay.
export function KeyField({
  expanded = false,
  info,
  placeholder,
  rowProps,
  varKey
}: {
  expanded?: boolean
  info: EnvVarInfo
  placeholder?: string
  rowProps: KeyRowProps
  varKey: string
}) {
  const { t } = useI18n()
  const { edits, onClear, onSave, saving, setEdits } = rowProps
  const editing = edits[varKey] !== undefined
  // Bare (plain subtext) only while the group is collapsed and idle. Expanding
  // the card counts as "focused in", so it gets full input chrome too.
  const bare = !editing && !expanded
  const draft = edits[varKey] ?? ''
  const dirty = draft.trim().length > 0
  const busy = saving === varKey
  const masked = info.redacted_value ?? '••••••••'
  const startEdit = () => setEdits(c => ({ ...c, [varKey]: '' }))
  const cancel = () => setEdits(c => withoutKey(c, varKey))
  const update = (e: ChangeEvent<HTMLInputElement>) => setEdits(c => ({ ...c, [varKey]: e.target.value }))

  const keydown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && dirty) {
      void onSave(varKey)
    } else if (e.key === 'Escape' && editing) {
      e.preventDefault()
      e.stopPropagation()
      cancel()
    }
  }

  const editType = info.is_password ? 'password' : 'text'

  if (info.is_set && !editing) {
    return (
      <Input
        className={cn(CREDENTIAL_CONTROL_CLASS, bare && CRED_BARE, 'cursor-pointer text-muted-foreground')}
        onFocus={startEdit}
        readOnly
        value={masked}
      />
    )
  }

  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2">
      <Input
        autoFocus={editing}
        className={cn(CREDENTIAL_CONTROL_CLASS, bare && CRED_BARE)}
        onChange={update}
        onFocus={() => {
          if (!editing) {
            startEdit()
          }
        }}
        onKeyDown={keydown}
        placeholder={placeholder ?? t.settings.credentials.pasteKey}
        type={editType}
        value={draft}
      />
      {/* Inline trailing controls — mirrors SearchField's inline clear button.
          No floating hint row that reflows the grid or overlaps the card body;
          Esc still cancels via keydown. */}
      {editing && (info.is_set || dirty) && (
        <div className="flex items-center gap-1">
          {info.is_set && (
            <Button
              aria-label={t.settings.credentials.remove}
              className="text-muted-foreground hover:text-destructive"
              disabled={busy}
              onClick={() => void onClear(varKey)}
              size="icon-xs"
              title={t.settings.credentials.remove}
              type="button"
              variant="ghost"
            >
              <Trash2 />
            </Button>
          )}
          {dirty && (
            <Button className="h-8" disabled={busy} onClick={() => void onSave(varKey)} size="sm">
              {busy ? <Loader2 className="animate-spin" /> : <Save />}
              {busy ? t.settings.credentials.saving : t.common.save}
            </Button>
          )}
        </div>
      )}
    </div>
  )
}

function CredentialDocsLink({ href }: { href: string }) {
  const { t } = useI18n()

  return (
    <a
      className="inline-flex w-fit items-center gap-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary) underline-offset-4 transition-colors hover:text-foreground hover:underline"
      href={href}
      onClick={e => e.stopPropagation()}
      rel="noreferrer"
      target="_blank"
    >
      {t.settings.credentials.getKey}
      <ExternalLink className="size-3" />
    </a>
  )
}

/** One credential row — collapsible; description and docs link expand on click. */
export function CredentialKeyCard({
  expanded,
  info,
  label,
  onExpand,
  onToggle,
  placeholder,
  rowProps,
  varKey
}: CredentialKeyCardProps) {
  const docsUrl = info.url?.trim()
  const description = info.description?.trim()
  const expandable = Boolean(description || docsUrl)

  return (
    <div
      className={cn(
        '@container group/card rounded-[6px] p-3 transition-colors',
        expandable && 'cursor-pointer',
        expandable && !expanded && 'row-hover',
        expanded && 'bg-(--ui-bg-quaternary) ring-1 ring-(--ui-stroke-secondary)'
      )}
      onClick={expandable ? onToggle : undefined}
      onKeyDown={
        expandable
          ? e => {
              // Only the card's own focus toggles it — ignore Enter/Space
              // bubbling up from the inputs/buttons inside (Enter saves a key,
              // Space types a space) so keyboard editing never collapses the card.
              if (e.target !== e.currentTarget) {
                return
              }

              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                onToggle()
              }
            }
          : undefined
      }
      role={expandable ? 'button' : undefined}
      tabIndex={expandable ? 0 : undefined}
    >
      {/* One CSS grid: 1 col stacked, 2 cols at @2xl. p-3 card padding = gap-3
          row/col gaps, everything top-left aligned (items-start), no indents.
          The label row is h-8 to line up with the input row beside it. */}
      <div className="grid grid-cols-1 items-start gap-x-3 gap-y-1.5 @2xl:grid-cols-[minmax(0,1fr)_minmax(15rem,22rem)] @2xl:gap-y-3">
        <div className="flex h-8 min-w-0 items-center gap-2">
          <span
            className={cn('size-2 shrink-0 rounded-full', info.is_set ? 'bg-primary' : 'bg-(--ui-stroke-secondary)')}
          />

          <span className="min-w-0 truncate text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
            {label}
          </span>

          {expandable && (
            <ChevronDown
              className={cn(
                'size-3.5 shrink-0 text-muted-foreground transition',
                expanded ? 'rotate-180 opacity-100' : 'opacity-0 group-hover/card:opacity-100'
              )}
            />
          )}
        </div>

        <div
          className="min-w-0"
          onClick={e => e.stopPropagation()}
          onFocus={() => {
            if (expandable && !expanded) {
              onExpand()
            }
          }}
        >
          <KeyField expanded={expanded} info={info} placeholder={placeholder} rowProps={rowProps} varKey={varKey} />
        </div>

        {expandable && expanded && (
          <div className="grid gap-3 @2xl:col-span-2" onClick={e => e.stopPropagation()}>
            {description && (
              <p className="text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                {description}
              </p>
            )}

            {docsUrl && <CredentialDocsLink href={docsUrl} />}
          </div>
        )}
      </div>
    </div>
  )
}

/** Provider API key group — collapsible card; description, docs link, and advanced fields expand on click. */
export function ProviderKeyRows({
  activatingIndex,
  expanded,
  group,
  onActivate,
  onExpand,
  onToggle,
  poolEntries,
  rowProps
}: ProviderKeyRowsProps) {
  const { t } = useI18n()
  const docsUrl = group.docsUrl?.trim()
  const description = group.description?.trim()
  const expandable = Boolean(description || docsUrl || group.advanced.length > 0)

  return (
    <div
      className={cn(
        '@container group/card rounded-[6px] p-3 transition-colors',
        expandable && 'cursor-pointer',
        expandable && !expanded && 'row-hover',
        expanded && 'bg-(--ui-bg-quaternary) ring-1 ring-(--ui-stroke-secondary)'
      )}
      onClick={expandable ? onToggle : undefined}
      onKeyDown={
        expandable
          ? e => {
              // Only the card's own focus toggles it — ignore Enter/Space
              // bubbling up from the inputs/buttons inside (Enter saves a key,
              // Space types a space) so keyboard editing never collapses the card.
              if (e.target !== e.currentTarget) {
                return
              }

              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                onToggle()
              }
            }
          : undefined
      }
      role={expandable ? 'button' : undefined}
      tabIndex={expandable ? 0 : undefined}
    >
      {/* Same grid as CredentialKeyCard: 1 col stacked, 2 cols at @2xl, p-3 =
          gap-3, items-start, label row h-8 to line up with the input row. */}
      <div className="grid grid-cols-1 items-start gap-x-3 gap-y-1.5 @2xl:grid-cols-[minmax(0,1fr)_minmax(15rem,22rem)] @2xl:gap-y-3">
        <div className="flex h-8 min-w-0 items-center gap-2">
          <span
            className={cn(
              'size-2 shrink-0 rounded-full',
              group.hasAnySet ? 'bg-primary' : 'bg-(--ui-stroke-secondary)'
            )}
          />

          <span className="min-w-0 truncate text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
            {group.name}
          </span>

          {expandable && (
            <ChevronDown
              className={cn(
                'size-3.5 shrink-0 text-muted-foreground transition',
                expanded ? 'rotate-180 opacity-100' : 'opacity-0 group-hover/card:opacity-100'
              )}
            />
          )}
        </div>

        <div
          className="min-w-0"
          onClick={e => e.stopPropagation()}
          onFocus={() => {
            if (expandable && !expanded) {
              onExpand()
            }
          }}
        >
          <KeyField
            expanded={expanded}
            info={group.primary[1]}
            placeholder={t.settings.credentials.pasteLabelKey(group.name)}
            rowProps={rowProps}
            varKey={group.primary[0]}
          />
        </div>

        {/* Only shown when the provider has >1 pooled credential (e.g. a
            personal + a shared Copilot account) — a single credential needs
            no "which one is active" affordance. */}
        {poolEntries && poolEntries.length > 1 && (
          <div className="@2xl:col-span-2">
            <CredentialPoolStatus activating={activatingIndex} entries={poolEntries} onActivate={onActivate} />
          </div>
        )}

        {expandable && expanded && (
          <div className="grid gap-3 @2xl:col-span-2" onClick={e => e.stopPropagation()}>
            {description && (
              <p className="text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                {description}
              </p>
            )}

            {group.advanced.map(([key, info]) => {
              const fieldLabel = isKeyVar(key, info)
                ? prettyName(key.replace(/(?:_API_KEY|_TOKEN|_KEY)$/i, ''))
                : friendlyFieldLabel(key, info)

              return (
                <ListRow
                  action={
                    <KeyField
                      expanded={expanded}
                      info={info}
                      placeholder={credentialPlaceholder(key, info, fieldLabel)}
                      rowProps={rowProps}
                      varKey={key}
                    />
                  }
                  key={key}
                  title={fieldLabel}
                />
              )
            })}

            {docsUrl && <CredentialDocsLink href={docsUrl} />}
          </div>
        )}
      </div>
    </div>
  )
}

export function credentialRowLabel(varKey: string, info: EnvVarInfo): string {
  if (isKeyVar(varKey, info)) {
    return prettyName(varKey.replace(/(?:_API_KEY|_TOKEN|_KEY)$/i, ''))
  }

  return prettyName(varKey)
}

interface CredentialKeyCardProps {
  expanded: boolean
  info: EnvVarInfo
  label: string
  onExpand: () => void
  onToggle: () => void
  placeholder: string
  rowProps: KeyRowProps
  varKey: string
}

interface ProviderKeyRowsProps {
  activatingIndex?: number
  expanded: boolean
  group: ProviderKeyRowGroup
  onActivate?: (index: number) => void
  onExpand: () => void
  onToggle: () => void
  poolEntries?: CredentialPoolEntry[]
  rowProps: KeyRowProps
}

export interface ProviderKeyRowGroup {
  advanced: [string, EnvVarInfo][]
  description?: string
  docsUrl?: string
  hasAnySet: boolean
  name: string
  poolProvider?: string
  primary: [string, EnvVarInfo]
}
