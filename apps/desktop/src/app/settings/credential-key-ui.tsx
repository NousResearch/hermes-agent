import { type ChangeEvent, type KeyboardEvent } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { translateNow, useI18n } from '@/i18n'
import { ChevronDown, ExternalLink, Loader2, Save } from '@/lib/icons'
import { cn } from '@/lib/utils'
import type { EnvVarInfo } from '@/types/hermes'

import { CONTROL_TEXT } from './constants'
import { prettyName, withoutKey } from './helpers'
import { ListRow } from './primitives'
import type { EnvRowProps } from './types'

export type KeyRowProps = Omit<EnvRowProps, 'info' | 'varKey'>

/** Matches Advanced / config field controls (ListRow + Input). */
export const CREDENTIAL_CONTROL_CLASS = cn('h-8', CONTROL_TEXT)

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
  info,
  placeholder,
  rowProps,
  varKey
}: {
  info: EnvVarInfo
  placeholder?: string
  rowProps: KeyRowProps
  varKey: string
}) {
  const { t } = useI18n()
  const { edits, onClear, onSave, saving, setEdits } = rowProps
  const editing = edits[varKey] !== undefined
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
        className={cn(CREDENTIAL_CONTROL_CLASS, 'cursor-pointer text-muted-foreground')}
        onFocus={startEdit}
        readOnly
        value={masked}
      />
    )
  }

  return (
    <div className="grid gap-1">
      <div className="flex items-center gap-2">
        <Input
          autoFocus={editing}
          className={cn(CREDENTIAL_CONTROL_CLASS, 'min-w-0 flex-1')}
          onChange={update}
          onKeyDown={keydown}
          placeholder={placeholder ?? t.settings.credentials.pasteKey}
          type={editType}
          value={draft}
        />
        {dirty && (
          <Button className="h-8 shrink-0" disabled={busy} onClick={() => void onSave(varKey)} size="sm">
            {busy ? <Loader2 className="animate-spin" /> : <Save />}
            {busy ? t.settings.credentials.saving : t.common.save}
          </Button>
        )}
      </div>
      {editing && (
        <div className="flex items-center gap-1 text-[0.6875rem]">
          {info.is_set && (
            <>
              <Button
                className="text-[0.6875rem] text-destructive hover:text-destructive"
                disabled={busy}
                onClick={() => void onClear(varKey)}
                size="inline"
                type="button"
                variant="text"
              >
                {t.settings.credentials.remove}
              </Button>
              <span className="text-muted-foreground">{t.settings.credentials.or}</span>
            </>
          )}
          <span className="text-muted-foreground">{t.settings.credentials.escToCancel}</span>
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
        'group/card rounded-[6px] px-2 py-1 transition-colors',
        expandable && 'cursor-pointer',
        expandable && !expanded && 'hover:bg-(--ui-row-hover-background)',
        expanded && 'bg-(--ui-bg-quaternary) ring-1 ring-(--ui-stroke-secondary)'
      )}
      onClick={expandable ? onToggle : undefined}
      onKeyDown={
        expandable
          ? e => {
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
      <div className="grid gap-3 py-2 sm:grid-cols-[minmax(0,1fr)_minmax(15rem,22rem)] sm:items-center">
        <div className="flex min-w-0 items-center gap-2">
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
          className="min-w-0 sm:justify-self-end"
          onClick={e => e.stopPropagation()}
          onFocus={() => {
            if (expandable && !expanded) {
              onExpand()
            }
          }}
        >
          <KeyField info={info} placeholder={placeholder} rowProps={rowProps} varKey={varKey} />
        </div>
      </div>

      {expandable && expanded && (
        <div className="grid gap-2.5 pb-2 pl-4" onClick={e => e.stopPropagation()}>
          {description && (
            <p className="text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
              {description}
            </p>
          )}

          {docsUrl && <CredentialDocsLink href={docsUrl} />}
        </div>
      )}
    </div>
  )
}

/** Provider API key group — collapsible card; description, docs link, and advanced fields expand on click. */
export function ProviderKeyRows({ expanded, group, onExpand, onToggle, rowProps }: ProviderKeyRowsProps) {
  const { t } = useI18n()
  const docsUrl = group.docsUrl?.trim()
  const description = group.description?.trim()
  const expandable = Boolean(description || docsUrl || group.advanced.length > 0)

  return (
    <div
      className={cn(
        'group/card rounded-[6px] px-2 py-1 transition-colors',
        expandable && 'cursor-pointer',
        expandable && !expanded && 'hover:bg-(--ui-row-hover-background)',
        expanded && 'bg-(--ui-bg-quaternary) ring-1 ring-(--ui-stroke-secondary)'
      )}
      onClick={expandable ? onToggle : undefined}
      onKeyDown={
        expandable
          ? e => {
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
      <div className="grid gap-3 py-2 sm:grid-cols-[minmax(0,1fr)_minmax(15rem,22rem)] sm:items-center">
        <div className="flex min-w-0 items-center gap-2">
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
          className="min-w-0 sm:justify-self-end"
          onClick={e => e.stopPropagation()}
          onFocus={() => {
            if (expandable && !expanded) {
              onExpand()
            }
          }}
        >
          <KeyField
            info={group.primary[1]}
            placeholder={t.settings.credentials.pasteLabelKey(group.name)}
            rowProps={rowProps}
            varKey={group.primary[0]}
          />
        </div>
      </div>

      {expandable && expanded && (
        <div className="grid gap-2.5 pb-2 pl-4" onClick={e => e.stopPropagation()}>
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
  expanded: boolean
  group: ProviderKeyRowGroup
  onExpand: () => void
  onToggle: () => void
  rowProps: KeyRowProps
}

export interface ProviderKeyRowGroup {
  advanced: [string, EnvVarInfo][]
  description?: string
  docsUrl?: string
  hasAnySet: boolean
  name: string
  primary: [string, EnvVarInfo]
}
