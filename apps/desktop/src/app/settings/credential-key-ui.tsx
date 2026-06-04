import { type ChangeEvent, type KeyboardEvent } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ExternalLink, Loader2, Save } from '@/lib/icons'
import { cn } from '@/lib/utils'
import type { EnvVarInfo } from '@/types/hermes'

import { CONTROL_TEXT } from './constants'
import { prettyName, withoutKey } from './helpers'
import { ListRow } from './primitives'
import type { EnvRowProps } from './types'

export type KeyRowProps = Omit<EnvRowProps, 'info' | 'varKey'>

/** Matches Advanced / config field controls (ListRow + Input). */
export const CREDENTIAL_CONTROL_CLASS = cn('h-8', CONTROL_TEXT)

export const isKeyVar = (key: string, info: EnvVarInfo) =>
  info.is_password || /(?:_API_KEY|_TOKEN|_KEY)$/.test(key)

export const friendlyFieldLabel = (key: string, info: EnvVarInfo) =>
  info.description?.trim() ||
  key
    .replace(/_/g, ' ')
    .toLowerCase()
    .replace(/\b\w/g, c => c.toUpperCase())

export const credentialPlaceholder = (key: string, info: EnvVarInfo, label: string): string =>
  isKeyVar(key, info) ? `Paste ${label} key` : /URL$/i.test(key) ? 'https://…' : 'Optional'

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
          placeholder={placeholder ?? 'Paste key'}
          type={editType}
          value={draft}
        />
        {dirty && (
          <Button className="h-8 shrink-0" disabled={busy} onClick={() => void onSave(varKey)} size="sm">
            {busy ? <Loader2 className="size-4 animate-spin" /> : <Save />}
            {busy ? 'Saving' : 'Save'}
          </Button>
        )}
      </div>
      {editing && (
        <div className="flex items-center gap-1 text-[0.6875rem]">
          {info.is_set && (
            <>
              <Button
                className="h-auto px-0 py-0 text-[0.6875rem] text-destructive hover:text-destructive"
                disabled={busy}
                onClick={() => void onClear(varKey)}
                type="button"
                variant="text"
              >
                Remove
              </Button>
              <span className="text-muted-foreground">or</span>
            </>
          )}
          <span className="text-muted-foreground">esc to cancel</span>
        </div>
      )}
    </div>
  )
}

function CredentialDocsLink({ href }: { href: string }) {
  return (
    <a
      className="inline-flex w-fit items-center gap-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary) underline-offset-4 transition-colors hover:text-foreground hover:underline"
      href={href}
      onClick={e => e.stopPropagation()}
      rel="noreferrer"
      target="_blank"
    >
      Get a key
      <ExternalLink className="size-3" />
    </a>
  )
}

/** One credential row — same ListRow layout as Advanced config fields. */
export function CredentialKeyCard({
  info,
  label,
  placeholder,
  rowProps,
  varKey
}: {
  info: EnvVarInfo
  label: string
  placeholder: string
  rowProps: KeyRowProps
  varKey: string
}) {
  const docsUrl = info.url?.trim()
  const description = info.description?.trim()

  return (
    <ListRow
      action={<KeyField info={info} placeholder={placeholder} rowProps={rowProps} varKey={varKey} />}
      below={docsUrl ? <CredentialDocsLink href={docsUrl} /> : undefined}
      description={description}
      title={label}
    />
  )
}

/** Provider API key group — primary + optional advanced fields as ListRows. */
export function ProviderKeyRows({
  group,
  rowProps
}: {
  group: ProviderKeyRowGroup
  rowProps: KeyRowProps
}) {
  const docsUrl = group.docsUrl?.trim()
  const description = group.description?.trim()
  const docsBelow = docsUrl ? <CredentialDocsLink href={docsUrl} /> : undefined

  return (
    <>
      <ListRow
        action={
          <KeyField
            info={group.primary[1]}
            placeholder={`Paste ${group.name} key`}
            rowProps={rowProps}
            varKey={group.primary[0]}
          />
        }
        below={docsBelow}
        description={description}
        title={group.name}
      />
      {group.advanced.map(([key, info]) => {
        const fieldLabel = isKeyVar(key, info) ? prettyName(key.replace(/(?:_API_KEY|_TOKEN|_KEY)$/i, '')) : friendlyFieldLabel(key, info)

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
    </>
  )
}

export function credentialRowLabel(varKey: string, info: EnvVarInfo): string {
  if (isKeyVar(varKey, info)) {
    return prettyName(varKey.replace(/(?:_API_KEY|_TOKEN|_KEY)$/i, ''))
  }

  return prettyName(varKey)
}

export interface ProviderKeyRowGroup {
  advanced: [string, EnvVarInfo][]
  description?: string
  docsUrl?: string
  name: string
  primary: [string, EnvVarInfo]
}
