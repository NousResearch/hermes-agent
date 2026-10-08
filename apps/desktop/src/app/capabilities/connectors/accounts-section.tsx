import { type Dispatch, type ReactNode, type SetStateAction, useState } from 'react'

import { ListRow } from '@/app/settings/primitives'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n/types'
import { readableError } from '@/store/notifications'

import { aliasProblem } from './data/alias'
import { accountName } from './data/join'
import type { WriteOutcome } from './data/mutations'
import type { AccountRow } from './types'

type AccountsCopy = Translations['connectorsPage']['accounts']
type BadgeVariant = 'destructive' | 'muted' | 'warn'

/** Statuses a reconnect repairs; `pending` waits for a sign-in and `active` needs nothing. */
const STATUS_VIEW = {
  expired: { reconnect: true, variant: 'warn' },
  failed: { reconnect: true, variant: 'warn' },
  inactive: { reconnect: true, variant: 'muted' },
  pending: { reconnect: false, variant: 'muted' },
  revoked: { reconnect: true, variant: 'destructive' }
} satisfies Record<Exclude<AccountRow['status'], 'active'>, { reconnect: boolean; variant: BadgeVariant }>

/** The name being typed: for one account, or for an account still to add. A refused write reopens it with the error. */
export type AccountEdit =
  | { connectionId: string; draft?: string; error?: string; kind: 'rename' }
  | {
      draft?: string
      error?: string
      kind: 'add'
    }

export interface AccountsSectionProps {
  accounts: readonly AccountRow[]
  /** A sign-in for this app is still open; another add or reconnect would act on the wrong account. */
  connecting: boolean
  edit: AccountEdit | null
  onAdd: (alias: string) => Promise<WriteOutcome>
  onEditChange: Dispatch<SetStateAction<AccountEdit | null>>
  onReconnect: (account: AccountRow) => void
  onRemove: (account: AccountRow) => void
  onRename: (connectionId: string, alias: string) => Promise<WriteOutcome>
  reconnecting: boolean
  retired: readonly AccountRow[]
}

export function AccountsSection({
  accounts,
  connecting,
  edit,
  onAdd,
  onEditChange,
  onReconnect,
  onRemove,
  onRename,
  reconnecting,
  retired
}: AccountsSectionProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.accounts
  const everyAccount = [...accounts, ...retired]
  const [adding, setAdding] = useState(false)

  if (accounts.length === 0 && retired.length === 0) {
    return null
  }

  const refusal = (outcome: WriteOutcome): string | undefined =>
    outcome.ok
      ? undefined
      : outcome.error.reason === 'ALIAS_TAKEN'
        ? copy.aliasTaken
        : readableError(outcome.error, t.connectorsPage.page.writeFailed).message

  // The row shows the new name as soon as the editor closes; a refusal restores the old name and reopens the
  // editor, unless the user has started another edit since.
  const rename = async (connectionId: string, alias: string) => {
    onEditChange(null)

    const error = refusal(await onRename(connectionId, alias))

    if (error) {
      onEditChange(current => current ?? { connectionId, draft: alias, error, kind: 'rename' })
    }
  }

  const add = async (alias: string) => {
    setAdding(true)

    const error = refusal(await onAdd(alias))

    setAdding(false)
    onEditChange(current => (current?.kind === 'add' ? (error ? { draft: alias, error, kind: 'add' } : null) : current))
  }

  return (
    <section
      aria-label={copy.heading}
      className="grid max-h-[40vh] shrink-0 overflow-y-auto overscroll-y-contain border-b border-(--ui-stroke-tertiary) px-3.5 py-1"
      data-slot="connector-accounts"
    >
      {accounts.map(account =>
        edit?.kind === 'rename' && edit.connectionId === account.connection_id ? (
          <AliasEditor
            accounts={everyAccount}
            error={edit.error}
            initial={edit.draft ?? account.alias ?? ''}
            key={account.connection_id}
            label={copy.renameLabel(accountName(account))}
            onCancel={() => onEditChange(null)}
            onSubmit={alias => void rename(account.connection_id, alias)}
            self={account.connection_id}
            submitLabel={t.common.save}
            unchanged={account.alias ?? ''}
          />
        ) : (
          <AccountLine
            account={account}
            actions={
              <>
                {account.status !== 'active' && STATUS_VIEW[account.status].reconnect ? (
                  <Button
                    disabled={reconnecting || connecting}
                    onClick={() => onReconnect(account)}
                    size="xs"
                    variant="secondary"
                  >
                    {t.connectorsPage.card.verb.reconnect}
                  </Button>
                ) : null}
                <Button
                  onClick={() => onEditChange({ connectionId: account.connection_id, kind: 'rename' })}
                  size="xs"
                  variant="text"
                >
                  {copy.rename}
                </Button>
                <Button onClick={() => onRemove(account)} size="xs" variant="text">
                  {copy.remove}
                </Button>
              </>
            }
            copy={copy}
            key={account.connection_id}
          />
        )
      )}

      {retired.length > 0 ? <RetiredFold accounts={retired} copy={copy} onRemove={onRemove} /> : null}

      {edit?.kind === 'add' ? (
        <AliasEditor
          accounts={everyAccount}
          busy={adding}
          error={edit.error}
          hint={copy.aliasHint}
          initial={edit.draft ?? ''}
          label={copy.aliasLabel}
          onCancel={() => onEditChange(null)}
          onSubmit={alias => void add(alias)}
          submitLabel={copy.add}
        />
      ) : accounts.length > 0 ? (
        <div className="py-2">
          <Button disabled={connecting} onClick={() => onEditChange({ kind: 'add' })} size="xs" variant="secondary">
            {copy.addAnother}
          </Button>
        </div>
      ) : null}
    </section>
  )
}

function AccountLine({ account, actions, copy }: { account: AccountRow; actions: ReactNode; copy: AccountsCopy }) {
  const named = account.alias ? account.label : undefined
  const status = account.status === 'active' ? null : account.status

  return (
    <ListRow
      action={actions}
      description={
        named || status ? (
          <span className="flex min-w-0 items-center gap-1.5">
            {named ? <span className="truncate">{named}</span> : null}
            {status ? (
              <Badge size="xs" variant={STATUS_VIEW[status].variant}>
                {copy.status[status]}
              </Badge>
            ) : null}
          </span>
        ) : undefined
      }
      title={<span className="truncate">{accountName(account)}</span>}
      wide
    />
  )
}

function RetiredFold({
  accounts,
  copy,
  onRemove
}: {
  accounts: readonly AccountRow[]
  copy: AccountsCopy
  onRemove: (account: AccountRow) => void
}) {
  return (
    <details className="group py-1.5" data-slot="connector-accounts-retired">
      <summary className="flex cursor-pointer list-none items-center gap-1.5 text-[0.72rem] text-(--ui-text-tertiary)">
        <Codicon
          className="shrink-0 transition-transform duration-100 group-open:rotate-90"
          name="chevron-right"
          size="0.75rem"
        />
        {copy.retiredCount(accounts.length)}
      </summary>
      {accounts.map(account => (
        <ListRow
          action={
            <Button onClick={() => onRemove(account)} size="xs" variant="text">
              {copy.remove}
            </Button>
          }
          description={copy.status.retired}
          key={account.connection_id}
          title={<span className="truncate">{accountName(account)}</span>}
          wide
        />
      ))}
    </details>
  )
}

interface AliasEditorProps {
  accounts: readonly AccountRow[]
  busy?: boolean
  error?: string
  hint?: string
  initial: string
  label: string
  onCancel: () => void
  onSubmit: (alias: string) => void
  self?: string
  submitLabel: string
  /** Submitting this value closes the editor without a write. */
  unchanged?: string
}

function AliasEditor({
  accounts,
  busy = false,
  error: refused,
  hint,
  initial,
  label,
  onCancel,
  onSubmit,
  self,
  submitLabel,
  unchanged
}: AliasEditorProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.accounts
  const [value, setValue] = useState(initial)
  const [invalid, setInvalid] = useState<null | string>(null)
  const [edited, setEdited] = useState(false)
  const error = invalid ?? (edited ? undefined : refused)

  const submit = () => {
    const alias = value.trim()

    if (alias === unchanged) {
      onCancel()

      return
    }

    const problem = aliasProblem(alias, accounts, self)

    if (problem) {
      setInvalid(problem === 'taken' ? copy.aliasTaken : copy.aliasInvalid)

      return
    }

    onSubmit(alias)
  }

  return (
    <form
      className="grid gap-1.5 py-2"
      data-slot="connector-alias-editor"
      onSubmit={event => {
        event.preventDefault()
        submit()
      }}
    >
      <div className="flex items-center gap-2">
        <Input
          aria-invalid={error ? true : undefined}
          aria-label={label}
          autoFocus
          maxLength={32}
          onChange={event => {
            setValue(event.target.value.toLowerCase())
            setInvalid(null)
            setEdited(true)
          }}
          placeholder={label}
          size="sm"
          value={value}
        />
        <Button loading={busy} size="xs" type="submit">
          {submitLabel}
        </Button>
        <Button disabled={busy} onClick={onCancel} size="xs" type="button" variant="text">
          {t.common.cancel}
        </Button>
      </div>
      {error ? (
        <p className="text-[0.7rem] text-destructive" role="alert">
          {error}
        </p>
      ) : hint ? (
        <p className="text-[0.7rem] text-(--ui-text-tertiary)">{hint}</p>
      ) : null}
    </form>
  )
}
