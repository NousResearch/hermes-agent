import { useEffect, useRef, useState } from 'react'

import { capabilityScoped, type ProfileScope, profileScopeKey } from '@/api/client'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { McpCatalogEntry } from '@/hermes'
import { useI18n } from '@/i18n'
import { completeMcpDesktopOAuth, McpOAuthCancelled } from '@/lib/mcp-dashboard-oauth'
import type { McpServers } from '@/lib/mcp-servers'

import { createLocalAccount, localAccountCatalog, localAccountName, localAccountProblem } from './data/local-accounts'

export interface LocalAccountsProps {
  catalog: readonly McpCatalogEntry[]
  servers: McpServers
  serverName: string
  profile: ProfileScope
  pending: boolean
  onChanged: () => Promise<void>
  onOpenAccount: (name: string) => void
}

export function LocalAccounts(props: LocalAccountsProps) {
  const entry = localAccountCatalog(props.catalog, props.servers, props.serverName)

  return entry ? (
    <AccountEditor {...props} entry={entry} key={`${profileScopeKey(props.profile)}:${props.serverName}`} />
  ) : null
}

interface AccountEditorProps extends LocalAccountsProps {
  entry: McpCatalogEntry
}

function AccountEditor({ entry, servers, serverName, profile, pending, onChanged, onOpenAccount }: AccountEditorProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.accounts
  const [editing, setEditing] = useState(false)
  const [alias, setAlias] = useState('')
  const [savedName, setSavedName] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const alive = useRef(true)
  const cancelled = useRef(false)
  const running = useRef(false)

  // eslint-disable-next-line no-restricted-syntax -- mount lifetime, not mirrored reactive state; StrictMode must re-arm it.
  useEffect(() => {
    alive.current = true

    return () => {
      alive.current = false
      cancelled.current = true
    }
  }, [])

  const isCancelled = () => !alive.current || cancelled.current

  const connect = async () => {
    if (running.current || pending) {
      return
    }

    const name = savedName ?? localAccountName(entry.name, alias.trim())
    const problem = savedName ? null : localAccountProblem(entry.name, alias.trim(), Object.keys(servers))

    if (problem) {
      setError(problem === 'taken' ? copy.aliasTaken : copy.aliasInvalid)

      return
    }

    // Capture this route before creation. A foreground switch cannot redirect OAuth to another profile.
    const scope = capabilityScoped(profile)
    running.current = true
    cancelled.current = false
    setBusy(true)
    setError(null)
    let created = false

    try {
      if (!savedName) {
        await createLocalAccount(name, entry.name, servers[serverName]?.tools, scope)
        created = true

        if (!alive.current) {
          return
        }

        if (!cancelled.current) {
          setSavedName(name)
        }
      }

      if (isCancelled()) {
        return
      }

      await completeMcpDesktopOAuth({
        serverName: name,
        profile: scope,
        cancelled: isCancelled
      })

      if (!alive.current) {
        return
      }

      await onChanged()
      created = false

      if (isCancelled()) {
        return
      }

      setEditing(false)
      setSavedName(null)
      setAlias('')
      onOpenAccount(name)
    } catch (err) {
      if (!isCancelled() && !(err instanceof McpOAuthCancelled)) {
        setError(err instanceof Error ? err.message : String(err))
      }
    } finally {
      // Keep the new entry out of background probes until OAuth succeeds. If sign-in
      // fails or is cancelled, expose it for an explicit retry or removal instead.
      if (created && alive.current) {
        try {
          await onChanged()
        } catch (err) {
          if (!cancelled.current) {
            setError(err instanceof Error ? err.message : String(err))
          }
        }
      }

      running.current = false

      if (alive.current) {
        setBusy(false)
      }
    }
  }

  return (
    <section aria-label={copy.heading} className="grid gap-2 px-5 py-3">
      <div className="flex flex-wrap items-center gap-2">
        {Object.entries(servers)
          .filter(([, server]) => server.auth === 'oauth' && server.url === entry.url)
          .map(([name]) => (
            <Button
              aria-current={name === serverName ? 'true' : undefined}
              key={name}
              onClick={() => onOpenAccount(name)}
              size="xs"
              variant="outline"
            >
              {name}
            </Button>
          ))}
        {!editing ? (
          <Button disabled={pending || busy} onClick={() => setEditing(true)} size="xs" variant="text">
            {copy.addAnother}
          </Button>
        ) : null}
      </div>
      {editing ? (
        <form
          className="grid gap-2"
          onSubmit={event => {
            event.preventDefault()
            void connect()
          }}
        >
          <p className="text-xs text-(--ui-text-secondary)">{copy.localHint}</p>
          <div className="flex items-center gap-2">
            <Input
              aria-invalid={error ? true : undefined}
              aria-label={copy.aliasLabel}
              autoFocus
              disabled={busy || savedName !== null}
              maxLength={32}
              onChange={event => {
                setAlias(event.target.value)
                setError(null)
              }}
              placeholder={copy.aliasLabel}
              size="sm"
              value={alias}
            />
            <Button disabled={busy || pending} loading={busy} size="xs" type="submit">
              {savedName ? t.connectorsPage.card.verb.tryAgain : copy.add}
            </Button>
            <Button
              onClick={() => {
                cancelled.current = true
                setEditing(false)
                setSavedName(null)
                setAlias('')
                setError(null)
              }}
              size="xs"
              type="button"
              variant="text"
            >
              {t.common.cancel}
            </Button>
          </div>
          <p className="text-xs text-(--ui-text-tertiary)">{copy.aliasHint}</p>
          <code className="text-xs">{savedName ?? localAccountName(entry.name, alias.trim())}</code>
          {savedName ? (
            <Button onClick={() => onOpenAccount(savedName)} size="xs" type="button" variant="text">
              {savedName}
            </Button>
          ) : null}
          {error ? (
            <p className="text-xs text-(--ui-red)" role="alert">
              {error}
            </p>
          ) : null}
        </form>
      ) : null}
    </section>
  )
}
