/**
 * The Browser bar's key button: fill a saved login into THIS tab's page.
 *
 * Lists vault logins (Hermes vault + every unlocked password manager + any
 * other login source the backend lists) whose saved origins match the page,
 * offers an unlock for each locked manager, and on a pick asks the backend to
 * fill it (`vault.preview.fill`). The renderer never holds the password: it
 * mints an opaque binding for this tab's live document, the backend resolves
 * the secret and writes it into that document through `preview.act` requests,
 * and the answer carries counts only. The form is never submitted.
 *
 * Only metadata (labels, identifiers, origins) lives in component state; the
 * master password for an unlock lives in a ref the request consumes and wipes,
 * never in React state that outlives the dialog or in a query cache.
 */

import { useStore } from '@nanostores/react'
import { useCallback, useRef, useState } from 'react'

import { SETTINGS_ROUTE } from '@/app/routes'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { KeyRound } from '@/lib/icons'
import { $activeConnectionId } from '@/store/connections'
import { requestGatewayForAgent } from '@/store/gateway'
import { notify } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { requestRoute } from '@/store/recovery-requests'

import { forgetPreviewVaultBinding, openUserPreviewVaultBinding } from './preview-vault'

interface VaultLogin {
  allowed_origins?: null | string[]
  backend: string
  id: string
  identifier?: null | string
  kind: string
  label: string
  origin?: null | string
}

interface VaultSource {
  display_name: string
  enabled: boolean
  installed: boolean
  name: string
  needs_unlock: boolean
  unlocked: boolean
}

interface FillResult {
  error?: null | string
  filled_fields?: null | number
  success: boolean
}

/** `https://Example.com:443/login` → `https://example.com`; null for pages a login can't belong to. */
export function pageOrigin(url: string): null | string {
  try {
    const parsed = new URL(url)

    return /^https?:$/.test(parsed.protocol) ? parsed.origin : null
  } catch {
    return null
  }
}

/** Logins saved for exactly this origin (no parent-domain or wildcard match, same rule as the fill). */
export function loginsForOrigin(items: readonly VaultLogin[], origin: string): VaultLogin[] {
  return items.filter(
    item => item.kind === 'login' && (item.origin === origin || (item.allowed_origins ?? []).includes(origin))
  )
}

const lockedSources = (sources: readonly VaultSource[]) =>
  sources.filter(source => source.needs_unlock && source.enabled && source.installed && !source.unlocked)

interface PreviewVaultButtonProps {
  tabId: string
  url: string
}

export function PreviewVaultButton({ tabId, url }: PreviewVaultButtonProps) {
  const { t } = useI18n()
  const copy = t.preview.web
  const unlockCopy = t.settings.vault.sources
  const connectionId = useStore($activeConnectionId)
  const profile = useStore($activeGatewayProfile)
  const origin = pageOrigin(url)
  const [open, setOpen] = useState(false)
  const [logins, setLogins] = useState<null | VaultLogin[]>(null)
  const [locked, setLocked] = useState<VaultSource[]>([])
  const [unlockTarget, setUnlockTarget] = useState<null | VaultSource>(null)
  const [masterPassword, setMasterPassword] = useState('')
  const [unlockError, setUnlockError] = useState<null | string>(null)
  const [busy, setBusy] = useState(false)
  const pendingMasterPassword = useRef('')

  const rpc = useCallback(
    <T,>(method: string, params: Record<string, unknown> = {}) =>
      requestGatewayForAgent<T>(connectionId, profile, method, params, undefined, undefined, {
        spawnPriority: 'foreground'
      }),
    [connectionId, profile]
  )

  const load = useCallback(async () => {
    setLogins(null)

    try {
      const [list, sources] = await Promise.all([
        rpc<{ items?: VaultLogin[] }>('vault.list'),
        rpc<{ sources?: VaultSource[] }>('vault.sources')
      ])

      setLogins(origin ? loginsForOrigin(list.items ?? [], origin) : [])
      setLocked(lockedSources(sources.sources ?? []))
    } catch (error) {
      setLogins([])
      notify({ kind: 'error', message: copy.vaultLoadFailed, detail: error instanceof Error ? error.message : '' })
    }
  }, [copy.vaultLoadFailed, origin, rpc])

  const fill = async (login: VaultLogin) => {
    const target = openUserPreviewVaultBinding(tabId)

    if (!target) {
      notify({ kind: 'error', message: copy.vaultPageNotReady })

      return
    }

    try {
      const result = await rpc<FillResult>('vault.preview.fill', { handle: login.id, target })

      if (result.success) {
        notify({ kind: 'success', message: copy.vaultFilled(login.label) })
      } else {
        notify({ kind: 'error', message: copy.vaultFillFailed, detail: result.error ?? '' })
      }
    } catch (error) {
      notify({ kind: 'error', message: copy.vaultFillFailed, detail: error instanceof Error ? error.message : '' })
    } finally {
      forgetPreviewVaultBinding(target)
    }
  }

  const closeUnlock = () => {
    setUnlockTarget(null)
    setMasterPassword('')
    setUnlockError(null)
    pendingMasterPassword.current = ''
  }

  const submitUnlock = async () => {
    if (!unlockTarget || !pendingMasterPassword.current) {
      return
    }

    const password = pendingMasterPassword.current
    pendingMasterPassword.current = ''
    setBusy(true)

    try {
      await rpc('vault.unlock', { name: unlockTarget.name, password })
      notify({ kind: 'success', message: unlockCopy.unlocked(unlockTarget.display_name) })
      closeUnlock()
      setOpen(true)
      void load()
    } catch (error) {
      setUnlockError(error instanceof Error ? error.message : String(error))
    } finally {
      setBusy(false)
    }
  }

  const openVaultSettings = () => {
    const params = new URLSearchParams({ tab: 'vault', ...(origin ? { kind: 'login', origin } : {}) })
    requestRoute(`${SETTINGS_ROUTE}?${params.toString()}`)
  }

  return (
    <>
      <DropdownMenu
        onOpenChange={next => {
          setOpen(next)

          if (next) {
            void load()
          }
        }}
        open={open}
      >
        <Tip label={copy.vaultFill} placement="toolbar">
          <DropdownMenuTrigger asChild>
            <Button
              aria-label={copy.vaultFill}
              className="self-center bg-transparent opacity-60 select-none hover:opacity-100 data-[state=open]:opacity-100 [-webkit-app-region:no-drag]"
              data-slot="preview-vault-button"
              disabled={!origin}
              onPointerDown={event => event.stopPropagation()}
              size="icon-xs"
              type="button"
              variant="ghost"
            >
              <Codicon name="key" size="0.8125rem" />
            </Button>
          </DropdownMenuTrigger>
        </Tip>
        <DropdownMenuContent align="end" className="min-w-60" data-slot="preview-vault-menu">
          <DropdownMenuLabel>{origin ? copy.vaultTitle(origin.replace(/^https?:\/\//, '')) : copy.vaultFill}</DropdownMenuLabel>
          {logins === null ? (
            <DropdownMenuItem disabled>{copy.vaultLoading}</DropdownMenuItem>
          ) : logins.length === 0 ? (
            <DropdownMenuItem disabled>{copy.vaultEmpty}</DropdownMenuItem>
          ) : (
            logins.map(login => (
              <DropdownMenuItem data-vault-handle={login.id} key={login.id} onSelect={() => void fill(login)}>
                <KeyRound className="size-3.5 shrink-0 opacity-70" />
                <span className="min-w-0 flex-1 truncate">{login.identifier || login.label}</span>
                <span className="shrink-0 text-[0.6875rem] text-muted-foreground">{login.label}</span>
              </DropdownMenuItem>
            ))
          )}
          {locked.length > 0 && <DropdownMenuSeparator />}
          {locked.map(source => (
            <DropdownMenuItem
              key={source.name}
              onSelect={() => {
                setUnlockTarget(source)
                setUnlockError(null)
              }}
            >
              <Codicon name="lock" size="0.75rem" />
              {copy.vaultUnlock(source.display_name)}
            </DropdownMenuItem>
          ))}
          <DropdownMenuSeparator />
          <DropdownMenuItem onSelect={openVaultSettings}>
            <Codicon name="settings-gear" size="0.75rem" />
            {copy.vaultOpenSettings}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog onOpenChange={next => !next && closeUnlock()} open={unlockTarget !== null}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle icon={KeyRound}>{unlockCopy.unlockTitle(unlockTarget?.display_name ?? '')}</DialogTitle>
            <DialogDescription>{unlockCopy.unlockDescription}</DialogDescription>
          </DialogHeader>
          <form
            className="grid gap-3"
            onSubmit={event => {
              event.preventDefault()
              pendingMasterPassword.current = masterPassword
              setMasterPassword('')
              void submitUnlock()
            }}
          >
            <Input
              autoComplete="current-password"
              autoFocus
              disabled={busy}
              onChange={event => setMasterPassword(event.target.value)}
              placeholder={unlockCopy.masterPasswordPlaceholder}
              type="password"
              value={masterPassword}
            />
            {unlockError && <p className="text-xs text-destructive">{unlockError}</p>}
            <DialogFooter>
              <Button onClick={closeUnlock} type="button" variant="ghost">
                {t.common.cancel}
              </Button>
              <Button disabled={busy || !masterPassword} type="submit">
                {busy ? unlockCopy.unlocking : unlockCopy.unlock}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </>
  )
}
