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
 *
 * OWNER SCOPE (#150283 P1-1/P1-2): a tab may belong to profile A while B is the
 * active view, so every vault RPC is routed through the owner this tab's
 * session resolves to — never the ambient `$activeConnectionId` /
 * `$activeGatewayProfile`. The unlock dialog captures that owner on open and
 * fences `submitUnlock` to it (an owner change wipes the password, closes the
 * dialog, and refuses a submit even if the owner moved after the await), so a
 * master password typed for A can never be sent to B. `load()` is keyed by the
 * owner + origin it started under and a stale reply (owner/origin changed since
 * the request started) is dropped instead of overwriting the current list.
 */

import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'

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
import { $previewTabs } from '@/store/preview'
import { currentSessionId } from '@/store/preview-ownership'
import { $activeGatewayProfile } from '@/store/profile'
import { requestRoute } from '@/store/recovery-requests'
import { knownSessionOwner, ownerLookupSessionRows } from '@/store/session'
import { profileScopeForSessionOwner, requestForSessionProfile, type SessionOwnerScope } from '@/store/session-request-router'

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

/**
 * The (connectionId, profile) that owns `tabId`'s page, resolved from the tab's
 * session — NOT the ambient view. A tab opened in a draft / a user-opened
 * browser tab with no session owner falls back to the active view (it has no
 * other owner), but a tab that belongs to a backend session is always routed to
 * that session's owner even while another profile is active (#150283 P1-1).
 */
export function resolveTabVaultOwner(tabId: string): SessionOwnerScope {
  const tab = $previewTabs.get().find(candidate => candidate.id === tabId)
  const owner = knownSessionOwner(ownerLookupSessionRows(), currentSessionId(tab?.sessionId))

  if (owner != null && (typeof owner !== 'object' || owner.connectionId.trim())) {
    return owner
  }

  // Session-less tab: the owner is the view it is shown in.
  const connectionId = $activeConnectionId.get()
  const profile = $activeGatewayProfile.get()

  return connectionId ? { connectionId, profile: profile || 'default' } : profile || 'default'
}

/** A stable key for owner equality across async boundaries. */
export function vaultOwnerKey(owner: SessionOwnerScope): string {
  const scope = profileScopeForSessionOwner(owner)

  if (scope == null) {
    return '::'
  }

  if (typeof scope === 'string') {
    return `::${scope.trim() || 'default'}`
  }

  return `${scope.connectionId ?? ''}::${(scope.profile ?? '').trim() || 'default'}`
}

interface PreviewVaultButtonProps {
  tabId: string
  url: string
}

export function PreviewVaultButton({ tabId, url }: PreviewVaultButtonProps) {
  const { t } = useI18n()
  const copy = t.preview.web
  const unlockCopy = t.settings.vault.sources
  // Re-render on a view change so the owner is re-resolved for a fresh menu,
  // but the OWNER used by any in-flight op is captured below, never read live.
  const activeConnectionId = useStore($activeConnectionId)
  const activeProfile = useStore($activeGatewayProfile)
  const origin = pageOrigin(url)
  const [open, setOpen] = useState(false)
  const [logins, setLogins] = useState<null | VaultLogin[]>(null)
  const [locked, setLocked] = useState<VaultSource[]>([])
  const [unlockTarget, setUnlockTarget] = useState<null | VaultSource>(null)
  const [masterPassword, setMasterPassword] = useState('')
  const [unlockError, setUnlockError] = useState<null | string>(null)
  const [busy, setBusy] = useState(false)
  const pendingMasterPassword = useRef('')
  // The owner the open unlock dialog was captured for; `submitUnlock` fences
  // against it and refuses if the tab's owner moved (#150283 P1-2).
  const unlockOwner = useRef<SessionOwnerScope>(undefined)
  // Sequence token so a stale `load()` reply (owner/origin changed since it
  // started) cannot overwrite the current list (#150283 P1-1).
  const loadToken = useRef(0)
  const closeUnlockRef = useRef<() => void>(() => {})

  // An owner change while the unlock dialog is open closes it and wipes BOTH
  // the visible field and the pending password: a master password typed for A
  // must never survive a switch to B (#150283 P1-2). Keyed to the active view
  // tuple so an identical profile name on a different connection still fires.
  useEffect(() => {
    const captured = unlockOwner.current

    if (captured !== undefined && vaultOwnerKey(captured) !== vaultOwnerKey(resolveTabVaultOwner(tabId))) {
      closeUnlockRef.current()
    }
  }, [activeConnectionId, activeProfile, tabId])

  const rpc = useCallback(
    <T,>(owner: SessionOwnerScope, method: string, params: Record<string, unknown> = {}) => {
      // Fallback dispatcher for a bare/unknown owner — unreachable in practice
      // because resolveTabVaultOwner always yields a concrete owner, but the
      // router requires it. Scoped to the view the owner was derived from.
      const ambient = <R,>(m: string, p?: Record<string, unknown>) =>
        requestGatewayForAgent<R>($activeConnectionId.get(), $activeGatewayProfile.get() || 'default', m, p, undefined, undefined, {
          spawnPriority: 'foreground'
        })

      return requestForSessionProfile<T>(owner, ambient, method, params)
    },
    []
  )

  const load = useCallback(async () => {
    setLogins(null)
    const owner = resolveTabVaultOwner(tabId)
    const requestOwner = vaultOwnerKey(owner)
    const requestOrigin = origin
    const token = (loadToken.current += 1)
    // True when the owner this request was keyed to, and the page origin, are
    // still current — otherwise this reply belongs to a superseded request.
    const current = () =>
      token === loadToken.current &&
      requestOrigin === pageOrigin(url) &&
      requestOwner === vaultOwnerKey(resolveTabVaultOwner(tabId))

    try {
      const [list, sources] = await Promise.all([
        rpc<{ items?: VaultLogin[] }>(owner, 'vault.list'),
        rpc<{ sources?: VaultSource[] }>(owner, 'vault.sources')
      ])

      if (!current()) {
        return
      }

      setLogins(requestOrigin ? loginsForOrigin(list.items ?? [], requestOrigin) : [])
      setLocked(lockedSources(sources.sources ?? []))
    } catch (error) {
      if (!current()) {
        return
      }

      setLogins([])
      notify({ kind: 'error', message: copy.vaultLoadFailed, detail: error instanceof Error ? error.message : '' })
    }
  }, [copy.vaultLoadFailed, origin, rpc, tabId, url])

  const fill = async (login: VaultLogin) => {
    const target = openUserPreviewVaultBinding(tabId)

    if (!target) {
      notify({ kind: 'error', message: copy.vaultPageNotReady })

      return
    }

    const owner = resolveTabVaultOwner(tabId)

    try {
      const result = await rpc<FillResult>(owner, 'vault.preview.fill', { handle: login.id, target })

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
    unlockOwner.current = undefined
  }

  // Latest closeUnlock for the owner-change effect (assigned in render, not in
  // an effect, so the lint ban on ref-mirroring inside useEffect does not apply).
  closeUnlockRef.current = closeUnlock

  const openUnlock = (source: VaultSource) => {
    unlockOwner.current = resolveTabVaultOwner(tabId)
    setUnlockTarget(source)
    setUnlockError(null)
  }

  const submitUnlock = async () => {
    const owner = unlockOwner.current

    // The owner this dialog was opened for must still own the tab. A profile
    // switch while the dialog was open would otherwise send the master password
    // to whatever owner is now active (#150283 P1-2).
    if (!unlockTarget || !pendingMasterPassword.current || owner === undefined) {
      pendingMasterPassword.current = ''

      return
    }

    if (vaultOwnerKey(owner) !== vaultOwnerKey(resolveTabVaultOwner(tabId))) {
      pendingMasterPassword.current = ''
      closeUnlock()

      return
    }

    const password = pendingMasterPassword.current
    pendingMasterPassword.current = ''
    const target = unlockTarget
    setBusy(true)

    try {
      await rpc(owner, 'vault.unlock', { name: target.name, password })

      // Re-validate AFTER the await: the user may have switched the tab's owner
      // while the unlock RPC was in flight. The unlock itself went to the
      // captured owner's socket; refuse to act on a now-foreign owner.
      if (vaultOwnerKey(owner) !== vaultOwnerKey(resolveTabVaultOwner(tabId))) {
        closeUnlock()

        return
      }

      notify({ kind: 'success', message: unlockCopy.unlocked(target.display_name) })
      closeUnlock()
      setOpen(true)
      void load()
    } catch (error) {
      setUnlockError(error instanceof Error ? error.message : String(error))
    } finally {
      setBusy(false)
    }
  }

  if (!origin) {
    return null
  }

  const openVaultSettings = () => {
    const params = new URLSearchParams({ tab: 'vault', kind: 'login', origin })
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
          <DropdownMenuLabel>{copy.vaultTitle(origin.replace(/^https?:\/\//, ''))}</DropdownMenuLabel>
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
            <DropdownMenuItem key={source.name} onSelect={() => openUnlock(source)}>
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
