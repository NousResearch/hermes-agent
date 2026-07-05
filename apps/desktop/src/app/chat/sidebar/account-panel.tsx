import { useStore } from '@nanostores/react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { profileColor } from '@/lib/profile-color'
import { cn } from '@/lib/utils'
import { $authState, type AuthAccount, signOutAccount } from '@/store/auth'

import { PROFILE_STATS_ROUTE, SETTINGS_ROUTE } from '../../routes'

// The signed-in display name: prefer an explicit name, else the email's local
// part, else a generic fallback ("账户"). The avatar shows its first letter.
function displayName(account: AuthAccount, fallback: string): string {
  if (account.name) {
    return account.name
  }

  if (account.email) {
    return account.email.split('@')[0] || account.email
  }

  return fallback
}

function initialOf(name: string): string {
  const match = name.replace(/[^\p{L}\p{N}]/gu, '').charAt(0)

  return (match || '?').toUpperCase()
}

// Bottom-left account panel (Codex account row, high-fidelity). The row is
// avatar (initial) + a two-line stack: display name over the signed-in email —
// no plan badge, no phone icon, no caret, matching the Codex reference. Click →
// a popover menu with 个人资料 (profile → the usage-stats page), 设置 (settings),
// 剩余用量 (usage — only when quota data is on hand), 退出登录 (logout). Rendered
// only on managed builds when signed in (the auth gate handles the signed-out
// case); on a managed-disabled build the panel stays hidden.
export function AccountPanel() {
  const { t } = useI18n()
  const a = t.auth.account
  const navigate = useNavigate()
  const { account, enabled, status } = useStore($authState)
  const [open, setOpen] = useState(false)

  // No account gate on this build (managed off), or not signed in yet → the
  // account panel has nothing to show. The login gate covers signed-out.
  if (enabled === false || status !== 'signed-in') {
    return null
  }

  const name = displayName(account, a.fallbackName)
  const initial = initialOf(name)
  // Deterministic tint for the avatar, seeded off the identity (email→name).
  const tint = profileColor(account.email || name) ?? 'var(--theme-primary)'
  const email = account.email.trim()

  // Usage is only shown when quota is genuinely available. The managed status
  // doesn't currently expose an account quota to the desktop, so this stays
  // omitted (per spec) — kept as a single flag so wiring real data later is a
  // one-line change.
  const usageLabel: null | string = null

  return (
    <DropdownMenu onOpenChange={setOpen} open={open}>
      <DropdownMenuTrigger asChild>
        <button
          aria-label={name}
          className={cn(
            'flex w-full items-center gap-2.5 rounded-lg px-1.5 py-1.5 text-left transition-colors',
            'hover:bg-(--ui-control-hover-background)',
            open && 'bg-(--ui-control-active-background)'
          )}
          type="button"
        >
          <span
            aria-hidden
            className="grid size-7 shrink-0 place-items-center rounded-full text-xs font-semibold uppercase leading-none text-white"
            style={{ backgroundColor: tint }}
          >
            {initial}
          </span>
          <span className="flex min-w-0 flex-1 flex-col leading-tight">
            <span className="truncate text-[0.8125rem] font-medium text-foreground">{name}</span>
            {email ? <span className="truncate text-[0.6875rem] text-(--ui-text-tertiary)">{email}</span> : null}
          </span>
        </button>
      </DropdownMenuTrigger>

      <DropdownMenuContent align="start" className="w-56" side="top" sideOffset={6}>
        <DropdownMenuItem onSelect={() => navigate(PROFILE_STATS_ROUTE)}>
          <Codicon name="account" size="0.875rem" />
          <span>{a.profile}</span>
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={() => navigate(`${SETTINGS_ROUTE}`)}>
          <Codicon name="settings-gear" size="0.875rem" />
          <span>{a.settings}</span>
        </DropdownMenuItem>
        {usageLabel ? (
          <DropdownMenuItem onSelect={() => navigate(`${SETTINGS_ROUTE}`)}>
            <Codicon name="graph" size="0.875rem" />
            <span className="flex-1">{a.usage}</span>
            <span className="text-xs text-(--ui-text-tertiary)">{usageLabel}</span>
          </DropdownMenuItem>
        ) : null}

        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={() => void signOutAccount()} variant="destructive">
          <Codicon name="sign-out" size="0.875rem" />
          <span>{a.logout}</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
