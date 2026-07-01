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

import { PROFILES_ROUTE, SETTINGS_ROUTE } from '../../routes'

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

// Bottom-left account panel (Codex account menu, minimal). Collapsed: avatar
// (initial) + display name + plan badge. Click → a popover menu with email,
// 个人资料 (profile), 设置 (settings), 剩余用量 (usage — only when quota data is on
// hand), 退出登录 (logout). Codex layout + our light-purple accent, no extra text.
// Rendered only on managed builds when signed in (the auth gate handles the
// signed-out case); on a managed-disabled build the panel stays hidden.
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
  const plan = account.plan.trim()

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
            'flex w-full items-center gap-2 rounded-lg px-1.5 py-1 text-left transition-colors',
            'hover:bg-(--ui-control-hover-background)',
            open && 'bg-(--ui-control-active-background)'
          )}
          type="button"
        >
          <span
            aria-hidden
            className="grid size-6 shrink-0 place-items-center rounded-full text-[0.6875rem] font-semibold uppercase leading-none text-white"
            style={{ backgroundColor: tint }}
          >
            {initial}
          </span>
          <span className="min-w-0 flex-1 truncate text-[0.8125rem] font-medium text-(--ui-text-secondary)">
            {name}
          </span>
          {plan ? (
            <span className="shrink-0 rounded-sm bg-[color-mix(in_srgb,var(--theme-primary)_15%,transparent)] px-1.5 py-0.5 text-[0.625rem] font-semibold uppercase tracking-wide text-(--theme-primary)">
              {plan}
            </span>
          ) : null}
          <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="chevron-up" size="0.75rem" />
        </button>
      </DropdownMenuTrigger>

      <DropdownMenuContent align="start" className="w-56" side="top" sideOffset={6}>
        {/* Email header (identity), non-interactive. Omitted when unknown. */}
        {account.email ? (
          <>
            <div className="truncate px-2 py-1.5 text-xs text-(--ui-text-tertiary)" title={account.email}>
              {account.email}
            </div>
            <DropdownMenuSeparator />
          </>
        ) : null}

        <DropdownMenuItem onSelect={() => navigate(PROFILES_ROUTE)}>
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
