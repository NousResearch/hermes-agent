/**
 * The settings-lock pill, next to the approval-mode pill it most often protects.
 *
 * The lock is enforced on the gateway inside `save_config`, so this is a window onto it, not the
 * mechanism: the pill says whether locked settings can currently be changed, and offers to open
 * a time-boxed unlock window (with the operator password when one is set).
 */

import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import type { StatusbarItem } from '@/app/shell/statusbar-controls'
import { Button } from '@/components/ui/button'
import { DropdownMenuLabel, DropdownMenuSeparator } from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { Lock } from '@/lib/icons'
import {
  $settingsLock,
  relockSettings,
  type SettingsLockRequester,
  syncSettingsLock,
  unlockSettings
} from '@/store/settings-lock'

function remainingLabel(until: number | null): string {
  const seconds = Math.max(0, Math.round((until ?? 0) - Date.now() / 1000))

  return seconds >= 60 ? `${Math.floor(seconds / 60)}m` : `${seconds}s`
}

export function useSettingsLockStatusbarItem(
  requestGateway: SettingsLockRequester,
  gatewayReady: boolean
): StatusbarItem {
  const { t } = useI18n()
  const copy = t.shell.settingsLock
  const status = useStore($settingsLock)
  const [password, setPassword] = useState('')
  const [failed, setFailed] = useState(false)

  // Gated on readiness, not just mount: a fetch fired before the gateway opens fails, and with
  // no other trigger the pill would stay hidden on an install that IS locked.
  useEffect(() => {
    if (!gatewayReady) {
      return
    }
    void syncSettingsLock(requestGateway)
  }, [gatewayReady, requestGateway])

  // While a window is open the pill counts down, so "unlocked" is never a state you forget you left on.
  useEffect(() => {
    if (!status.unlocked) {
      return
    }

    const timer = setInterval(() => void syncSettingsLock(requestGateway), 15_000)

    return () => clearInterval(timer)
  }, [requestGateway, status.unlocked])

  const submit = useCallback(async () => {
    const ok = await unlockSettings(requestGateway, password)
    setFailed(!ok)

    if (ok) {
      setPassword('')
    }
  }, [password, requestGateway])

  const label = useMemo(() => {
    if (!status.enabled) {
      return copy.title
    }

    return status.unlocked ? copy.remaining(remainingLabel(status.unlockedUntil)) : copy.locked
  }, [copy, status.enabled, status.unlocked, status.unlockedUntil])

  return {
    // Hidden entirely on an install with no lock: an inert padlock in everyone's status bar is noise.
    hidden: !status.enabled,
    className: status.unlocked ? undefined : 'bg-(--chrome-action-hover) text-foreground',
    icon: <Lock className="size-3.5" />,
    id: 'settings-lock',
    label,
    menuAlign: 'end',
    menuClassName: 'w-80 p-1',
    menuContent: (
      <>
        <DropdownMenuLabel>{copy.title}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <div className="flex flex-col gap-2 px-2 py-1.5">
          <p className="text-[0.6875rem] leading-snug text-(--ui-text-tertiary)">
            {status.unusable
              ? copy.unusable
              : status.unlocked
                ? copy.unlockedDescription
                : copy.lockedDescription}
          </p>
          {status.keys.length ? (
            <div className="flex flex-col gap-0.5">
              <span className="text-[0.625rem] uppercase tracking-wide text-(--ui-text-quaternary)">
                {copy.lockedPaths}
              </span>
              {status.keys.map(key => (
                <span className="font-mono text-[0.6875rem] text-(--ui-text-tertiary)" key={key}>
                  {key}
                </span>
              ))}
            </div>
          ) : null}
          {status.unlocked ? (
            <Button onClick={() => void relockSettings(requestGateway)} size="sm" variant="secondary">
              {copy.relock}
            </Button>
          ) : (
            <>
              {status.passwordRequired ? (
                <Input
                  aria-label={copy.passwordLabel}
                  autoComplete="off"
                  onChange={event => {
                    setPassword(event.target.value)
                    setFailed(false)
                  }}
                  // The menu treats keystrokes as type-ahead navigation; the field needs them.
                  onKeyDown={event => {
                    event.stopPropagation()

                    if (event.key === 'Enter') {
                      void submit()
                    }
                  }}
                  placeholder={copy.passwordPlaceholder}
                  type="password"
                  value={password}
                />
              ) : null}
              {failed ? <p className="text-[0.6875rem] text-(--ui-danger)">{copy.wrongPassword}</p> : null}
              <Button
                disabled={status.passwordRequired && !password}
                onClick={() => void submit()}
                size="sm"
              >
                {copy.unlock}
              </Button>
            </>
          )}
        </div>
      </>
    ),
    title: copy.ariaLabel(status.unlocked ? copy.unlocked : copy.locked),
    // 'menu' is what makes `menuContent` render at all — an 'action' item with no onSelect is a
    // button that does nothing when clicked.
    variant: 'menu'
  }
}
