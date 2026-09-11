import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { $activeGatewayProfile } from '@/store/profile'
import { noteProfileChange, $profileLocks, tryUnlockProfile, $unlockedProfile } from '@/store/profile-lock'

/**
 * Full-screen gate for the per-profile passcode lock (#94028).
 *
 * Renders whenever the active gateway profile has a passcode set and has not
 * been unlocked in the current activation (boot + live profile swap alike).
 * An unlock is scoped to one activation: switching to another profile forgets
 * it, so returning re-prompts instead of switching silently.
 */
export function ProfileLockGate() {
  const { t } = useI18n()
  const g = t.profileLock
  const active = useStore($activeGatewayProfile)
  const locks = useStore($profileLocks)
  const unlocked = useStore($unlockedProfile)
  const locked = Boolean(locks[active]) && unlocked !== active
  const [passcode, setPasscode] = useState('')
  const [error, setError] = useState(false)
  const [busy, setBusy] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  // Re-evaluate on every profile change; an unlock never survives a switch.
  useEffect(() => {
    noteProfileChange(active)
    setPasscode('')
    setError(false)
    setBusy(false)
  }, [active])

  useEffect(() => {
    if (locked) {
      inputRef.current?.focus()
    }
  }, [locked])

  if (!locked) {
    return null
  }

  const submit = async () => {
    if (!passcode || busy) {
      return
    }
    setBusy(true)
    setError(false)
    const ok = await tryUnlockProfile(active, passcode)
    setBusy(false)
    if (!ok) {
      setError(true)
      setPasscode('')
      inputRef.current?.focus()
    }
  }

  return (
    <div className="fixed inset-0 z-(--z-setup) flex items-center justify-center bg-(--ui-chat-surface-background) p-6">
      <form
        className="w-full max-w-sm space-y-4"
        onSubmit={event => {
          event.preventDefault()
          void submit()
        }}
      >
        <div className="space-y-1">
          <h2 className="text-lg font-semibold tracking-tight text-foreground">{g.gateTitle}</h2>
          <p className="text-sm text-muted-foreground">{g.gatePrompt(active)}</p>
        </div>
        <Input
          aria-label={g.passcodeLabel}
          autoComplete="off"
          disabled={busy}
          onChange={event => setPasscode(event.target.value)}
          placeholder={g.gatePlaceholder}
          ref={inputRef}
          type="password"
          value={passcode}
        />
        {error ? <p className="text-xs text-destructive">{g.wrongPasscode}</p> : null}
        <Button className="w-full" disabled={busy || !passcode} type="submit">
          {g.unlock}
        </Button>
      </form>
    </div>
  )
}
