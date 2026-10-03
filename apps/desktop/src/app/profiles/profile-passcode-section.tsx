import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Field } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { PanelSectionLabel } from '@/app/overlays/panel'
import { useI18n } from '@/i18n'
import { clearProfilePasscode, hasProfilePasscode, $profileLocks, setProfilePasscode } from '@/store/profile-lock'

const MIN_PASSCODE_LENGTH = 4

/**
 * Passcode lock management for one profile (#94028): set / change / remove
 * from the profile detail panel. The gate itself lives in
 * `components/profile-lock-gate.tsx`; this is only the configuration surface.
 */
export function ProfilePasscodeSection({ profileName }: { profileName: string }) {
  const { t } = useI18n()
  const g = t.profileLock
  useStore($profileLocks)
  const locked = hasProfilePasscode(profileName)
  const [dialog, setDialog] = useState<'remove' | 'set' | null>(null)

  return (
    <section className="space-y-2">
      <PanelSectionLabel>{g.sectionLabel}</PanelSectionLabel>
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs text-muted-foreground">{locked ? g.lockedStatus : g.unlockedStatus}</span>
        <div className="flex shrink-0 gap-1.5">
          {locked ? (
            <>
              <Button onClick={() => setDialog('set')} size="sm" variant="outline">
                {g.changePasscode}
              </Button>
              <Button onClick={() => setDialog('remove')} size="sm" variant="destructive">
                {g.removePasscode}
              </Button>
            </>
          ) : (
            <Button onClick={() => setDialog('set')} size="sm" variant="outline">
              {g.setPasscode}
            </Button>
          )}
        </div>
      </div>
      {dialog !== null && (
        <ProfilePasscodeDialog mode={dialog} onClose={() => setDialog(null)} profileName={profileName} />
      )}
    </section>
  )
}

function ProfilePasscodeDialog({
  mode,
  onClose,
  profileName
}: {
  mode: 'remove' | 'set'
  onClose: () => void
  profileName: string
}) {
  const { t } = useI18n()
  const g = t.profileLock
  const [passcode, setPasscode] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState<null | string>(null)
  const [saving, setSaving] = useState(false)

  const save = async () => {
    if (saving) {
      return
    }
    if (mode === 'set') {
      if (passcode.length < MIN_PASSCODE_LENGTH) {
        setError(g.passcodeTooShort)
        return
      }
      if (passcode !== confirm) {
        setError(g.passcodeMismatch)
        return
      }
    }
    setSaving(true)
    setError(null)
    try {
      if (mode === 'remove') {
        clearProfilePasscode(profileName)
      } else {
        await setProfilePasscode(profileName, passcode)
      }
      onClose()
    } catch {
      setError(g.failedSave)
      setSaving(false)
    }
  }

  return (
    <Dialog open onOpenChange={open => !open && !saving && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{mode === 'remove' ? g.removeConfirmTitle : g.setTitle}</DialogTitle>
          {mode === 'remove' ? <DialogDescription>{g.removeConfirmBody(profileName)}</DialogDescription> : null}
        </DialogHeader>

        {mode === 'remove' ? (
          <DialogFooter>
            <Button disabled={saving} onClick={onClose} variant="outline">
              {t.common.cancel}
            </Button>
            <Button disabled={saving} onClick={() => void save()} variant="destructive">
              {g.removePasscode}
            </Button>
          </DialogFooter>
        ) : (
          <div className="grid gap-4">
            <Field label={g.newPasscodeLabel}>
              <Input
                autoComplete="new-password"
                onChange={event => setPasscode(event.target.value)}
                placeholder={g.passcodePlaceholder}
                type="password"
                value={passcode}
              />
            </Field>
            <Field label={g.confirmPasscodeLabel}>
              <Input
                autoComplete="new-password"
                onChange={event => setConfirm(event.target.value)}
                placeholder={g.passcodePlaceholder}
                type="password"
                value={confirm}
              />
            </Field>
            {error ? <p className="text-xs text-destructive">{error}</p> : null}
            <DialogFooter>
              <Button disabled={saving} onClick={onClose} variant="outline">
                {t.common.cancel}
              </Button>
              <Button disabled={saving || passcode.length === 0} onClick={() => void save()}>
                {g.savePasscode}
              </Button>
            </DialogFooter>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
