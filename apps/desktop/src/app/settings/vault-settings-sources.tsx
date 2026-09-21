import { useMutation } from '@tanstack/react-query'
import { useCallback, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { KeyRound, Lock } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'

import { ListRow, Pill, SectionHeading } from './primitives'
import type { VaultData, VaultSource, VaultSourceName } from './vault-settings-data'

interface VaultSourcesProps {
  vault: VaultData
  standalone?: boolean
}

export function VaultSources({ vault, standalone = false }: VaultSourcesProps) {
  const { t } = useI18n()
  const v = t.settings.vault
  const { externalSources, requestGateway, invalidateVault } = vault
  const [unlockTarget, setUnlockTarget] = useState<null | VaultSource>(null)
  const [masterPassword, setMasterPassword] = useState('')
  const [unlockError, setUnlockError] = useState<null | string>(null)
  // The master password is never retained in React Query mutation variables.
  const pendingMasterPassword = useRef('')

  const setSourceEnabled = useMutation({
    mutationFn: ({ name, enabled }: { name: VaultSourceName; enabled: boolean }) =>
      requestGateway<{ enabled: boolean }>('vault.source.set', { name, enabled }),
    onSuccess: invalidateVault,
    onError: err => notifyError(err, v.sources.toggleFailed)
  })

  const lockSource = useMutation({
    mutationFn: (name: VaultSourceName) => requestGateway<{ locked: boolean }>('vault.lock', { name }),
    onSuccess: invalidateVault
  })

  // The master password lives only in this dialog's state; it is cleared the moment the request
  // returns (success or failure) and never touches a store or the transcript.
  const closeUnlock = useCallback(() => {
    setUnlockTarget(null)
    setMasterPassword('')
    setUnlockError(null)
  }, [])

  const unlockSource = useMutation({
    mutationFn: ({ name }: { name: VaultSourceName }) => {
      const password = pendingMasterPassword.current
      pendingMasterPassword.current = ''

      return requestGateway<{ unlocked: boolean }>('vault.unlock', { name, password })
    },
    onSuccess: (_result, { name }) => {
      triggerHaptic('submit')
      const source = externalSources.find(s => s.name === name)
      notify({ kind: 'success', message: v.sources.unlocked(source?.display_name ?? name) })
      closeUnlock()
      invalidateVault()
    },
    onError: err => {
      setMasterPassword('')
      setUnlockError(err instanceof Error ? err.message : String(err))
    }
  })

  return (
    <>
      {/* Password managers */}
      <div className={standalone ? undefined : 'mt-6'}>
        <SectionHeading icon={KeyRound} page title={v.sources.title} />
      </div>
      <p className="mb-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {v.sources.blurb}
      </p>
      {externalSources.map(source => (
        <ListRow
          action={
            <span className="flex items-center justify-end gap-2">
              {source.enabled &&
                (source.unlocked ? (
                  <Button
                    className="gap-1.5"
                    disabled={lockSource.isPending}
                    onClick={() => lockSource.mutate(source.name)}
                    size="sm"
                    type="button"
                    variant="ghost"
                  >
                    <Lock className="size-3.5" />
                    {v.sources.lock}
                  </Button>
                ) : (
                  <Button
                    className="gap-1.5"
                    onClick={() => setUnlockTarget(source)}
                    size="sm"
                    type="button"
                    variant="outline"
                  >
                    <KeyRound className="size-3.5" />
                    {v.sources.unlock}
                  </Button>
                ))}
              {source.installed && (
                <Switch
                  aria-label={source.display_name}
                  checked={source.enabled}
                  disabled={setSourceEnabled.isPending}
                  onCheckedChange={enabled => {
                    triggerHaptic('selection')
                    setSourceEnabled.mutate({ name: source.name, enabled })
                  }}
                />
              )}
            </span>
          }
          description={
            !source.installed
              ? v.sources.notInstalled(source.display_name)
              : source.enabled
                ? source.unlocked
                  ? v.sources.unlockedDesc
                  : v.sources.lockedDesc
                : v.sources.disabledDesc
          }
          key={source.name}
          title={
            <span className="flex items-center gap-2">
              <span>{source.display_name}</span>
              <Pill tone={source.enabled && source.unlocked ? 'primary' : 'muted'}>
                {!source.installed
                  ? v.sources.statusNotDetected
                  : !source.enabled
                    ? v.sources.statusOff
                    : source.unlocked
                      ? v.sources.statusUnlocked
                      : v.sources.statusLocked}
              </Pill>
            </span>
          }
        />
      ))}

      {/* Unlock dialog */}
      <Dialog onOpenChange={open => !open && closeUnlock()} open={unlockTarget !== null}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle icon={KeyRound}>{v.sources.unlockTitle(unlockTarget?.display_name ?? '')}</DialogTitle>
            <DialogDescription>{v.sources.unlockDescription}</DialogDescription>
          </DialogHeader>
          <form
            className="grid gap-3"
            onSubmit={e => {
              e.preventDefault()

              if (unlockTarget && masterPassword) {
                pendingMasterPassword.current = masterPassword
                setMasterPassword('')
                unlockSource.mutate({ name: unlockTarget.name })
              }
            }}
          >
            <Input
              autoComplete="current-password"
              autoFocus
              disabled={unlockSource.isPending}
              onChange={e => setMasterPassword(e.target.value)}
              placeholder={v.sources.masterPasswordPlaceholder}
              type="password"
              value={masterPassword}
            />
            {unlockError && <p className="text-xs text-destructive">{unlockError}</p>}
            <DialogFooter>
              <Button onClick={closeUnlock} type="button" variant="ghost">
                {t.common.cancel}
              </Button>
              <Button disabled={unlockSource.isPending || !masterPassword} type="submit">
                {unlockSource.isPending ? v.sources.unlocking : v.sources.unlock}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </>
  )
}
