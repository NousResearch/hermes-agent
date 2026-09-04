import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { notify, notifyError } from '@/store/notifications'
import {
  $remoteOverrideDialogProfile,
  closeRemoteOverrideDialog,
  refreshProfileRemoteOverrides,
  remoteHostLabel
} from '@/store/profile-remote-override'

// "Connect this profile to a remote host…" — the profile-rail affordance for
// the per-profile override the Electron main already honors
// (connection.json `profiles.<name>` → profileRemoteOverride). The renderer
// never touches connection.json: everything goes through the existing typed
// getConnectionConfig / applyConnectionConfig bridge with a `profile` scope,
// which encrypts the token via safeStorage (or the explicit plain-text opt-in
// on machines without an OS keychain).
//
// Safety beats requested in #91349:
//  - first-time connect shows a confirmation with a plain-language risk note;
//  - saving warns when the profile name collides with a v2 registry gateway
//    (two different stores keyed by the same name is how routing surprises
//    start);
//  - a keyring-less machine requires the explicit unencrypted-token opt-in.

interface LoadedScope {
  authMode: 'oauth' | 'token'
  hasOverride: boolean
  secureTokenStorage: boolean
  tokenPlainText: boolean
  tokenSet: boolean
  url: string
}

export function ProfileRemoteOverrideDialog({ profileNames }: { profileNames: string[] }) {
  const { t } = useI18n()
  const p = t.profiles.remoteOverride
  const profile = useStore($remoteOverrideDialogProfile)
  const open = profile !== null

  const [loaded, setLoaded] = useState<LoadedScope | null>(null)
  const [url, setUrl] = useState('')
  const [token, setToken] = useState('')
  const [allowPlainText, setAllowPlainText] = useState(false)
  const [confirming, setConfirming] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<null | string>(null)
  const [collision, setCollision] = useState<null | string>(null)
  const urlRef = useRef<HTMLInputElement>(null)

  // Load this profile's saved scope + the registry (for the name-collision
  // warning) each time the dialog opens. The registry read goes straight to
  // the bridge instead of the connections store so this dialog stays a leaf.
  useEffect(() => {
    if (!profile) {
      return
    }

    let cancelled = false
    setLoaded(null)
    setUrl('')
    setToken('')
    setAllowPlainText(false)
    setConfirming(false)
    setSaving(false)
    setError(null)
    setCollision(null)

    window.hermesDesktop
      ?.getConnectionConfig?.(profile)
      .then(config => {
        if (cancelled) {
          return
        }

        const hasOverride = (config.mode === 'remote' || config.mode === 'cloud') && Boolean(config.remoteUrl)
        setLoaded({
          authMode: config.remoteAuthMode === 'oauth' ? 'oauth' : 'token',
          hasOverride,
          secureTokenStorage: config.secureTokenStorage !== false,
          tokenPlainText: config.remoteTokenPlainText === true,
          tokenSet: config.remoteTokenSet === true,
          url: hasOverride ? config.remoteUrl : ''
        })
        setUrl(hasOverride ? config.remoteUrl : '')
        window.setTimeout(() => urlRef.current?.focus(), 0)
      })
      .catch(err => !cancelled && setError(err instanceof Error ? err.message : String(err)))

    window.hermesDesktop?.connections
      ?.list()
      .then(registry => {
        if (cancelled) {
          return
        }

        const lowered = profile.trim().toLowerCase()

        const match = registry.connections.find(
          connection => connection.id.toLowerCase() === lowered || connection.label.trim().toLowerCase() === lowered
        )

        setCollision(match ? match.label : null)
      })
      .catch(() => undefined)

    return () => void (cancelled = true)
  }, [profile])

  const close = () => {
    if (!saving) {
      closeRemoteOverrideDialog()
    }
  }

  const trimmedUrl = url.trim()
  const urlValid = /^https?:\/\/\S+$/i.test(trimmedUrl)
  const tokenReady = Boolean(token.trim()) || (loaded?.tokenSet === true && trimmedUrl === loaded.url)
  const canSubmit = Boolean(loaded) && urlValid && tokenReady && !saving
  // Writing a NEW token on a machine without an OS keychain stores it as
  // plain text on disk — that needs the explicit opt-in checked first.
  const needsPlainTextOptIn = Boolean(loaded && loaded.secureTokenStorage === false && token.trim() && !allowPlainText)

  const performSave = async () => {
    if (!profile) {
      return
    }

    setSaving(true)
    setError(null)

    try {
      await window.hermesDesktop.applyConnectionConfig({
        mode: 'remote',
        profile,
        remoteAuthMode: 'token',
        remoteToken: token.trim() || undefined,
        remoteUrl: trimmedUrl,
        ...(allowPlainText ? { allowPlainTextToken: true } : {})
      })

      notify({
        kind: 'success',
        title: p.savedTitle,
        message: p.savedMessage(profile, remoteHostLabel(trimmedUrl) || trimmedUrl)
      })
      await refreshProfileRemoteOverrides(profileNames)
      closeRemoteOverrideDialog()
    } catch (err) {
      setConfirming(false)
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSaving(false)
    }
  }

  const submit = () => {
    if (!canSubmit || needsPlainTextOptIn) {
      return
    }

    // First-time connect gets the one-time risk confirmation; editing an
    // existing override (token rotation, URL fix) skips straight to the save.
    if (!loaded?.hasOverride && !confirming) {
      setConfirming(true)

      return
    }

    void performSave()
  }

  const removeOverride = async () => {
    if (!profile) {
      return
    }

    setSaving(true)
    setError(null)

    try {
      await window.hermesDesktop.applyConnectionConfig({ mode: 'local', profile })
      notify({ kind: 'success', title: p.removedTitle, message: p.removedMessage(profile) })
      await refreshProfileRemoteOverrides(profileNames)
      closeRemoteOverrideDialog()
    } catch (err) {
      notifyError(err, p.removeFailed)
      setSaving(false)
    }
  }

  const host = remoteHostLabel(trimmedUrl) || trimmedUrl

  // Mounted permanently in the rail; render nothing until a profile's dialog
  // is actually requested (also keeps the closed tree from touching i18n).
  if (!open) {
    return null
  }

  return (
    <Dialog onOpenChange={next => !next && close()} open={open}>
      <DialogContent className="max-w-md">
        {confirming ? (
          <>
            <DialogHeader>
              <DialogTitle>{p.confirmTitle}</DialogTitle>
              <DialogDescription>{p.confirmNote(profile ?? '', host)}</DialogDescription>
            </DialogHeader>
            {error && <p className="text-xs text-destructive">{error}</p>}
            <DialogFooter>
              <Button disabled={saving} onClick={() => setConfirming(false)} type="button" variant="ghost">
                {p.confirmBack}
              </Button>
              <Button disabled={saving} onClick={() => void performSave()} type="button">
                {saving ? p.connecting : p.connect}
              </Button>
            </DialogFooter>
          </>
        ) : (
          <>
            <DialogHeader>
              <DialogTitle>{p.title(profile ?? '')}</DialogTitle>
              <DialogDescription>{p.description}</DialogDescription>
            </DialogHeader>

            <div className="flex flex-col gap-3">
              <label className="flex flex-col gap-1 text-xs">
                <span className="text-(--ui-text-secondary)">{p.urlLabel}</span>
                <Input
                  autoCorrect="off"
                  onChange={event => setUrl(event.target.value)}
                  onKeyDown={event => event.key === 'Enter' && submit()}
                  placeholder={p.urlPlaceholder}
                  ref={urlRef}
                  spellCheck={false}
                  type="url"
                  value={url}
                />
                {trimmedUrl && !urlValid && <span className="text-destructive">{p.urlInvalid}</span>}
              </label>

              <label className="flex flex-col gap-1 text-xs">
                <span className="text-(--ui-text-secondary)">{p.tokenLabel}</span>
                <Input
                  autoComplete="off"
                  onChange={event => setToken(event.target.value)}
                  onKeyDown={event => event.key === 'Enter' && submit()}
                  placeholder={p.tokenPlaceholder}
                  type="password"
                  value={token}
                />
                {loaded?.tokenSet && !token.trim() && (
                  <span className="text-(--ui-text-tertiary)">{p.tokenSavedHint}</span>
                )}
              </label>

              {loaded && loaded.secureTokenStorage === false && Boolean(token.trim()) && (
                <label className="flex items-start gap-2 text-xs text-(--ui-text-secondary)">
                  <Checkbox
                    checked={allowPlainText}
                    className="mt-0.5"
                    onCheckedChange={checked => setAllowPlainText(checked === true)}
                  />
                  <span>{p.plainTextOptIn}</span>
                </label>
              )}

              {collision && <p className="text-xs text-(--ui-text-secondary)">{p.collisionWarning(collision)}</p>}
              {error && <p className="text-xs text-destructive">{error}</p>}
            </div>

            <DialogFooter>
              {loaded?.hasOverride && (
                <Button
                  className="mr-auto"
                  disabled={saving}
                  onClick={() => void removeOverride()}
                  type="button"
                  variant="ghost"
                >
                  {p.disconnect}
                </Button>
              )}
              <Button disabled={saving} onClick={close} type="button" variant="ghost">
                {t.common.cancel}
              </Button>
              <Button disabled={!canSubmit || needsPlainTextOptIn} onClick={submit} type="button">
                {saving ? p.connecting : p.connect}
              </Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}
