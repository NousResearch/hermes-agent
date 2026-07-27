import { useCallback, useEffect, useRef, useState } from 'react'

import { BrandMark } from '@/components/brand-mark'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { DesktopConnectionProbeResult } from '@/global'
import { useI18n } from '@/i18n'
import { deriveRemoteAuthProviderShape } from '@/lib/desktop-remote-auth'
import { AlertCircle, Check, Loader2, LogIn } from '@/lib/icons'

type AuthMode = 'oauth' | 'token'
type ProbeStatus = 'idle' | 'probing' | 'done' | 'error'

interface FirstRunRemoteFormProps {
  onBack: () => void
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err || 'Unknown error')
}

export function FirstRunRemoteForm({ onBack }: FirstRunRemoteFormProps) {
  const { t } = useI18n()
  const copy = t.install
  const [remoteUrl, setRemoteUrl] = useState('')
  const [remoteToken, setRemoteToken] = useState('')
  const [probeStatus, setProbeStatus] = useState<ProbeStatus>('idle')
  const [probe, setProbe] = useState<DesktopConnectionProbeResult | null>(null)
  const [oauthConnected, setOauthConnected] = useState(false)
  const [signingIn, setSigningIn] = useState(false)
  const [testing, setTesting] = useState(false)
  const [applying, setApplying] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [lastTestedPayloadKey, setLastTestedPayloadKey] = useState<string | null>(null)
  const probeSeq = useRef(0)
  const testSeq = useRef(0)

  const trimmedUrl = remoteUrl.trim()

  const invalidateTest = useCallback(() => {
    testSeq.current += 1
    setTesting(false)
    setError(null)
    setSuccess(null)
    setLastTestedPayloadKey(null)
  }, [])

  useEffect(() => {
    const seq = ++probeSeq.current

    if (!trimmedUrl || !/^https?:\/\//i.test(trimmedUrl)) {
      setProbeStatus('idle')
      setProbe(null)
      setOauthConnected(false)

      return
    }

    const desktop = window.hermesDesktop

    if (!desktop?.probeConnectionConfig) {
      return
    }

    setProbeStatus('probing')

    const timer = window.setTimeout(() => {
      desktop
        .probeConnectionConfig(trimmedUrl)
        .then(result => {
          if (seq !== probeSeq.current) {
            return
          }

          invalidateTest()
          setProbe(result)
          setProbeStatus(result.reachable ? 'done' : 'error')

          if (result.reachable && result.authMode !== 'oauth') {
            setOauthConnected(false)
          }
        })
        .catch(err => {
          if (seq !== probeSeq.current) {
            return
          }

          setProbe(null)
          setProbeStatus('error')
          setError(errorMessage(err))
        })
    }, 500)

    return () => window.clearTimeout(timer)
  }, [invalidateTest, trimmedUrl])

  const authMode: AuthMode = probeStatus === 'done' && probe?.authMode === 'oauth' ? 'oauth' : 'token'
  const authResolved = probeStatus === 'done' && probe?.authMode !== 'unknown'
  const authProviderShape = deriveRemoteAuthProviderShape(probe?.providers, copy.identityProvider)
  const { isPassword: isPasswordProvider, providerLabel } = authProviderShape
  const canRetryProbe = Boolean(trimmedUrl && probeStatus === 'error')

  const canTest = Boolean(
    trimmedUrl && (canRetryProbe || (authResolved && (authMode === 'oauth' ? oauthConnected : remoteToken.trim())))
  )

  const payload = () => ({
    mode: 'remote' as const,
    remoteAuthMode: authMode,
    remoteToken: authMode === 'token' ? remoteToken.trim() || undefined : undefined,
    remoteUrl: trimmedUrl
  })

  const currentPayloadKey = JSON.stringify(payload())
  const payloadKeyRef = useRef(currentPayloadKey)
  payloadKeyRef.current = currentPayloadKey
  const canApply = lastTestedPayloadKey === currentPayloadKey

  const signIn = async () => {
    if (!trimmedUrl) {
      setError(copy.enterUrlFirst)

      return
    }

    setSigningIn(true)
    setError(null)

    try {
      // Unlike Settings, first-run intentionally does not pre-save remote mode:
      // backing out must still allow local install without leaving a remote
      // connection selected. The login IPC accepts the raw URL and stores only
      // its OAuth cookies; config is persisted once the user applies.
      const result = await window.hermesDesktop.oauthLoginConnectionConfig(trimmedUrl)
      invalidateTest()
      setOauthConnected(Boolean(result.connected))

      if (!result.connected) {
        setError(copy.signInIncomplete)
      }
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setSigningIn(false)
    }
  }

  const testRemote = async () => {
    if (!canTest) {
      setError(authMode === 'oauth' ? copy.incompleteSignInTest : copy.incompleteTokenTest)

      return
    }

    const seq = ++testSeq.current
    const testedPayload = payload()
    const testedPayloadKey = JSON.stringify(testedPayload)

    setTesting(true)
    setError(null)
    setSuccess(null)
    setLastTestedPayloadKey(null)

    try {
      if (!authResolved) {
        const result = await window.hermesDesktop.probeConnectionConfig(trimmedUrl)

        if (seq !== testSeq.current || testedPayloadKey !== payloadKeyRef.current) {
          return
        }

        setProbe(result)
        setProbeStatus(result.reachable ? 'done' : 'error')
        setError(result.reachable && result.authMode !== 'unknown' ? null : result.error || copy.probeError)

        return
      }

      const result = await window.hermesDesktop.testConnectionConfig(testedPayload)

      if (seq !== testSeq.current || testedPayloadKey !== payloadKeyRef.current) {
        return
      }

      setSuccess(copy.testSucceeded(result.baseUrl || trimmedUrl, result.version ?? undefined))
      setLastTestedPayloadKey(testedPayloadKey)
    } catch (err) {
      if (seq === testSeq.current && testedPayloadKey === payloadKeyRef.current) {
        setError(errorMessage(err))
      }
    } finally {
      if (seq === testSeq.current) {
        setTesting(false)
      }
    }
  }

  const applyRemote = async () => {
    if (!canApply) {
      return
    }

    const testedPayload = payload()

    setApplying(true)
    setError(null)
    let applied = false

    try {
      await window.hermesDesktop.applyConnectionConfig(testedPayload)
      applied = true
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setApplying(false)
    }

    if (applied) {
      onBack()
    }
  }

  return (
    <div className="fixed inset-0 z-[1400] flex items-center justify-center bg-background/90 p-4 backdrop-blur-md">
      <div className="flex w-full max-w-xl flex-col rounded-xl border border-(--stroke-nous) bg-card p-8 shadow-nous">
        <div className="flex items-start gap-4">
          <BrandMark className="size-11 shrink-0" />
          <div className="min-w-0">
            <h2 className="text-xl font-semibold tracking-tight">{copy.remoteSetupTitle}</h2>
            <p className="mt-1.5 text-sm text-muted-foreground">{copy.remoteSetupDesc}</p>
          </div>
        </div>

        <div className="mt-6 grid gap-4">
          <label className="grid gap-1.5">
            <span className="text-xs font-medium text-muted-foreground">{copy.remoteUrlTitle}</span>
            <Input
              autoComplete="url"
              disabled={applying}
              onChange={event => {
                invalidateTest()
                setRemoteUrl(event.target.value)
              }}
              placeholder={copy.remoteUrlPlaceholder}
              value={remoteUrl}
            />
            <span className="text-xs text-muted-foreground">{copy.remoteUrlDesc}</span>
          </label>

          {probeStatus === 'probing' ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
              {copy.probing}
            </div>
          ) : null}

          {probeStatus === 'error' ? (
            <div className="flex items-start gap-2 text-sm text-destructive">
              <AlertCircle className="mt-0.5 size-4 shrink-0" />
              <span>{probe?.error || copy.probeError}</span>
            </div>
          ) : null}

          {authResolved && authMode === 'oauth' ? (
            <div className="rounded-md border border-(--ui-stroke-tertiary) p-3">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="text-sm font-medium">{copy.authTitle}</div>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {oauthConnected ? copy.authSignedIn : copy.authNeedsOauth(providerLabel)}
                  </p>
                </div>
                {oauthConnected ? (
                  <div className="flex items-center gap-1.5 text-sm text-primary">
                    <Check className="size-4" />
                    {copy.connected}
                  </div>
                ) : (
                  <Button disabled={signingIn || applying} onClick={() => void signIn()} size="sm">
                    {signingIn ? <Loader2 className="size-4 animate-spin" /> : <LogIn className="size-4" />}
                    {isPasswordProvider ? copy.signIn : copy.signInWith(providerLabel)}
                  </Button>
                )}
              </div>
            </div>
          ) : null}

          {authResolved && authMode === 'token' ? (
            <label className="grid gap-1.5">
              <span className="text-xs font-medium text-muted-foreground">{copy.tokenTitle}</span>
              <Input
                autoComplete="off"
                disabled={applying}
                onChange={event => {
                  invalidateTest()
                  setRemoteToken(event.target.value)
                }}
                placeholder={copy.pasteSessionToken}
                type="password"
                value={remoteToken}
              />
              <span className="text-xs text-muted-foreground">{copy.tokenDesc}</span>
            </label>
          ) : null}

          {error ? (
            <div className="flex items-start gap-2 text-sm text-destructive">
              <AlertCircle className="mt-0.5 size-4 shrink-0" />
              <span>{error}</span>
            </div>
          ) : null}

          {success ? (
            <div className="flex items-center gap-2 text-sm text-primary">
              <Check className="size-4" />
              <span>{success}</span>
            </div>
          ) : null}
        </div>

        <div className="mt-7 flex flex-wrap items-center justify-between gap-3">
          <Button disabled={applying} onClick={onBack} size="sm" variant="ghost">
            {copy.backToSetup}
          </Button>
          <div className="flex items-center gap-2">
            <Button
              disabled={testing || applying || !canTest}
              onClick={() => void testRemote()}
              size="sm"
              variant="secondary"
            >
              {testing ? <Loader2 className="size-4 animate-spin" /> : null}
              {copy.testConnection}
            </Button>
            <Button disabled={applying || !canApply} onClick={() => void applyRemote()} size="sm">
              {applying ? <Loader2 className="size-4 animate-spin" /> : null}
              {copy.applyRemote}
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
