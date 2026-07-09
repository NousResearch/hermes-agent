import { useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'

import { Field } from './bits'

/**
 * Native IX Agency sign-in — the same email-OTP flow as the admin portal,
 * driven entirely over IPC (main process talks to /api/auth/otp/* through
 * the portal session partition). No webview: this form IS the login.
 */

export function LoginPane({ detail, onSignedIn }: { detail?: string; onSignedIn: () => void }) {
  const bridge = window.hermesDesktop?.ixAgency

  const [step, setStep] = useState<'code' | 'email'>('email')
  const [email, setEmail] = useState('')
  const [code, setCode] = useState('')
  const [challenge, setChallenge] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<null | string>(null)
  const [resendAt, setResendAt] = useState(0)
  const [now, setNow] = useState(() => Date.now())

  const codeRef = useRef<HTMLInputElement | null>(null)

  // Tick the resend cooldown display.
  useEffect(() => {
    if (resendAt <= Date.now()) {
      return
    }

    const timer = setInterval(() => setNow(Date.now()), 1000)

    return () => clearInterval(timer)
  }, [resendAt])

  const sendCode = useCallback(async () => {
    if (!bridge?.authSendOtp || busy) {
      return
    }

    setBusy(true)
    setError(null)

    try {
      const result = await bridge.authSendOtp(email)

      setChallenge(result.challenge)
      setStep('code')
      setCode(result.devCode ?? '')
      setResendAt(Date.now() + 30_000)
      setTimeout(() => codeRef.current?.focus(), 50)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }, [bridge, busy, email])

  const verify = useCallback(async () => {
    if (!bridge?.authVerifyOtp || busy) {
      return
    }

    setBusy(true)
    setError(null)

    try {
      const result = await bridge.authVerifyOtp({ challenge, code, email })

      if (result.authenticated) {
        onSignedIn()
      } else {
        setError(result.detail || 'Signed in, but the session probe still reports signed out.')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }, [bridge, busy, challenge, code, email, onSignedIn])

  if (!bridge?.authSendOtp) {
    return (
      <div className="mx-auto max-w-2xl px-5 py-6 text-xs text-muted-foreground">
        The native sign-in IPC is unavailable in this build.
      </div>
    )
  }

  const resendWait = Math.max(0, Math.ceil((resendAt - now) / 1000))

  return (
    <div className="flex h-full items-center justify-center px-6">
      <div className="w-full max-w-sm space-y-4">
        <div className="space-y-1 text-center">
          <Codicon className="text-muted-foreground/60" name="shield" size="1.5rem" />
          <h2 className="text-sm font-semibold">Sign in to IX Agency</h2>
          <p className="text-xs text-muted-foreground">
            {step === 'email'
              ? 'Enter your admin email — we email you a 6-digit code. Nothing here needs the VPN.'
              : `Code sent to ${email}. It expires in 5 minutes.`}
          </p>
          {detail && step === 'email' && <p className="text-[0.68rem] text-muted-foreground/60">{detail}</p>}
        </div>

        {step === 'email' ? (
          <form
            className="space-y-3"
            onSubmit={event => {
              event.preventDefault()
              void sendCode()
            }}
          >
            <Field label="Admin email">
              <Input
                autoComplete="email"
                autoFocus
                onChange={event => setEmail(event.target.value)}
                placeholder="you@intelli-verse-x.ai"
                type="email"
                value={email}
              />
            </Field>
            <Button className="w-full" disabled={busy || !email.trim()} type="submit">
              {busy ? <Codicon name="loading~spin" size="0.8125rem" /> : <Codicon name="mail" size="0.8125rem" />}
              Email me a code
            </Button>
          </form>
        ) : (
          <form
            className="space-y-3"
            onSubmit={event => {
              event.preventDefault()
              void verify()
            }}
          >
            <Field label="6-digit code">
              <Input
                autoComplete="one-time-code"
                className="text-center font-mono text-base tracking-[0.4em]"
                inputMode="numeric"
                maxLength={6}
                onChange={event => setCode(event.target.value.replace(/\D/g, ''))}
                placeholder="••••••"
                ref={codeRef}
                value={code}
              />
            </Field>
            <Button className="w-full" disabled={busy || code.length !== 6} type="submit">
              {busy ? <Codicon name="loading~spin" size="0.8125rem" /> : <Codicon name="unlock" size="0.8125rem" />}
              Sign in
            </Button>
            <div className="flex items-center justify-between text-[0.68rem] text-muted-foreground">
              <button className="hover:text-foreground" onClick={() => setStep('email')} type="button">
                Use a different email
              </button>
              <button
                className="hover:text-foreground disabled:opacity-50"
                disabled={busy || resendWait > 0}
                onClick={() => void sendCode()}
                type="button"
              >
                {resendWait > 0 ? `Resend in ${resendWait}s` : 'Resend code'}
              </button>
            </div>
          </form>
        )}

        {error && (
          <p className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-600 dark:text-red-400">
            {error}
          </p>
        )}
      </div>
    </div>
  )
}
