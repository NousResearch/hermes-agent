import { useState } from 'react'

import { passwordLogin, type ProbeResult } from '~bridge/auth'

import { Brand, Button, Card, ErrorNote, Field, Screen } from '../ui'

/** Step 2 (gated gateways): username/password → session cookies. */
export function LoginScreen({
  probe,
  onBack,
  onLoggedIn,
}: {
  probe: ProbeResult
  onBack: () => void
  onLoggedIn: (provider: string) => void
}) {
  // Pick the password-capable provider; default to "basic" if the gateway didn't
  // enumerate any (a generic password gateway stays generic on purpose). If it
  // DID enumerate providers but none support password, it's OAuth-only — the
  // /auth/password-login endpoint 404s there, so we can't sign in via password.
  const passwordProvider = probe.providers.find((p) => p.supportsPassword)
  const oauthOnly = probe.providers.length > 0 && !passwordProvider
  const provider = passwordProvider?.name ?? 'basic'

  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit() {
    if (busy) return
    setBusy(true)
    setError(null)
    try {
      await passwordLogin(probe.baseUrl, { provider, username, password })
      onLoggedIn(provider)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  // OAuth-only gateway: don't submit a password we know will 404. The redirect
  // flow isn't implemented on mobile yet, so explain and send them back.
  if (oauthOnly) {
    const names = probe.providers.map((p) => p.displayName).join(', ')
    return (
      <Screen>
        <Brand subtitle={hostLabel(probe.baseUrl)} />
        <Card>
          <p className="text-sm leading-relaxed text-muted-foreground">
            This gateway signs in with {names || 'OAuth'}, which the mobile app
            doesn&apos;t support yet. Point it at a gateway with password login, or
            sign in from the desktop app.
          </p>
          <Button variant="ghost" onClick={onBack}>
            Use a different gateway
          </Button>
        </Card>
      </Screen>
    )
  }

  return (
    <Screen>
      <Brand subtitle={hostLabel(probe.baseUrl)} />
      <Card>
        <Field
          label="Username"
          autoCapitalize="none"
          autoCorrect="off"
          spellCheck={false}
          autoComplete="username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <Field
          label="Password"
          type="password"
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && submit()}
        />
        <ErrorNote>{error}</ErrorNote>
        <Button busy={busy} onClick={submit}>
          Sign in
        </Button>
        <Button variant="ghost" onClick={onBack} disabled={busy}>
          Use a different gateway
        </Button>
      </Card>
    </Screen>
  )
}

function hostLabel(baseUrl: string): string {
  try {
    return new URL(baseUrl).host
  } catch {
    return baseUrl
  }
}
