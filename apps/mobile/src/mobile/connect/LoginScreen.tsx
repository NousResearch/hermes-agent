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
  // Pick the password-capable provider; default to "basic" if the gateway
  // didn't enumerate one (it stays generic on purpose).
  const provider = probe.providers.find((p) => p.supportsPassword)?.name ?? 'basic'

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
