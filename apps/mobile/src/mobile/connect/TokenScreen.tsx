import { useState } from 'react'

import type { ProbeResult } from '~bridge/auth'

import { Brand, Button, Card, ErrorNote, Field, Screen } from '../ui'

/**
 * Token-auth gateways (auth_required:false but not open) need a static session
 * token — the gateway rejects an empty `?token=`/`X-Hermes-Session-Token`, and
 * the desktop likewise refuses a token-mode config with no saved token. So we
 * collect one here before committing the target, rather than connecting blind.
 */
export function TokenScreen({
  probe,
  onBack,
  onToken,
}: {
  probe: ProbeResult
  onBack: () => void
  onToken: (token: string) => void
}) {
  const [token, setToken] = useState('')
  const [error, setError] = useState<string | null>(null)

  function submit() {
    const value = token.trim()
    if (!value) {
      setError('A session token is required for this gateway.')
      return
    }
    onToken(value)
  }

  return (
    <Screen>
      <Brand subtitle={hostLabel(probe.baseUrl)} />
      <Card>
        <Field
          label="Session token"
          type="password"
          autoCapitalize="none"
          autoCorrect="off"
          spellCheck={false}
          value={token}
          onChange={(e) => setToken(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && submit()}
        />
        <ErrorNote>{error}</ErrorNote>
        <Button onClick={submit}>Connect</Button>
        <Button variant="ghost" onClick={onBack}>
          Use a different gateway
        </Button>
      </Card>
      <p className="px-1 text-center text-xs text-muted-foreground/80">
        This gateway uses token auth. Paste the same session token your desktop
        uses under Settings &rarr; Gateway.
      </p>
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
