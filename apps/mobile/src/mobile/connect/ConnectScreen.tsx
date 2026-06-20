import { useState } from 'react'

import { probeGateway, type ProbeResult } from '~bridge/auth'

import { Brand, Button, Card, ErrorNote, Field, Screen } from '../ui'

/** Step 1: enter the gateway URL and probe it. */
export function ConnectScreen({
  initialUrl = '',
  onResult,
}: {
  initialUrl?: string
  onResult: (probe: ProbeResult) => void
}) {
  const [url, setUrl] = useState(initialUrl)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit() {
    if (!url.trim() || busy) return
    setBusy(true)
    setError(null)
    try {
      const probe = await probeGateway(url)
      if (!probe.reachable) {
        setError(probe.error ?? 'Could not reach that gateway.')
        return
      }
      onResult(probe)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Screen>
      <Brand subtitle="Connect to your gateway" />
      <Card>
        <Field
          label="Gateway URL"
          placeholder="http://100.x.y.z:9119"
          inputMode="url"
          autoCapitalize="none"
          autoCorrect="off"
          spellCheck={false}
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && submit()}
        />
        <ErrorNote>{error}</ErrorNote>
        <Button busy={busy} onClick={submit}>
          Connect
        </Button>
      </Card>
      <p className="px-1 text-center text-xs text-muted-foreground/80">
        Your Hermes agent runs on the server. This app is just the screen — point it at the same
        gateway your desktop uses (Tailscale IP or LAN host).
      </p>
    </Screen>
  )
}
