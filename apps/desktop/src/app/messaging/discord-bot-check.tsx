import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { ErrorBanner } from '@/components/ui/error-state'
import { checkDiscordBot } from '@/hermes'
import { useI18n } from '@/i18n'
import { openExternalLink } from '@/lib/external-link'
import { ExternalLink } from '@/lib/icons'
import type { DiscordBotCheckResponse, MessagingPlatformInfo } from '@/types/hermes'

export interface DiscordBotCheckProps {
  platform: MessagingPlatformInfo
  scopeProfile: string | undefined
}

export function DiscordBotCheck({ platform, scopeProfile }: DiscordBotCheckProps) {
  const { t } = useI18n()
  const q = t.messaging.discordCheck
  const [result, setResult] = useState<DiscordBotCheckResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const tokenSaved = platform.env_vars.some(field => field.key === 'DISCORD_BOT_TOKEN' && field.is_set)

  const check = async () => {
    setBusy(true)
    setError('')
    setResult(null)

    try {
      setResult(await checkDiscordBot(scopeProfile))
    } catch (checkError) {
      setError(String(checkError))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button disabled={busy || !tokenSaved} onClick={() => void check()} size="sm">
          {busy ? q.checking : result ? q.checkAgain : q.checkBot}
        </Button>
        {!tokenSaved && <p className="text-xs text-muted-foreground">{q.saveFirst}</p>}
      </div>
      {error && <ErrorBanner>{error}</ErrorBanner>}
      {result && (
        <div className="grid gap-3 text-sm">
          <p className="font-medium">{result.bot_name}</p>
          {result.message_content ? (
            <p className="text-muted-foreground">{q.intentOn}</p>
          ) : (
            <ErrorBanner>
              <p>{q.intentOff}</p>
              <Button onClick={() => openExternalLink(result.bot_settings_url)} size="sm" variant="secondary">
                <ExternalLink />
                {q.openSettings}
              </Button>
            </ErrorBanner>
          )}
          {!result.server_members && <p className="text-muted-foreground">{q.membersOptional}</p>}
          {result.server_count === 0 && <p className="text-muted-foreground">{q.noServers}</p>}
          <div>
            <Button onClick={() => openExternalLink(result.invite_url)} size="sm" variant="secondary">
              <ExternalLink />
              {q.addToServer}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
