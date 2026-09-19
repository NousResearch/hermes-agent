import { useEffect, useRef, useState } from 'react'

import {
  connectDiscoveredMcp,
  discoverMcpServers,
  type McpCandidate,
  type McpDiscoveryResult
} from '@/api/mcp-discovery'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { type ProfileScope, profileScopeKey } from '@/hermes'
import { useI18n } from '@/i18n'

interface McpDiscoveryProps {
  profile: ProfileScope
  configuredNames: string[]
  disabled?: boolean
  onConnected: () => Promise<void>
}

/** A scope change unmounts in-flight UI work; the request itself keeps its original destination. */
export function McpDiscovery(props: McpDiscoveryProps) {
  return <DiscoveryPanel key={profileScopeKey(props.profile)} {...props} />
}

function DiscoveryPanel({ profile, configuredNames, disabled, onConnected }: McpDiscoveryProps) {
  const { t } = useI18n()
  const m = t.settings.mcp
  const [result, setResult] = useState<McpDiscoveryResult | null>(null)
  const [searching, setSearching] = useState(false)
  const [connecting, setConnecting] = useState<string | null>(null)
  const [saved, setSaved] = useState<string[]>([])
  const [error, setError] = useState('')
  const alive = useRef(true)
  const busy = useRef(false)
  const target = typeof profile === 'string' ? profile : profile?.profile || 'default'

  // eslint-disable-next-line no-restricted-syntax -- mounted lifetime flag, not mirrored reactive state
  useEffect(() => {
    alive.current = true

    return () => {
      alive.current = false
    }
  }, [])

  const refresh = async () => {
    if (busy.current || disabled) {
      return
    }

    busy.current = true
    setSearching(true)
    setError('')

    try {
      const next = await discoverMcpServers(profile)

      if (alive.current) {
        setResult(next)
        setSaved([])
      }
    } catch (err) {
      if (alive.current) {
        setError(String(err instanceof Error ? err.message : err))
      }
    } finally {
      busy.current = false

      if (alive.current) {
        setSearching(false)
      }
    }
  }

  const connect = async (candidate: McpCandidate) => {
    if (busy.current || disabled || !candidate.connectable) {
      return
    }

    busy.current = true
    setConnecting(candidate.id)
    setError('')

    try {
      const response = await connectDiscoveredMcp(candidate.id, profile)

      if (!response.ok) {
        throw new Error(m.saveFailed)
      }

      if (!alive.current) {
        return
      }

      // This records configuration, not server health. The configured-server
      // list owns the subsequent handshake/status and its ordinary off switch.
      setSaved(previous => [...previous, response.name])
      await onConnected()
    } catch (err) {
      if (alive.current) {
        setError(String(err instanceof Error ? err.message : err))
      }
    } finally {
      busy.current = false

      if (alive.current) {
        setConnecting(null)
      }
    }
  }

  const candidates =
    result?.candidates.filter(
      candidate => !configuredNames.includes(candidate.name) && !saved.includes(candidate.name)
    ) ?? []

  return (
    <section aria-label={m.discoveryTitle} className="mt-3 border-t border-(--ui-stroke-quaternary) pt-2">
      <div className="flex items-center justify-between gap-2 px-2">
        <span className="text-[0.72rem] font-medium text-(--ui-text-tertiary)">{m.discoveryTitle}</span>
        <Button
          disabled={disabled || searching || connecting !== null}
          onClick={() => void refresh()}
          size="xs"
          variant="ghost"
        >
          {searching ? m.discoverySearching : result ? t.common.refresh : m.discoveryFind}
        </Button>
      </div>
      <p className="px-2 py-1 text-[0.68rem] text-muted-foreground">{m.discoveryHint}</p>
      <p className="px-2 pb-1 text-[0.68rem] font-medium">{m.discoveryTarget(target)}</p>
      {error && (
        <p className="px-2 py-1 text-xs text-destructive" role="alert">
          {error}
        </p>
      )}
      {result?.warnings.map((warning, index) => (
        <p className="px-2 py-1 text-[0.68rem] text-amber-600" key={index}>
          {warning}
        </p>
      ))}
      {result && !searching && candidates.length === 0 && (
        <p className="px-2 py-2 text-xs text-muted-foreground">{m.discoveryEmpty}</p>
      )}
      {candidates.map(candidate => (
        <div className="flex items-center gap-3 rounded px-2 py-2 hover:bg-accent/40" key={candidate.id}>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 text-xs font-medium">
              <span className="truncate">{candidate.name}</span>
              <span className="text-[0.65rem] font-normal text-muted-foreground">{candidate.transport}</span>
            </div>
            <p className="break-words text-[0.68rem] text-muted-foreground">{candidate.source}</p>
            <p className="break-words text-[0.68rem] text-muted-foreground">{candidate.summary}</p>
            {candidate.reason && <p className="text-[0.68rem] text-amber-600">{candidate.reason}</p>}
          </div>
          <Switch
            aria-label={`${m.discoveryTarget(target)}: ${candidate.name} (${candidate.source})`}
            checked={false}
            disabled={disabled || searching || connecting !== null || !candidate.connectable}
            onCheckedChange={checked => {
              if (checked) {
                void connect(candidate)
              }
            }}
            size="xs"
          />
        </div>
      ))}
    </section>
  )
}
