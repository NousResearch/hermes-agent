import { useEffect, useState } from 'react'

import { StableText } from '@/components/chat/stable-text'
import { compactNumber } from '@/lib/format'
import type { UsageStats } from '@/types/hermes'

export function formatDuration(elapsedMs: number): string {
  const totalSeconds = Math.max(0, Math.floor(elapsedMs / 1000))
  const seconds = totalSeconds % 60
  const minutes = Math.floor(totalSeconds / 60) % 60
  const hours = Math.floor(totalSeconds / 3600)
  const ss = String(seconds).padStart(2, '0')
  const mm = String(minutes).padStart(2, '0')

  return hours > 0 ? `${hours}:${mm}:${ss}` : `${minutes}:${ss}`
}

export function compactPath(path: string, max = 44): string {
  const trimmed = path.trim()

  if (trimmed.length <= max) {
    return trimmed
  }

  const segments = trimmed.split('/').filter(Boolean)

  if (segments.length < 2) {
    return `…${trimmed.slice(-(max - 1))}`
  }

  const tail = segments.slice(-2).join('/')

  return tail.length + 2 >= max ? `…${tail.slice(-(max - 1))}` : `…/${tail}`
}

export function contextBar(percent: number | undefined, width = 10): string {
  const bounded = Math.max(0, Math.min(100, percent ?? 0))
  const filled = Math.round((bounded / 100) * width)

  return `${'█'.repeat(filled)}${'░'.repeat(width - filled)}`
}

export function usageContextLabel(usage: UsageStats): string {
  if (usage.context_max) {
    return `${compactNumber(usage.context_used ?? 0)}/${compactNumber(usage.context_max)}`
  }

  return usage.total > 0 ? `${compactNumber(usage.total)} tok` : ''
}

export function contextBarLabel(usage: UsageStats): string {
  if (!usage.context_max) {
    return ''
  }

  const pct = Math.max(0, Math.min(100, Math.round(usage.context_percent ?? 0)))

  return `[${contextBar(usage.context_percent)}] ${pct}%`
}

const fmtK = (n: number): string => {
  if (!Number.isFinite(n) || n < 0) {
    return '0'
  }

  if (n < 1000) {
    return String(Math.round(n))
  }

  if (n < 10_000) {
    return `${(n / 1000).toFixed(1).replace(/\.0$/, '')}k`
  }

  if (n < 1_000_000) {
    return `${Math.round(n / 1000)}k`
  }

  return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, '')}M`
}

/**
 * Claude-style cache segment. Returns '' when cache_read is missing/zero so
 * providers without cache hits never fabricate a percentage.
 * Full: `cache R=134k(96%) fresh=5k` · compact: `R=134k(96%)`
 */
export function formatCacheFresh(usage: UsageStats, compact = false): string {
  const cacheRead = Math.max(0, Number(usage.cache_read ?? 0) || 0)

  if (cacheRead <= 0) {
    return ''
  }

  const lastPrompt = Math.max(0, Number(usage.last_prompt ?? 0) || 0)
  const ctxUsed = Math.max(0, Number(usage.context_used ?? 0) || 0)
  const denom = lastPrompt > 0 ? lastPrompt : ctxUsed > 0 ? ctxUsed : cacheRead
  const hit = Math.max(0, Math.min(100, Math.round((cacheRead / denom) * 100)))
  const freshBase = lastPrompt > 0 ? lastPrompt : ctxUsed
  const fresh = freshBase > 0 ? Math.max(0, freshBase - cacheRead) : 0

  if (compact) {
    return `R=${fmtK(cacheRead)}(${hit}%)`
  }

  const pieces = [`cache R=${fmtK(cacheRead)}(${hit}%)`]

  if (freshBase > 0) {
    pieces.push(`fresh=${fmtK(fresh)}`)
  }

  return pieces.join(' ')
}

/**
 * Claude-like dollar amount; empty when cost is missing or non-positive.
 * Prefix with ~ when backend marks the figure as estimated (not provider invoice).
 */
export function formatStatusCost(usd?: number, status?: string): string {
  if (usd == null || !Number.isFinite(usd) || usd <= 0) {
    return ''
  }

  let body: string
  if (usd < 0.01) {
    body = '<$0.01'
  } else if (usd < 10) {
    body = `$${usd.toFixed(2)}`
  } else {
    body = `$${Number(usd.toFixed(3))}`
  }

  if (status === 'estimated') {
    return body.startsWith('<') ? `~${body}` : `~${body}`
  }
  return body
}

export function LiveDuration({ since }: { since: number | null | undefined }) {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    if (!since) {
      return
    }

    const tick = () => setNow(Date.now())
    tick()
    const timer = window.setInterval(tick, 1000)

    return () => window.clearInterval(timer)
  }, [since])

  if (!since) {
    return null
  }

  return <StableText>{formatDuration(now - since)}</StableText>
}
