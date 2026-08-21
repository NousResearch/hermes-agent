import { compactNumber, ErrorState, host, Loader } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

interface UsageSummary {
  total_sessions: number
  token_sessions: number
  cost_sessions: number
  total_estimated_cost_usd: number
  total_input_tokens: number
  total_output_tokens: number
  total_cache_read_tokens: number
  total_cache_write_tokens: number
  most_expensive_session_usd: number
  cheapest_session_usd: number
}

const EMPTY_SUMMARY: UsageSummary = {
  total_sessions: 0,
  token_sessions: 0,
  cost_sessions: 0,
  total_estimated_cost_usd: 0,
  total_input_tokens: 0,
  total_output_tokens: 0,
  total_cache_read_tokens: 0,
  total_cache_write_tokens: 0,
  most_expensive_session_usd: 0,
  cheapest_session_usd: 0
}

// Micro-costs (e.g. $0.000056) must not collapse to "$0.00". Use more
// decimals when the value is tiny, so small positive costs stay visible.
const formatUsd = (value: number) => {
  const v = value || 0
  const fractionDigits = v > 0 && v < 0.01 ? 6 : 2
  return new Intl.NumberFormat('en-US', { currency: 'USD', style: 'currency', minimumFractionDigits: fractionDigits, maximumFractionDigits: fractionDigits }).format(v)
}

function SummaryCard({ label, value }: { label: string; value: string }) {
  return (
    <article className="rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-4">
      <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">{label}</p>
      <p className="mt-2 text-xl font-semibold tabular-nums text-foreground">{value}</p>
    </article>
  )
}

export function UsagePage() {
  const [summary, setSummary] = useState<UsageSummary | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    void host
      .request<UsageSummary>('usage.summary')
      .then(data => {
        if (!cancelled) {
          setSummary({ ...EMPTY_SUMMARY, ...data })
        }
      })
      .catch(reason => {
        if (!cancelled) {
          setError(reason instanceof Error ? reason.message : String(reason))
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  if (error) {
    return <ErrorState title="Unable to load usage summary" description={error} />
  }

  if (!summary) {
    return <Loader />
  }

  const totalTokens = summary.total_input_tokens + summary.total_output_tokens + summary.total_cache_read_tokens + summary.total_cache_write_tokens

  return (
    <main className="flex h-full min-h-0 flex-col overflow-auto bg-(--ui-chat-surface-background) px-6 py-8">
      <div className="mx-auto w-full max-w-5xl">
        <header className="mb-6">
          <h1 className="text-2xl font-semibold text-foreground">Usage</h1>
          <p className="mt-1 text-sm text-(--ui-text-tertiary)">Local session and estimated token costs</p>
        </header>

        <section aria-label="Usage summary" className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <SummaryCard label="Total sessions" value={compactNumber(summary.total_sessions)} />
          <SummaryCard label="Sessions with tokens" value={compactNumber(summary.token_sessions)} />
          <SummaryCard label="Sessions with cost" value={compactNumber(summary.cost_sessions)} />
          <SummaryCard label="Estimated spend" value={formatUsd(summary.total_estimated_cost_usd)} />
          <SummaryCard label="Total tokens" value={compactNumber(totalTokens)} />
          <SummaryCard label="Input tokens" value={compactNumber(summary.total_input_tokens)} />
          <SummaryCard label="Output tokens" value={compactNumber(summary.total_output_tokens)} />
          <SummaryCard label="Cache read tokens" value={compactNumber(summary.total_cache_read_tokens)} />
          <SummaryCard label="Most expensive session" value={formatUsd(summary.most_expensive_session_usd)} />
          <SummaryCard label="Least expensive session" value={formatUsd(summary.cheapest_session_usd)} />
        </section>
      </div>
    </main>
  )
}
