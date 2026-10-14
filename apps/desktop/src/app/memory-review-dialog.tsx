import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'
import { DiffLines } from '@/components/chat/diff-lines'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { useI18n } from '@/i18n'
import { $memoryReview, type MemoryPending } from '@/store/memory-review'

export function MemoryReviewDialog() {
  const review = useStore($memoryReview)
  return review ? <MemoryReviewContent key={review.sessionId} review={review} /> : null
}

function MemoryReviewContent({ review }: { review: NonNullable<ReturnType<typeof $memoryReview.get>> }) {
  const { t, locale } = useI18n()
  const copy = t.memoryReview
  const [raw, setRaw] = useState(false)
  const [data, setData] = useState<MemoryPending | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(true)
  const params = { session_id: review.sessionId }
  const refresh = async () => {
    setLoading(true)
    try {
      const result = await review.request<MemoryPending>('memory.pending', params)
      if ($memoryReview.get() === review) setData(result)
    } catch (err) {
      if ($memoryReview.get() === review) setError(String(err))
    } finally {
      setLoading(false)
    }
  }
  useEffect(() => {
    void refresh()
  }, [review])
  const decide = async (batch: MemoryPending['batches'][number], decision: 'approve' | 'reject') => {
    if (busy) return
    setBusy(true)
    setError('')
    try {
      const result = await review.request<{ success: boolean; error: string }>('memory.decide', {
        ...params,
        id: batch.id,
        decision,
        revision: batch.revision
      })
      if (!result.success) setError(result.error)
      await refresh()
    } catch (err) {
      setError(String(err))
    } finally {
      setBusy(false)
    }
  }
  return (
    <Dialog
      open
      onOpenChange={open => {
        if (!open) $memoryReview.set(null)
      }}
    >
      <DialogContent className="max-w-4xl" bodyClassName="max-h-[80vh] overflow-auto">
        <DialogHeader>
          <DialogTitle>{copy.title}</DialogTitle>
          <DialogDescription>{copy.description}</DialogDescription>
        </DialogHeader>
        {data && <p>{data.write_approval ? copy.gateOn : copy.gateOff}</p>}
        {loading && <p role="status">{copy.loading}</p>}
        {error && <p role="alert">{error}</p>}
        <Button
          variant="secondary"
          size="sm"
          disabled={busy || loading}
          onClick={() => {
            setError('')
            void refresh()
          }}
        >
          {copy.refresh}
        </Button>
        {data?.batches.length === 0 && <p>{copy.empty}</p>}
        {data?.batches.map(batch => (
          <section key={batch.id} className="space-y-3 py-4">
            <h3>{batch.summary}</h3>
            <p className="text-sm text-(--ui-text-secondary)">
              {batch.id} · {batch.target === 'user' ? 'USER.md' : 'MEMORY.md'} ·{' '}
              {batch.origin === 'background_review' ? copy.background : copy.foreground} ·{' '}
              {copy.operations(batch.operation_count)} · {new Date(batch.created_at * 1000).toLocaleString(locale)}
            </p>
            <p className="font-mono text-xs">
              --- a/{batch.target === 'user' ? 'USER.md' : 'MEMORY.md'} → +++ b/
              {batch.target === 'user' ? 'USER.md' : 'MEMORY.md'}
            </p>
            <Button variant="secondary" size="sm" onClick={() => setRaw(!raw)}>
              {raw ? copy.formatted : copy.raw}
            </Button>
            {raw ? (
              <pre data-testid="memory-raw-diff" className="overflow-auto whitespace-pre font-mono text-xs">
                {batch.diff}
              </pre>
            ) : (
              <DiffLines text={batch.diff || batch.before || copy.noChange} className="m-0 max-h-none" />
            )}
            {batch.error && <p role="alert">{batch.error}</p>}
            <div className="flex gap-2">
              <Button
                size="sm"
                disabled={busy || loading || !batch.can_approve}
                onClick={() => void decide(batch, 'approve')}
              >
                {copy.approve}
              </Button>
              <Button
                size="sm"
                variant="secondary"
                disabled={busy || loading}
                onClick={() => void decide(batch, 'reject')}
              >
                {copy.reject}
              </Button>
            </div>
          </section>
        ))}
      </DialogContent>
    </Dialog>
  )
}
