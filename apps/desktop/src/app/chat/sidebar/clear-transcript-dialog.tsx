import * as React from 'react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { FieldHint } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'

export type ClearMode = 'all' | 'last_n' | 'before'

export interface ClearTranscriptOptions {
  keep_last_n?: number
  before_timestamp?: string
}

export interface ClearTranscriptDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onConfirm: (options?: ClearTranscriptOptions) => Promise<void> | void
  sessionTitle: string
}

export function ClearTranscriptDialog({
  open,
  onOpenChange,
  onConfirm,
  sessionTitle
}: ClearTranscriptDialogProps) {
  const [mode, setMode] = useState<ClearMode>('all')
  const [keepLastN, setKeepLastN] = useState<number>(5)
  const [beforeTime, setBeforeTime] = useState<string>('1d')
  const [submitting, setSubmitting] = useState<boolean>(false)

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault()
    if (submitting) return

    setSubmitting(true)
    try {
      if (mode === 'all') {
        await onConfirm()
      } else if (mode === 'last_n') {
        await onConfirm({ keep_last_n: Math.max(1, keepLastN) })
      } else if (mode === 'before') {
        await onConfirm({ before_timestamp: beforeTime.trim() })
      }
      onOpenChange(false)
    } finally {
      setSubmitting(false)
    }
  }

  const PRESET_LAST = [2, 5, 10]
  const PRESET_BEFORE = [
    { label: '1 hour', value: '1h' },
    { label: '1 day', value: '1d' },
    { label: '1 week', value: '1w' },
    { label: '30 days', value: '30d' }
  ]

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-sm font-semibold">
            <Codicon name="clear-all" size="1rem" className="text-destructive" />
            <span>Clear transcript</span>
          </DialogTitle>
          <DialogDescription className="text-xs text-muted-foreground">
            Clear messages from &ldquo;{sessionTitle}&rdquo;. The session title, pins, and settings will be preserved.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-3 py-2 text-xs">
          {/* Mode 1: All */}
          <div
            onClick={() => setMode('all')}
            className={cn(
              'cursor-pointer rounded border p-2.5 transition-colors',
              mode === 'all'
                ? 'border-destructive/60 bg-destructive/5 dark:bg-destructive/10'
                : 'border-border hover:bg-muted/40'
            )}
          >
            <div className="flex items-center justify-between">
              <span className="font-medium text-foreground">All messages</span>
              <span className="text-[0.6875rem] text-muted-foreground">Reset context to 0%</span>
            </div>
            <p className="mt-1 text-[0.6875rem] text-muted-foreground">
              Delete all conversation turns and tool results completely.
            </p>
          </div>

          {/* Mode 2: Keep last N */}
          <div
            onClick={() => setMode('last_n')}
            className={cn(
              'cursor-pointer rounded border p-2.5 transition-colors',
              mode === 'last_n'
                ? 'border-primary/60 bg-primary/5 dark:bg-primary/10'
                : 'border-border hover:bg-muted/40'
            )}
          >
            <div className="flex items-center justify-between">
              <span className="font-medium text-foreground">Keep recent messages</span>
            </div>
            <p className="mt-1 text-[0.6875rem] text-muted-foreground">
              Keep the newest messages and wipe older turns.
            </p>

            {mode === 'last_n' && (
              <div className="mt-2.5 flex items-center gap-2 pt-1" onClick={e => e.stopPropagation()}>
                <span className="text-muted-foreground">Keep last:</span>
                <Input
                  type="number"
                  min={1}
                  max={500}
                  value={keepLastN}
                  onChange={e => setKeepLastN(Math.max(1, parseInt(e.target.value, 10) || 1))}
                  className="w-16 text-center"
                  size="sm"
                />
                <span className="text-muted-foreground">messages</span>
                <div className="ml-auto flex gap-1">
                  {PRESET_LAST.map(n => (
                    <Button
                      key={n}
                      type="button"
                      variant={keepLastN === n ? 'default' : 'outline'}
                      size="xs"
                      onClick={() => setKeepLastN(n)}
                    >
                      {n}
                    </Button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Mode 3: Clear older than */}
          <div
            onClick={() => setMode('before')}
            className={cn(
              'cursor-pointer rounded border p-2.5 transition-colors',
              mode === 'before'
                ? 'border-primary/60 bg-primary/5 dark:bg-primary/10'
                : 'border-border hover:bg-muted/40'
            )}
          >
            <div className="flex items-center justify-between">
              <span className="font-medium text-foreground">Clear older than</span>
            </div>
            <p className="mt-1 text-[0.6875rem] text-muted-foreground">
              Delete messages created before a timeframe or date.
            </p>

            {mode === 'before' && (
              <div className="mt-2.5 space-y-2 pt-1" onClick={e => e.stopPropagation()}>
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-muted-foreground">Older than:</span>
                  {PRESET_BEFORE.map(p => (
                    <Button
                      key={p.value}
                      type="button"
                      variant={beforeTime === p.value ? 'default' : 'outline'}
                      size="xs"
                      onClick={() => setBeforeTime(p.value)}
                    >
                      {p.label}
                    </Button>
                  ))}
                </div>
                <Input
                  type="text"
                  value={beforeTime}
                  onChange={e => setBeforeTime(e.target.value)}
                  placeholder="e.g. 2d, 5h, 2026-08-01"
                  className="w-full text-xs"
                  size="sm"
                />
                <FieldHint>
                  Formats: <code className="rounded bg-muted/80 px-1 py-0.5 font-mono text-[0.625rem] text-foreground">30m</code>, <code className="rounded bg-muted/80 px-1 py-0.5 font-mono text-[0.625rem] text-foreground">5h</code>, <code className="rounded bg-muted/80 px-1 py-0.5 font-mono text-[0.625rem] text-foreground">2d</code>, <code className="rounded bg-muted/80 px-1 py-0.5 font-mono text-[0.625rem] text-foreground">1w</code>, <code className="rounded bg-muted/80 px-1 py-0.5 font-mono text-[0.625rem] text-foreground">30d</code>, or <code className="rounded bg-muted/80 px-1 py-0.5 font-mono text-[0.625rem] text-foreground">YYYY-MM-DD</code>
                </FieldHint>
              </div>
            )}
          </div>

          <DialogFooter className="gap-2 pt-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={submitting}
              onClick={() => onOpenChange(false)}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              variant="destructive"
              size="sm"
              disabled={submitting}
            >
              {submitting ? 'Clearing…' : 'Clear transcript'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
