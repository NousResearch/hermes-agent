import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { Progress } from '@/components/ui/progress'
import { useI18n } from '@/i18n'
import { $previewUblock, loadPreviewUblock, setPreviewUblockEnabled } from '@/store/preview-ublock'

import { ToggleRow } from './primitives'

const BUSY_PHASES = new Set([
  'checking-cache',
  'downloading',
  'extracting',
  'loading',
  'preparing',
  'validating',
  'verifying'
])

function formatBytes(value: number): string {
  if (value < 1024) {
    return `${value} B`
  }

  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`
  }

  return `${(value / (1024 * 1024)).toFixed(1)} MB`
}

function phaseLabel(
  phase: string,
  copy: {
    checkingCache: string
    downloading: string
    extracting: string
    loading: string
    preparing: string
    validating: string
    verifying: string
  }
): string {
  const labels: Record<string, string> = {
    'checking-cache': copy.checkingCache,
    downloading: copy.downloading,
    extracting: copy.extracting,
    loading: copy.loading,
    preparing: copy.preparing,
    validating: copy.validating,
    verifying: copy.verifying
  }

  return labels[phase] ?? phase
}

export function PreviewUblockSetting() {
  const { t } = useI18n()
  const copy = t.settings.config
  const previewUblock = useStore($previewUblock)
  const operation = previewUblock.operation
  const [pending, setPending] = useState(false)
  const [transportFailure, setTransportFailure] = useState(false)
  // A rejected IPC request is no longer in flight, even if the last
  // authoritative operation update was a busy phase. The local failure must
  // restore the retry affordance instead of leaving the toggle stranded.
  const busy = !transportFailure && (pending || BUSY_PHASES.has(operation.phase))
  const visiblePhase = pending && operation.phase === 'idle' ? 'checking-cache' : operation.phase
  const determinate = operation.totalBytes !== null && operation.totalBytes > 0

  const status =
    busy || operation.phase === 'failed' || operation.phase === 'ready' || transportFailure ? (
      <div
        aria-live="polite"
        className="mt-2 space-y-2 rounded-md border border-(--ui-border-subtle) bg-(--ui-bg-secondary) px-3 py-2 text-[length:var(--conversation-caption-font-size)]"
        data-testid="preview-ublock-operation"
      >
        <div className="flex items-center gap-2 text-(--ui-text-secondary)">
          {busy && <GlyphSpinner ariaLabel={phaseLabel(visiblePhase, copy.previewUblock)} className="size-3" />}
          <span>
            {visiblePhase === 'ready'
              ? copy.previewUblock.enabled(previewUblock.version ?? 'uBlock Origin Lite')
              : phaseLabel(visiblePhase, copy.previewUblock)}
          </span>
        </div>
        {busy && visiblePhase === 'downloading' && (
          <div className="space-y-1">
            <Progress
              aria-label={copy.previewUblock.downloadProgress}
              indeterminate={!determinate}
              value={determinate ? operation.receivedBytes / (operation.totalBytes ?? 1) : undefined}
            />
            <div className="text-(--ui-text-tertiary)">
              {determinate && operation.totalBytes !== null
                ? `${formatBytes(operation.receivedBytes)} / ${formatBytes(operation.totalBytes)}`
                : copy.previewUblock.transferred(formatBytes(operation.receivedBytes))}
            </div>
          </div>
        )}
        {((operation.phase === 'failed' && operation.failure) || transportFailure) && (
          <div className="flex items-center justify-between gap-3 text-destructive" role="alert">
            <span>
              {transportFailure ? copy.previewUblockFailure : operation.failure?.message || copy.previewUblockFailure}
            </span>
            <Button disabled={busy} onClick={() => void handleChange(true)} size="sm" type="button" variant="outline">
              {copy.previewUblock.retry}
            </Button>
          </div>
        )}
      </div>
    ) : undefined

  useEffect(() => {
    void loadPreviewUblock()
  }, [])

  useEffect(() => {
    setTransportFailure(false)
  }, [previewUblock])

  const handleChange = async (enabled: boolean) => {
    setTransportFailure(false)
    setPending(true)

    try {
      await setPreviewUblockEnabled(enabled)
    } catch {
      setTransportFailure(true)
    } finally {
      setPending(false)
    }
  }

  return (
    <ToggleRow
      below={status}
      checked={previewUblock.enabled}
      description={copy.previewUblockDescription}
      disabled={busy}
      label={copy.previewUblockTitle}
      onChange={on => void handleChange(on)}
    />
  )
}
