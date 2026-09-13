import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'

interface CompressionCountStatusProps {
  count?: number
}

/** Current live runtime count from `usage.compressions`.
 * Missing stays missing for older/cold usage snapshots; zero is real data. */
export function CompressionCountStatus({ count }: CompressionCountStatusProps) {
  const { t } = useI18n()

  if (count === undefined) {
    return null
  }

  return (
    <Tip label={t.shell.statusbar.compressions(count)}>
      <div
        aria-label={t.shell.statusbar.compressions(count)}
        className="inline-flex h-full items-center gap-1 px-1.5 text-[0.6875rem] text-(--ui-text-tertiary)"
        data-testid="compression-count"
      >
        <span aria-hidden="true">🗜️</span>
        <span className="tabular-nums">{count}</span>
      </div>
    </Tip>
  )
}
