import { REASONING_EFFORTS, reasoningEffortLabel, resolveReasoningEffort } from '@/lib/reasoning-effort'

/** Resolve a raw effort value to meter props. Segments follow the effort
 *  scale itself, so a new backend level reshapes every meter. `none`
 *  (thinking off) fills nothing and labels Off. */
export function resolveEffortMeter(value: string, fallback?: string): { fill: number; label: string } {
  const resolved = resolveReasoningEffort(value || '', fallback)

  if (resolved === '') {
    return { fill: 0, label: 'Off' }
  }

  const index = REASONING_EFFORTS.findIndex(level => level === resolved)

  return { fill: index < 0 ? 0 : index + 1, label: reasoningEffortLabel(resolved) || resolved }
}

/** Thinking-level pill: short label over an underline gauge for the depth.
 *  Always fully visible — surrounding names may truncate, the level never does. */
export function EffortMeter({ value, fallback }: { value: string; fallback?: string }) {
  const { fill, label } = resolveEffortMeter(value, fallback)

  return (
    <span
      aria-label={label}
      className="relative flex h-5 w-12 shrink-0 flex-col items-center justify-center gap-[3px] overflow-hidden rounded-full bg-(--ui-bg-tertiary) px-1.5"
      role="img"
      title={label}
    >
      <span className="text-[0.62rem] font-semibold leading-none text-(--ui-text-secondary)">{label}</span>
      <span className="h-[3px] w-full overflow-hidden rounded-full bg-(--ui-text-tertiary)/25">
        <span
          className="block h-full rounded-full bg-(--ui-accent)"
          style={{ width: `${(fill / REASONING_EFFORTS.length) * 100}%` }}
        />
      </span>
    </span>
  )
}
