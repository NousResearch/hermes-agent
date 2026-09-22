export interface RolloutAttempt {
  installId: string
  phase: string
  expanded: boolean
  receipt: { outcome: string; correlationId: string } | null
  readiness: { ready: boolean; reason: string | null } | null
}

export function AttemptDetail({ attempt, onToggle }: { attempt: RolloutAttempt; onToggle: (installId: string) => void }) {
  return <article className="border-b border-(--ui-stroke-secondary) py-2">
    <button aria-expanded={attempt.expanded} className="w-full text-left text-sm" onClick={() => onToggle(attempt.installId)}>{attempt.installId} · {attempt.phase}</button>
    {attempt.expanded ? <div className="grid gap-1 pt-2 text-xs"><p>Receipt: {attempt.receipt?.outcome ?? 'pending'}</p><p>Readiness: {attempt.readiness?.ready ? 'ready' : attempt.readiness?.reason ?? 'pending evidence'}</p></div> : null}
  </article>
}
