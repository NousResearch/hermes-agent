export interface RolloutAttempt {
  installId: string
  phase: string
  expanded: boolean
  receipt: { outcome: string; correlationId: string } | null
  readiness: { ready: boolean; reason: string | null } | null
  unknown?: boolean
  fenced?: boolean
}

export function AttemptDetail({ attempt, onToggle }: { attempt: RolloutAttempt; onToggle: (installId: string) => void }) {
  return <article className="border-b border-(--ui-stroke-secondary) py-2">
    <button aria-expanded={attempt.expanded} className="w-full text-left text-sm" onClick={() => onToggle(attempt.installId)}>{attempt.installId} · {attempt.phase}</button>
    {attempt.expanded ? <div className="grid gap-1 pt-2 text-xs">
      <p>Receipt: {attempt.receipt ? `${attempt.receipt.outcome} (${attempt.receipt.correlationId})` : 'pending or unavailable'}</p>
      <p>Readiness: {attempt.readiness?.ready ? 'last observed ready' : attempt.readiness?.reason ?? 'pending evidence'}</p>
      {attempt.unknown ? <p className="text-amber-600">Remote outcome remains unknown.</p> : null}
      {attempt.fenced ? <p className="text-amber-600">Recovery fence remains active.</p> : null}
    </div> : null}
  </article>
}
