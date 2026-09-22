import { Button } from '@/components/ui/button'

export interface RecoveryTarget { installId: string; phase: string; unknown: boolean; fenced: boolean; reason: string | null }

export function RecoveryPanel({ target, onRecheck, onRecover, onRetry, onExclude, onStop }: { target: RecoveryTarget; onRecheck: () => void; onRecover: () => void; onRetry: () => void; onExclude: () => void; onStop: () => void }) {
  return <section aria-label="Managed rollout recovery" className="grid gap-2"><p className="text-sm">{target.installId} · {target.phase}</p>{target.unknown || target.fenced ? <p className="text-xs text-amber-600">Unknown outcome or recovery fence retained: {target.reason ?? 'manual action required'}.</p> : null}<div className="flex flex-wrap gap-2"><Button onClick={onRecheck} variant="outline">Recheck</Button><Button disabled={!target.fenced} onClick={onRecover}>Recover</Button><Button onClick={onRetry} variant="secondary">Retry</Button><Button onClick={onExclude} variant="secondary">Exclude</Button><Button onClick={onStop} variant="destructive">Stop</Button></div></section>
}
