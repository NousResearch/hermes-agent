import { Button } from '@/components/ui/button'

export interface RecoveryTarget { installId: string; phase: string; unknown: boolean; fenced: boolean; reason: string | null }

export function RecoveryPanel({ target, onRecheck, onRecover, onRetry, onExclude, onStop }: { target: RecoveryTarget; onRecheck: () => void; onRecover: () => void; onRetry: () => void; onExclude: () => void; onStop: () => void }) {
  return <section aria-label="Managed rollout recovery" className="grid min-w-0 gap-2"><p className="text-sm">{target.installId} · {target.phase}</p>{target.unknown || target.fenced ? <p className="text-xs text-amber-600">Unknown outcome or recovery fence retained: {target.reason ?? 'manual action required'}.</p> : null}<div className="flex min-w-0 flex-wrap gap-2"><Button className="motion-reduce:transition-none" onClick={onRecheck} type="button" variant="outline">Recheck</Button><Button className="motion-reduce:transition-none" disabled={!target.fenced} onClick={onRecover} type="button">Recover</Button><Button className="motion-reduce:transition-none" onClick={onRetry} type="button" variant="secondary">Retry</Button><Button className="motion-reduce:transition-none" onClick={onExclude} type="button" variant="secondary">Exclude</Button><Button className="motion-reduce:transition-none" onClick={onStop} type="button" variant="destructive">Stop</Button></div></section>
}
