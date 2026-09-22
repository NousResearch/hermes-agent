import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'

export function RolloutControls({ phase, onCommand, onVerify }: { phase: string; onCommand: (action: 'pause' | 'stop') => void; onVerify: () => void }) {
  const [pending, setPending] = useState<'pause' | 'stop' | null>(null)

  useEffect(() => {
    if ((pending === 'pause' && phase === 'paused') || (pending === 'stop' && phase === 'stopped')) {
      setPending(null)
    }
  }, [pending, phase])
  const command = (action: 'pause' | 'stop') => { if (pending) return; setPending(action); onCommand(action) }
  return <div className="flex flex-wrap gap-2" aria-label="Managed rollout controls">
    <Button disabled={Boolean(pending) || phase === 'stopped'} onClick={() => command('pause')} variant="secondary">{pending === 'pause' ? 'Pausing…' : 'Pause'}</Button>
    <Button disabled={Boolean(pending) || phase === 'stopped'} onClick={() => command('stop')} variant="destructive">{pending === 'stop' ? 'Stopping…' : 'Stop'}</Button>
    <Button disabled={Boolean(pending)} onClick={onVerify} variant="outline">Verify before promotion</Button>
  </div>
}
