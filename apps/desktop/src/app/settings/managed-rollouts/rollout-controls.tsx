import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'

export function RolloutControls({ phase, onCommand, onVerify }: { phase: string; onCommand: (action: 'pause' | 'stop') => void; onVerify: () => void }) {
  const [pending, setPending] = useState<'pause' | 'stop' | null>(null)

  useEffect(() => {
    if ((pending === 'pause' && phase === 'paused') || (pending === 'stop' && phase === 'stopped')) {
      setPending(null)
    }
  }, [pending, phase])

  const command = (action: 'pause' | 'stop') => { if (pending) {return;} setPending(action); onCommand(action) }

  return <div aria-label="Managed rollout controls" className="flex min-w-0 flex-wrap gap-2">
    <Button className="motion-reduce:transition-none" disabled={Boolean(pending) || phase === 'stopped'} onClick={() => command('pause')} type="button" variant="secondary">{pending === 'pause' ? 'Pausing…' : 'Pause'}</Button>
    <Button className="motion-reduce:transition-none" disabled={Boolean(pending) || phase === 'stopped'} onClick={() => command('stop')} type="button" variant="destructive">{pending === 'stop' ? 'Stopping…' : 'Stop'}</Button>
    <Button className="motion-reduce:transition-none" disabled={Boolean(pending)} onClick={onVerify} type="button" variant="outline">Verify before promotion</Button>
  </div>
}
