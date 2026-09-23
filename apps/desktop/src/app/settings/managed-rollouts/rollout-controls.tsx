import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { getManagedRolloutMessages } from '@/i18n/managed-rollouts'

export function RolloutControls({ phase, onCommand, onVerify }: { phase: string; onCommand: (action: 'pause' | 'resume' | 'stop') => Promise<boolean | void> | boolean | void; onVerify: () => void }) {
  const { t } = useI18n()
  const messages = getManagedRolloutMessages(t)
  const [pending, setPending] = useState<'pause' | 'resume' | 'stop' | null>(null)

  useEffect(() => {
    if ((pending === 'pause' && phase === 'paused') || (pending === 'resume' && phase === 'running') || (pending === 'stop' && phase === 'stopped')) {
      setPending(null)
    }
  }, [pending, phase])

  const command = (action: 'pause' | 'resume' | 'stop') => {
    if (pending) {return}
    setPending(action)

    try {
      // The command owner waits for its acknowledgement and snapshot refresh.
      // A refresh can fail after acceptance, or skip an intermediate phase;
      // neither case may leave these controls disabled indefinitely.
      void Promise.resolve(onCommand(action)).then(() => setPending(null), () => setPending(null))
    } catch {
      setPending(null)
    }
  }

  return <div aria-label={messages.sections.controls} className="flex min-w-0 flex-wrap gap-2" role="group">
    {phase === 'paused'
      ? <Button className="motion-reduce:transition-none" disabled={Boolean(pending)} onClick={() => command('resume')} type="button" variant="secondary">{messages.actions.resume}</Button>
      : <Button className="motion-reduce:transition-none" disabled={Boolean(pending) || phase === 'stopped'} onClick={() => command('pause')} type="button" variant="secondary">{pending === 'pause' ? messages.actions.pausing : messages.actions.pause}</Button>}
    <Button className="motion-reduce:transition-none" disabled={Boolean(pending) || phase === 'stopped'} onClick={() => command('stop')} type="button" variant="destructive">{pending === 'stop' ? messages.actions.stopping : messages.actions.stop}</Button>
    <Button className="motion-reduce:transition-none" disabled={Boolean(pending)} onClick={onVerify} type="button" variant="outline">{messages.actions.verifyBeforePromotion}</Button>
  </div>
}
