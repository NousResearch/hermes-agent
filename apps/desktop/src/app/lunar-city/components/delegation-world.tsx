import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useState } from 'react'

import type { EntityKey, LunarCityWorldHandle, LunarEntity, Vec3 } from '../model'
import { $lunarCitySnapshot } from '../store'

import { DelegationLinks } from './delegation-links'

interface DelegationWorldProps {
  worldRef: { current: LunarCityWorldHandle | undefined }
  enabled: boolean
  reducedMotion: boolean
  onSelect(entity: LunarEntity): void
}

function ActiveDelegationWorld({ worldRef, onSelect }: Pick<DelegationWorldProps, 'worldRef' | 'onSelect'>) {
  const snapshot = useStore($lunarCitySnapshot)
  const presentation = useCallback((key: EntityKey) => worldRef.current?.getWorkerPresentation?.(key), [worldRef])

  const project = useCallback((point: Vec3) => {
    const projected = worldRef.current?.projectWorldPoint?.(point)

    return projected?.visible ? projected : undefined
  }, [worldRef])

  return <DelegationLinks enabled onSelectEntity={onSelect} presentation={presentation} project={project}
    reducedMotion={false} snapshot={snapshot} visible />
}

/** Disabled overlays do not subscribe to the high-frequency operational snapshot. */
export function DelegationWorld({ worldRef, enabled, reducedMotion, onSelect }: DelegationWorldProps) {
  const [visible, setVisible] = useState(!document.hidden)
  useEffect(() => {
    const changed = () => setVisible(!document.hidden)
    document.addEventListener('visibilitychange', changed)

    return () => document.removeEventListener('visibilitychange', changed)
  }, [])

  return enabled && !reducedMotion && visible
    ? <ActiveDelegationWorld onSelect={onSelect} worldRef={worldRef} /> : null
}
