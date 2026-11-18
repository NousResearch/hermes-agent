import { createCronTriggerController, type CronTriggerController } from '@hermes/shared'
import { useEffect, useRef, useState } from 'react'

/** The Run-now controller for a surface's lifetime, and which jobs it is triggering right now. The
 *  ref is null once the surface unmounts, so a request that settles later can tell it is orphaned. */
export function useCronTriggerController() {
  const [triggeringJobKeys, setTriggeringJobKeys] = useState<ReadonlySet<string>>(() => new Set())
  const triggerControllerRef = useRef<CronTriggerController | null>(null)

  // eslint-disable-next-line no-restricted-syntax -- controller mount identity, not an atom mirror
  useEffect(() => {
    const controller = createCronTriggerController((key, running) => {
      if (triggerControllerRef.current !== controller) {
        return
      }

      setTriggeringJobKeys(current => {
        const next = new Set(current)

        if (running) {
          next.add(key)
        } else {
          next.delete(key)
        }

        return next
      })
    })

    triggerControllerRef.current = controller

    return () => {
      triggerControllerRef.current = null
    }
  }, [])

  return { triggerControllerRef, triggeringJobKeys }
}
