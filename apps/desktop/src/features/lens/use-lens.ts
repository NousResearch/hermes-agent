import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'

import { type LensGuest, readLensGuest, reloadLensGuest } from './capture'
import type { LensCard } from './model'
import {
  $lensScope,
  findLensGuest,
  lensOperationIsCurrent,
  pinLensCapture,
  syncLensCards,
  updateLensCapture
} from './store'

export function useLens(getGuest: () => LensGuest | null) {
  const { t } = useI18n()
  const scope = useStore($lensScope)
  const [open, setOpen] = useState(false)
  const [pendingScope, setPendingScope] = useState<string | null>(null)
  const [failure, setFailure] = useState({ scope: '', text: '' })
  const busyRef = useRef(false)
  const lifetime = useRef(0)
  useEffect(
    () => () => {
      lifetime.current += 1
    },
    []
  )

  const onError = (error: unknown) => {
    const key = error instanceof Error ? error.message : 'unavailable'
    const errors = t.lens.errors
    setFailure({ scope, text: key in errors ? errors[key as keyof typeof errors] : errors.unavailable })
  }

  const run = async (action: (alive: () => boolean) => Promise<void>) => {
    if (busyRef.current) {
      return
    }

    const started = lifetime.current
    const alive = () => lifetime.current === started
    busyRef.current = true
    setPendingScope(scope)
    setFailure({ scope, text: '' })

    try {
      await action(alive)
    } catch (error) {
      if (alive()) {
        onError(error)
      }
    } finally {
      busyRef.current = false

      if (alive()) {
        setPendingScope(null)
      }
    }
  }

  return {
    open,
    busy: pendingScope === scope,
    error: failure.scope === scope ? failure.text : '',
    onError,
    onClose: () => setOpen(false),
    toggle: () => {
      syncLensCards()
      setOpen(value => !value)
    },
    onPin: (mode: 'page' | 'selection') => {
      void run(async alive => {
        const guest = getGuest()

        if (!guest) {
          throw new Error('unavailable')
        }

        const current = lensOperationIsCurrent(scope, guest)

        if (!current()) {
          throw new Error('unavailable')
        }

        const capture = await readLensGuest(guest, mode)

        if (alive() && current()) {
          pinLensCapture(capture, scope)
        }
      })
    },
    onRefresh: (card: LensCard) => {
      void run(async alive => {
        const guest = findLensGuest(card)

        if (!guest) {
          throw new Error('openFirst')
        }

        const current = lensOperationIsCurrent(scope, guest)

        if (!current()) {
          throw new Error('unavailable')
        }

        const capture = await reloadLensGuest(guest, card)

        if (alive() && current()) {
          updateLensCapture(card, capture)
        }
      })
    }
  }
}
