import { useStore } from '@nanostores/react'
import { useRef, useState } from 'react'

import { useI18n } from '@/i18n'

import { type LensGuest, readLensGuest, reloadLensGuest } from './capture'
import type { LensCard } from './model'
import { $lensScope, pinLensCapture, syncLensCards, updateLensCapture } from './store'

/** Readers live with the mounted guests, including hidden browser tabs. */
const guests = new Map<LensGuest, string>()

export function registerLensGuest(guest: LensGuest) {
  guests.set(guest, $lensScope.get())

  return () => {
    guests.delete(guest)
  }
}

export function useLens(getGuest: () => LensGuest | null) {
  const { t } = useI18n()
  const scope = useStore($lensScope)
  const [open, setOpen] = useState(false)
  const [pendingScope, setPendingScope] = useState<string | null>(null)
  const [failure, setFailure] = useState({ scope: '', text: '' })
  const busyRef = useRef(false)

  const onError = (error: unknown) => {
    const key = error instanceof Error ? error.message : 'unavailable'
    const errors = t.lens.errors
    setFailure({ scope, text: key in errors ? errors[key as keyof typeof errors] : errors.unavailable })
  }

  const run = async (action: () => Promise<void>) => {
    if (busyRef.current) {
      return
    }
    busyRef.current = true
    setPendingScope(scope)
    setFailure({ scope, text: '' })

    try {
      await action()
    } catch (error) {
      onError(error)
    } finally {
      busyRef.current = false
      setPendingScope(null)
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
      void run(async () => {
        const guest = getGuest()

        if (!guest) {
          throw new Error('unavailable')
        }
        const capture = await readLensGuest(guest, mode)
        pinLensCapture(capture, scope)
      })
    },
    onRefresh: (card: LensCard) => {
      void run(async () => {
        const guest = [...guests].find(([candidate, owner]) => {
          try {
            return owner === card.scope && candidate.getURL?.() === card.url
          } catch {
            return false
          }
        })?.[0]

        if (!guest) {
          throw new Error('openFirst')
        }
        updateLensCapture(card, await reloadLensGuest(guest, card))
      })
    }
  }
}
