/**
 * The quick tour after Start (D19): only once the overlay is gone and the tour's targets have
 * painted, and it finishes before the first message is sent so it never covers an approval card.
 */

import { $tourActive } from '@/lib/tour/tour-active'

const TARGET = '[data-tour="composer"]'
const PAINT_POLL_MS = 100
const PAINT_WAIT_MS = 3_000

function targetsPainted(): Promise<void> {
  const started = Date.now()

  return new Promise(resolve => {
    const poll = () => {
      if (document.querySelector(TARGET) || Date.now() - started > PAINT_WAIT_MS) {
        resolve()
      } else {
        setTimeout(poll, PAINT_POLL_MS)
      }
    }

    poll()
  })
}

function tourClosed(): Promise<void> {
  if (!$tourActive.get()) {
    return Promise.resolve()
  }

  return new Promise(resolve => {
    const stop = $tourActive.listen(active => {
      if (!active) {
        stop()
        resolve()
      }
    })
  })
}

export async function runQuickTourWhenTargetsPaint(): Promise<void> {
  await targetsPainted()
  // Loaded on use: the tour engine pulls driver.js, which stays off the boot path.
  const { runBuiltInTour } = await import('@/app/chat/built-in-tour')
  const result = await runBuiltInTour('quick', () => true, '')

  if (result.success) {
    await tourClosed()
  }
}
