// Dev-only: drive the credit-notice UX without a backend that can hit real
// usage bands. Each trigger emits ONE synthetic `notification.show` /
// `notification.clear` gateway event through the real fan-out
// (`emitLocalGatewayEvent`), so it exercises the actual dispatcher branch in
// `use-message-stream/gateway-event.ts` — toast render, key-replacement
// escalation, TTL self-dismiss, native OS notification, and billing re-poll.
//
// Installed only under `import.meta.env.DEV` (see contrib/wiring.tsx), so none
// of this ships in a production build.

import type { GatewayEvent } from '@hermes/shared'

import { PALETTE_AREA, type PaletteContribution } from '@/app/command-palette/contrib'
import { registry } from '@/contrib/registry'
import { CreditCard } from '@/lib/icons'
import { emitLocalGatewayEvent } from '@/store/gateway'
import { $activeSessionId } from '@/store/session'

interface NoticeStep {
  key: string
  level: string
  kind: 'sticky' | 'ttl'
  text: string
  ttl_ms?: number
}

// Walks the same lifecycle the Nous credits tracker drives: usage escalates in
// place (50→75→90, one key), then grant-spent, then the depleted/restored pair.
// Wraps around. These are all separate SHOW steps; the stepper auto-clears the
// previous notice when the key changes, so the demo shows one toast at a time
// (real usage CAN stack these, but that's noise when you're eyeballing a single
// transition). The same-key usage steps still demonstrate in-place escalation.
const STEPS: readonly NoticeStep[] = [
  { key: 'credits.usage', kind: 'sticky', level: 'info', text: "• You've used $110.00 of your $220.00 cap" },
  { key: 'credits.usage', kind: 'sticky', level: 'warn', text: "⚠ You've used $165.00 of your $220.00 cap" },
  { key: 'credits.usage', kind: 'sticky', level: 'warn', text: "⚠ You've used $198.00 of your $220.00 cap" },
  { key: 'credits.grant_spent', kind: 'sticky', level: 'info', text: '• Grant spent · $12.00 top-up left' },
  { key: 'credits.depleted', kind: 'sticky', level: 'error', text: '✕ Credit access paused · run /topup to top up' },
  { key: 'credits.restored', kind: 'ttl', level: 'success', text: '✓ Credit access restored', ttl_ms: 8000 }
]

let cursor = 0
let lastShownKey: null | string = null

function clearNotice(key: string): void {
  emitLocalGatewayEvent({
    payload: { key },
    session_id: $activeSessionId.get() ?? '',
    type: 'notification.clear'
  } as GatewayEvent)
}

function showNotice(step: NoticeStep): void {
  emitLocalGatewayEvent({
    payload: {
      id: `${step.key}:${Date.now()}`,
      key: step.key,
      kind: step.kind,
      level: step.level,
      text: step.text,
      ttl_ms: step.ttl_ms ?? null
    },
    session_id: $activeSessionId.get() ?? '',
    type: 'notification.show'
  } as GatewayEvent)
}

/** Fire the next notice in the scripted sequence, wrapping at the end. */
export function stepCreditsNoticeDemo(): void {
  const step = STEPS[cursor % STEPS.length]

  // One toast at a time: retire the previous notice when we move to a new key.
  // (Same-key steps skip this, so the usage line still escalates in place.)
  if (lastShownKey && lastShownKey !== step.key) {
    clearNotice(lastShownKey)
  }

  showNotice(step)
  lastShownKey = step.key
  cursor += 1
}

// The hotkey: Ctrl+Shift+C on every platform (Ctrl, not Cmd, so it can't clash
// with a system Cmd chord and works the same on Windows/Linux). Matched on
// `code` so it's keyboard-layout independent, and captured before the composer
// so a focused input can't swallow it.
function isTriggerChord(e: KeyboardEvent): boolean {
  return e.ctrlKey && e.shiftKey && !e.metaKey && !e.altKey && e.code === 'KeyC'
}

/**
 * Install the dev trigger: a capture-phase hotkey (Ctrl+Shift+C), a ⌘K palette
 * entry, and a `window.__creditsDemo()` console hook. Returns a disposer.
 */
export function installCreditsNoticeDemo(): () => void {
  const onKeyDown = (e: KeyboardEvent) => {
    if (!isTriggerChord(e)) {
      return
    }

    e.preventDefault()
    e.stopPropagation()
    stepCreditsNoticeDemo()
  }

  window.addEventListener('keydown', onKeyDown, { capture: true })
  ;(window as unknown as { __creditsDemo?: () => void }).__creditsDemo = stepCreditsNoticeDemo

  const disposePalette = registry.register({
    id: 'dev.creditsNotice',
    area: PALETTE_AREA,
    data: {
      id: 'dev.creditsNotice',
      icon: CreditCard,
      keywords: ['credits', 'notice', 'toast', 'billing', 'dev', 'demo'],
      label: 'Dev: cycle credit notices',
      run: stepCreditsNoticeDemo
    } satisfies PaletteContribution
  })

  console.info('[dev] credit-notice demo ready — press Ctrl+Shift+C, or run window.__creditsDemo()')

  return () => {
    window.removeEventListener('keydown', onKeyDown, { capture: true })
    disposePalette()
    delete (window as unknown as { __creditsDemo?: () => void }).__creditsDemo
  }
}
