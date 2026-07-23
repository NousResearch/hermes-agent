import { beforeEach, expect, test } from 'vitest'

import {
  type AgentNoticePayload,
  clearAgentNotice,
  nativeNoticeInput,
  noticeAccent,
  noticeToToast,
  showAgentNotice,
  splitMeta,
  stripGlyph,
  usageFraction
} from './agent-notices'
import { $notifications, clearNotifications } from './notifications'

function usage(overrides: Partial<AgentNoticePayload> = {}): AgentNoticePayload {
  return {
    key: 'credits.usage',
    kind: 'sticky',
    level: 'info',
    text: "• You've used $110.00 of your $220.00 cap",
    ...overrides
  }
}

beforeEach(() => {
  clearNotifications()
})

// ── noticeToToast: the whole mapping contract ────────────────────────────────

test('drops a notice with no text', () => {
  expect(noticeToToast(undefined)).toBeNull()
  expect(noticeToToast({ text: '' })).toBeNull()
  expect(noticeToToast({ text: '   ' })).toBeNull()
})

test('level maps to toast kind (warn → warning)', () => {
  expect(noticeToToast(usage({ level: 'info' }))?.kind).toBe('info')
  expect(noticeToToast(usage({ level: 'warn' }))?.kind).toBe('warning')
  expect(noticeToToast(usage({ level: 'error' }))?.kind).toBe('error')
  expect(noticeToToast(usage({ level: 'success' }))?.kind).toBe('success')
})

test('unknown / missing level falls back to info', () => {
  expect(noticeToToast({ text: 'x', level: 'bogus' })?.kind).toBe('info')
  expect(noticeToToast({ text: 'x' })?.kind).toBe('info')
})

test('sticky notices never auto-dismiss', () => {
  expect(noticeToToast(usage({ kind: 'sticky' }))?.durationMs).toBe(0)
})

test('ttl notice carries its ttl_ms as the duration', () => {
  const toast = noticeToToast({
    key: 'credits.restored',
    kind: 'ttl',
    level: 'success',
    text: '✓ restored',
    ttl_ms: 8000
  })

  expect(toast?.durationMs).toBe(8000)
})

test('ttl notice without a usable ttl_ms defers to notify()’s default', () => {
  expect(noticeToToast({ text: 'x', kind: 'ttl' })?.durationMs).toBeUndefined()
  expect(noticeToToast({ text: 'x', kind: 'ttl', ttl_ms: 0 })?.durationMs).toBeUndefined()
})

test('the notice key is the toast id, falling back to id', () => {
  expect(noticeToToast(usage({ key: 'credits.usage' }))?.id).toBe('credits.usage')
  expect(noticeToToast({ text: 'x', id: 'n1', key: undefined })?.id).toBe('n1')
})

test('the leading severity glyph is stripped from the toast message', () => {
  // The toast renders a kind icon, so the message must not double up on a glyph.
  expect(noticeToToast(usage())?.message).toBe("You've used $110.00 of your $220.00 cap")
  expect(noticeToToast(usage({ level: 'error', text: '✕ Credit access paused' }))?.message).toBe('Credit access paused')
  expect(noticeToToast(usage({ level: 'success', text: '✓ Credit access restored' }))?.message).toBe(
    'Credit access restored'
  )
})

test('the trailing "· detail" is split off as a secondary meta line, not inlined', () => {
  // grant_spent still carries a `· detail` tail.
  const grant = noticeToToast({ key: 'credits.grant_spent', level: 'info', text: '• Grant spent · $12.00 top-up left' })
  expect(grant?.message).toBe('Grant spent')
  expect(grant?.meta).toBe('$12.00 top-up left')

  // The usage line has no middot → whole line is the message, no meta.
  const plain = noticeToToast(usage())
  expect(plain?.message).toBe("You've used $110.00 of your $220.00 cap")
  expect(plain?.meta).toBeUndefined()
})

test('splitMeta splits on the first space-middot-space only', () => {
  expect(splitMeta('Grant spent · $12.00 top-up left')).toEqual(['Grant spent', '$12.00 top-up left'])
  expect(splitMeta('Credit access restored')).toEqual(['Credit access restored', undefined])
  // Interior middots after the first split stay in the meta.
  expect(splitMeta('a · b · c')).toEqual(['a', 'b · c'])
})

test('stripGlyph removes only a single leading severity glyph', () => {
  expect(stripGlyph('• Credits 50% used')).toBe('Credits 50% used')
  expect(stripGlyph('⚠ warn')).toBe('warn')
  expect(stripGlyph('✕ paused')).toBe('paused')
  expect(stripGlyph('✓ ok')).toBe('ok')
  // No leading glyph → unchanged; interior glyphs are preserved.
  expect(stripGlyph('Credits 50% used')).toBe('Credits 50% used')
  expect(stripGlyph('spent · $12.00 • top-up left')).toBe('spent · $12.00 • top-up left')
})

// ── noticeAccent: severity color ramp keyed off $used / $cap ─────────────────

test('usageFraction derives $used / $cap from the notice text', () => {
  expect(usageFraction("You've used $15.00 of your $20.00 cap")).toBeCloseTo(0.75)
  expect(usageFraction("You've used $198.00 of your $220.00 cap")).toBeCloseTo(0.9)
  // Fewer than two amounts, or a zero cap → no fraction.
  expect(usageFraction('Grant spent')).toBeNull()
  expect(usageFraction("You've used $5.00 of your $0.00 cap")).toBeNull()
  expect(usageFraction(undefined)).toBeNull()
})

test('usage accent stays muted below 75%, then ramps orange → red', () => {
  expect(noticeAccent(usage({ text: "• You've used $10.00 of your $20.00 cap" }))).toBeUndefined() // 50%
  expect(noticeAccent(usage({ text: "• You've used $14.80 of your $20.00 cap" }))).toBeUndefined() // 74%
  expect(noticeAccent(usage({ level: 'warn', text: "⚠ You've used $15.00 of your $20.00 cap" }))).toBe(
    'var(--ui-orange)'
  ) // 75%
  expect(noticeAccent(usage({ level: 'warn', text: "⚠ You've used $17.80 of your $20.00 cap" }))).toBe(
    'var(--ui-orange)'
  ) // 89%
  expect(noticeAccent(usage({ level: 'warn', text: "⚠ You've used $18.00 of your $20.00 cap" }))).toBe('var(--ui-red)') // 90%
  expect(noticeAccent(usage({ level: 'warn', text: "⚠ You've used $20.00 of your $20.00 cap" }))).toBe('var(--ui-red)') // 100%
})

test('terminal credit states carry their own accent; others stay default', () => {
  expect(noticeAccent({ key: 'credits.depleted', text: '✕ paused' })).toBe('var(--ui-red)')
  expect(noticeAccent({ key: 'credits.restored', text: '✓ restored' })).toBe('var(--ui-green)')
  expect(noticeAccent({ key: 'credits.grant_spent', text: '• Grant spent' })).toBeUndefined()
  expect(noticeAccent(undefined)).toBeUndefined()
})

test('noticeToToast attaches the band accent to the toast', () => {
  expect(noticeToToast(usage({ level: 'warn', text: "⚠ You've used $15.00 of your $20.00 cap" }))?.accentColor).toBe(
    'var(--ui-orange)'
  )
  expect(noticeToToast(usage({ text: "• You've used $10.00 of your $20.00 cap" }))?.accentColor).toBeUndefined()
})

// ── show / clear: rendered through the notifications store ────────────────────

test('showAgentNotice renders a toast; empty text is a no-op', () => {
  showAgentNotice(usage())
  expect($notifications.get()).toHaveLength(1)
  expect($notifications.get()[0]?.id).toBe('credits.usage')

  showAgentNotice({ text: '' })
  expect($notifications.get()).toHaveLength(1)
})

test('re-emitting the same key replaces the toast instead of stacking (50→75→90)', () => {
  showAgentNotice(usage({ level: 'info', text: "• You've used $10.00 of your $20.00 cap" }))
  showAgentNotice(usage({ level: 'warn', text: "⚠ You've used $15.00 of your $20.00 cap" }))
  showAgentNotice(usage({ level: 'warn', text: "⚠ You've used $18.00 of your $20.00 cap" }))

  const toasts = $notifications.get().filter(item => item.id === 'credits.usage')
  expect(toasts).toHaveLength(1)
  expect(toasts[0]?.message).toBe("You've used $18.00 of your $20.00 cap")
  expect(toasts[0]?.kind).toBe('warning')
})

test('clearAgentNotice dismisses only the matching key', () => {
  showAgentNotice(usage())
  showAgentNotice({ key: 'credits.depleted', kind: 'sticky', level: 'error', text: '✕ paused' })
  expect($notifications.get()).toHaveLength(2)

  clearAgentNotice('credits.usage')
  const ids = $notifications.get().map(item => item.id)
  expect(ids).toContain('credits.depleted')
  expect(ids).not.toContain('credits.usage')

  clearAgentNotice(undefined)
  expect($notifications.get()).toHaveLength(1)
})

// ── nativeNoticeInput: only the urgent credit pair breaks through the OS ──────

test('only credits.depleted and credits.restored map to a native notification', () => {
  expect(nativeNoticeInput(usage({ key: 'credits.usage' }), 'Credits')).toBeNull()
  expect(nativeNoticeInput(usage({ key: 'credits.grant_spent' }), 'Credits')).toBeNull()
  expect(nativeNoticeInput({ text: 'x', key: undefined }, 'Credits')).toBeNull()
  expect(nativeNoticeInput({ text: '', key: 'credits.depleted' }, 'Credits')).toBeNull()
})

test('the urgent pair maps to a global native input carrying the text as its body', () => {
  const depleted = nativeNoticeInput(
    { key: 'credits.depleted', kind: 'sticky', level: 'error', text: '✕ Credit access paused · run /topup to top up' },
    'Credits'
  )

  expect(depleted).toEqual({
    body: '✕ Credit access paused · run /topup to top up',
    global: true,
    kind: 'credits',
    title: 'Credits'
  })

  const restored = nativeNoticeInput(
    { key: 'credits.restored', kind: 'ttl', level: 'success', text: '✓ Credit access restored', ttl_ms: 8000 },
    'Credits'
  )

  expect(restored?.kind).toBe('credits')
  expect(restored?.body).toBe('✓ Credit access restored')
})
