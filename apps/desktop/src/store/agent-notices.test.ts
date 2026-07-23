import { beforeEach, expect, test } from 'vitest'

import {
  type AgentNoticePayload,
  clearAgentNotice,
  nativeNoticeInput,
  noticeToToast,
  showAgentNotice
} from './agent-notices'
import { $notifications, clearNotifications } from './notifications'

function usage(overrides: Partial<AgentNoticePayload> = {}): AgentNoticePayload {
  return {
    key: 'credits.usage',
    kind: 'sticky',
    level: 'info',
    text: '• Credits 50% used · $220.00 cap',
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
  const toast = noticeToToast({ key: 'credits.restored', kind: 'ttl', level: 'success', text: '✓ restored', ttl_ms: 8000 })
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

test('the glyph-bearing text passes through verbatim as the message', () => {
  expect(noticeToToast(usage())?.message).toBe('• Credits 50% used · $220.00 cap')
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
  showAgentNotice(usage({ level: 'info', text: '• Credits 50% used' }))
  showAgentNotice(usage({ level: 'warn', text: '• Credits 75% used' }))
  showAgentNotice(usage({ level: 'warn', text: '• Credits 90% used' }))

  const toasts = $notifications.get().filter(item => item.id === 'credits.usage')
  expect(toasts).toHaveLength(1)
  expect(toasts[0]?.message).toBe('• Credits 90% used')
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
