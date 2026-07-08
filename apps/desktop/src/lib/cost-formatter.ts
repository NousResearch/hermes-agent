/**
 * Cost and quota formatting utilities for the desktop status bar.
 * Follows the pattern of lib/statusbar.ts — pure functions, no side effects.
 */

export function formatCost(value: number | null | undefined): string {
  const n = Number(value)
  if (!Number.isFinite(n) || n < 0) return '$0.00'
  // For very small values (< $0.01), show more precision
  if (n > 0 && n < 0.01) {
    return '$' + n.toFixed(4).replace(/\.?0+$/, '')
  }
  return n.toLocaleString('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 })
}

export function quotaColorClass(pct: number | undefined): string {
  if (pct == null || pct < 80) return ''
  if (pct < 95) return 'text-amber-500'
  return 'text-red-500'
}

export function quotaLabel(pct: number | undefined, _reset?: string | undefined): string {
  if (pct == null) return ''
  return `quota ${Math.round(pct)}%`
}