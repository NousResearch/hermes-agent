import { describe, expect, it } from 'vitest'
import { formatCost, quotaColorClass, quotaLabel } from './cost-formatter'

describe('formatCost', () => {
  it('formats zero as $0.00', () => {
    expect(formatCost(0)).toBe('$0.00')
  })
  it('formats small values', () => {
    expect(formatCost(0.0034)).toBe('$0.0034')
  })
  it('formats typical values', () => {
    expect(formatCost(0.05)).toBe('$0.05')
    expect(formatCost(1.23)).toBe('$1.23')
    expect(formatCost(12.5)).toBe('$12.50')
  })
  it('formats large values', () => {
    expect(formatCost(999.99)).toBe('$999.99')
    expect(formatCost(1234.5)).toBe('$1,234.50')
  })
  it('handles negative as $0.00', () => {
    expect(formatCost(-1)).toBe('$0.00')
  })
  it('handles NaN as $0.00', () => {
    expect(formatCost(NaN)).toBe('$0.00')
  })
  it('handles Infinity as $0.00', () => {
    expect(formatCost(Infinity)).toBe('$0.00')
  })
  it('handles undefined as $0.00', () => {
    expect(formatCost(undefined)).toBe('$0.00')
  })
  it('handles null as $0.00', () => {
    expect(formatCost(null)).toBe('$0.00')
  })
})

describe('quotaColorClass', () => {
  it('returns empty for missing quota', () => {
    expect(quotaColorClass(undefined)).toBe('')
  })
  it('returns empty for quota < 80', () => {
    expect(quotaColorClass(50)).toBe('')
    expect(quotaColorClass(79)).toBe('')
  })
  it('returns amber for quota >= 80', () => {
    expect(quotaColorClass(80)).toBe('text-amber-500')
    expect(quotaColorClass(94)).toBe('text-amber-500')
  })
  it('returns red for quota >= 95', () => {
    expect(quotaColorClass(95)).toBe('text-red-500')
    expect(quotaColorClass(100)).toBe('text-red-500')
    expect(quotaColorClass(150)).toBe('text-red-500')
  })
})

describe('quotaLabel', () => {
  it('returns empty for missing quota', () => {
    expect(quotaLabel(undefined, undefined)).toBe('')
  })
  it('returns percentage when no reset info', () => {
    expect(quotaLabel(50, undefined)).toBe('quota 50%')
    expect(quotaLabel(95, undefined)).toBe('quota 95%')
  })
  it('returns percentage with reset detail', () => {
    expect(quotaLabel(50, '5h23m')).toBe('quota 50%')
  })
})