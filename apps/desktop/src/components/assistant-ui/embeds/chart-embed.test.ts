import { describe, expect, it } from 'vitest'
import { parseChartSpec, getNiceMax } from './chart-embed'

describe('parseChartSpec', () => {
  it('parses valid JSON chart specifications', () => {
    const jsonSpec = JSON.stringify({
      type: 'area',
      title: 'Cost Scaling',
      unit: '$',
      labels: ['1', '10', '50'],
      series: [{ name: 'Cost', values: [39, 34.5, 28.5], color: '#38bdf8' }]
    })
    const parsed = parseChartSpec(jsonSpec)
    expect(parsed).not.toBeNull()
    expect(parsed?.type).toBe('area')
    expect(parsed?.title).toBe('Cost Scaling')
    expect(parsed?.unit).toBe('$')
    expect(parsed?.labels).toEqual(['1', '10', '50'])
    expect(parsed?.series?.[0].values).toEqual([39, 34.5, 28.5])
  })

  it('parses shorthand key-value chart specifications', () => {
    const textSpec = `
      type: bar
      title: Margin Comparison
      unit: %
      Wholesale: 76.7
      POD: 55.3
    `
    const parsed = parseChartSpec(textSpec)
    expect(parsed).not.toBeNull()
    expect(parsed?.type).toBe('bar')
    expect(parsed?.title).toBe('Margin Comparison')
    expect(parsed?.labels).toEqual(['Wholesale', 'POD'])
    expect(parsed?.series?.[0].values).toEqual([76.7, 55.3])
  })

  it('handles pie/donut and scatter specifications', () => {
    const scatterSpec = JSON.stringify({
      type: 'scatter',
      title: 'Density vs Price',
      points: [
        { x: 280, y: 12.5, label: 'Tee' },
        { x: 450, y: 22.0, label: 'Hoodie' }
      ]
    })
    const parsed = parseChartSpec(scatterSpec)
    expect(parsed).not.toBeNull()
    expect(parsed?.type).toBe('scatter')
    expect(parsed?.points?.length).toBe(2)
  })

  it('returns null for malformed or empty inputs', () => {
    expect(parseChartSpec('')).toBeNull()
    expect(parseChartSpec('not a valid chart spec at all')).toBeNull()
  })
})

describe('getNiceMax', () => {
  it('calculates human-readable upper scale bounds', () => {
    expect(getNiceMax(76.7)).toBe(100)
    expect(getNiceMax(42)).toBe(50)
    expect(getNiceMax(18.5)).toBe(20)
    expect(getNiceMax(980)).toBe(1000)
    expect(getNiceMax(2400)).toBe(2500)
  })
})
