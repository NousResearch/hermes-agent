import { describe, expect, it } from 'vitest'

import { evaluateFormula, type FormulaValue } from './ooxml-formula'

function ctx(cells: Record<string, FormulaValue>, sheet = 'Sheet1') {
  return {
    currentSheet: sheet,
    get(name: string, col: number, row: number): FormulaValue {
      return cells[`${name}!${col}:${row}`] ?? ''
    }
  }
}

describe('evaluateFormula', () => {
  it('adds cell refs', () => {
    expect(evaluateFormula('A1+B1', ctx({ 'Sheet1!0:0': 2, 'Sheet1!1:0': 3 }))).toBe(5)
  })

  it('treats blank cells as empty in IF', () => {
    expect(evaluateFormula('IF(G2="",0,G2)-IF(F2="",0,F2)', ctx({}))).toBe(0)
  })

  it('evaluates SUMIFS across sheets', () => {
    const cells: Record<string, FormulaValue> = {
      'Pivot!0:5': 'Cat',
      'Transactions!7:1': 10,
      'Transactions!8:1': 'Cat',
      'Transactions!10:1': 'Inflow',
      'Transactions!7:2': 4,
      'Transactions!8:2': 'Cat',
      'Transactions!10:2': 'Outflow'
    }

    expect(
      evaluateFormula(
        'SUMIFS(Transactions!$H$2:$H$3,Transactions!$I$2:$I$3,$A6,Transactions!$K$2:$K$3,"Inflow")',
        ctx(cells, 'Pivot')
      )
    ).toBe(10)
  })
})
