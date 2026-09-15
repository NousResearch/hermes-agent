export type FormulaValue = Array<boolean | number | string> | boolean | number | string

export type FormulaContext = {
  currentSheet: string
  get: (sheet: string, col: number, row: number) => FormulaValue
}

class FormulaError extends Error {
  constructor(message = '#VALUE!') {
    super(message)
    this.name = 'FormulaError'
  }
}

type Token =
  | { type: 'number'; value: number }
  | { type: 'string'; value: string }
  | { type: 'ref'; sheet: string; col: number; row: number }
  | { type: 'range'; sheet: string; c1: number; r1: number; c2: number; r2: number }
  | { type: 'func'; name: string }
  | { type: 'op'; value: string }
  | { type: 'comma' }
  | { type: 'lparen' }
  | { type: 'rparen' }

export function evaluateFormula(formula: string, ctx: FormulaContext): FormulaValue {
  const tokens = tokenize(formula.replace(/^\s*=/, ''), ctx.currentSheet)
  const parser = new Parser(tokens, ctx)
  const value = parser.parseExpr()
  parser.expectEnd()

  return value
}

export function formulaValueToRaw(value: FormulaValue): string {
  if (Array.isArray(value)) {
    return value.length ? formulaValueToRaw(value[0] ?? 0) : ''
  }

  if (typeof value === 'boolean') {
    return value ? 'TRUE' : 'FALSE'
  }

  if (typeof value === 'number') {
    return Number.isFinite(value) ? String(roundNumeric(value)) : '#DIV/0!'
  }

  return value
}

function roundNumeric(value: number) {
  return Math.abs(value) < 1e-12 ? 0 : Math.round(value * 1e12) / 1e12
}

function tokenize(source: string, defaultSheet: string): Token[] {
  const tokens: Token[] = []
  let index = 0

  const peek = () => source[index] || ''
  const starts = (value: string) => source.startsWith(value, index)

  while (index < source.length) {
    const char = peek()

    if (/\s/.test(char)) {
      index += 1

      continue
    }

    if (starts('<>') || starts('<=') || starts('>=')) {
      tokens.push({ type: 'op', value: source.slice(index, index + 2) })
      index += 2

      continue
    }

    if ('+-*/=<>&'.includes(char)) {
      tokens.push({ type: 'op', value: char })
      index += 1

      continue
    }

    if (char === ',') {
      tokens.push({ type: 'comma' })
      index += 1

      continue
    }

    if (char === '(') {
      tokens.push({ type: 'lparen' })
      index += 1

      continue
    }

    if (char === ')') {
      tokens.push({ type: 'rparen' })
      index += 1

      continue
    }

    if (char === '"') {
      index += 1
      let value = ''

      while (index < source.length && source[index] !== '"') {
        value += source[index]
        index += 1
      }

      index += 1
      tokens.push({ type: 'string', value })

      continue
    }

    const sheetRef = readSheetRef()

    if (sheetRef) {
      tokens.push(sheetRef)

      continue
    }

    if (/[0-9.]/.test(char)) {
      const match = /[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?/.exec(source.slice(index))

      if (!match) {
        throw new FormulaError()
      }

      tokens.push({ type: 'number', value: Number(match[0]) })
      index += match[0].length

      continue
    }

    if (/[A-Za-z_]/.test(char)) {
      const match = /[A-Za-z_][A-Za-z0-9_.]*/.exec(source.slice(index))
      const name = match?.[0] || ''
      index += name.length

      if (peek() === '(') {
        tokens.push({ type: 'func', name: name.toUpperCase() })

        continue
      }

      throw new FormulaError()
    }

    throw new FormulaError()
  }

  return tokens

  function readSheetRef(): Token | null {
    let cursor = index
    let sheet = defaultSheet

    if (source[cursor] === "'") {
      cursor += 1
      let name = ''

      while (cursor < source.length && source[cursor] !== "'") {
        name += source[cursor]
        cursor += 1
      }

      if (source[cursor] !== "'") {
        return null
      }

      cursor += 1

      if (source[cursor] !== '!') {
        return null
      }

      sheet = name
      cursor += 1
    } else {
      const nameMatch = /^[A-Za-z_][A-Za-z0-9_.]*!/.exec(source.slice(cursor))

      if (nameMatch) {
        sheet = nameMatch[0].slice(0, -1)
        cursor += nameMatch[0].length
      }
    }

    const refMatch = /^\$?([A-Za-z]+)\$?(\d+)/.exec(source.slice(cursor))

    if (!refMatch) {
      return null
    }

    const first = parseA1(refMatch[1], refMatch[2])
    cursor += refMatch[0].length

    if (source[cursor] === ':') {
      const secondMatch = /^:\$?([A-Za-z]+)\$?(\d+)/.exec(source.slice(cursor))

      if (!secondMatch) {
        return null
      }

      const second = parseA1(secondMatch[1], secondMatch[2])
      index = cursor + secondMatch[0].length

      return {
        type: 'range',
        sheet,
        c1: Math.min(first.col, second.col),
        r1: Math.min(first.row, second.row),
        c2: Math.max(first.col, second.col),
        r2: Math.max(first.row, second.row)
      }
    }

    index = cursor

    return { type: 'ref', sheet, col: first.col, row: first.row }
  }
}

function parseA1(letters: string, digits: string) {
  let col = 0

  for (const char of letters.toUpperCase()) {
    col = col * 26 + (char.charCodeAt(0) - 64)
  }

  return { col: col - 1, row: Number(digits) - 1 }
}

class Parser {
  private index = 0

  constructor(
    private readonly tokens: Token[],
    private readonly ctx: FormulaContext
  ) {}

  expectEnd() {
    if (this.index < this.tokens.length) {
      throw new FormulaError()
    }
  }

  parseExpr(): FormulaValue {
    return this.parseCompare()
  }

  private peek(): Token | undefined {
    return this.tokens[this.index]
  }

  private take(): Token {
    return this.tokens[this.index++]
  }

  private parseCompare(): FormulaValue {
    let left = this.parseAdd()
    const token = this.peek()

    if (token?.type !== 'op' || !['=', '<>', '<', '>', '<=', '>='].includes(token.value)) {
      return left
    }

    this.take()
    const right = this.parseAdd()

    return compareValues(left, token.value, right)
  }

  private parseAdd(): FormulaValue {
    let left = this.parseMul()

    while (this.peek()?.type === 'op' && (this.peek() as { value: string }).value && '+-'.includes((this.peek() as { value: string }).value)) {
      const op = (this.take() as { value: string }).value
      const right = this.parseMul()
      left = op === '+' ? asNumber(left) + asNumber(right) : asNumber(left) - asNumber(right)
    }

    return left
  }

  private parseMul(): FormulaValue {
    let left = this.parseUnary()

    while (this.peek()?.type === 'op' && (this.peek() as { value: string }).value && '*/'.includes((this.peek() as { value: string }).value)) {
      const op = (this.take() as { value: string }).value
      const right = this.parseUnary()

      if (op === '/') {
        const denom = asNumber(right)

        if (denom === 0) {
          left = '#DIV/0!'
        } else {
          left = asNumber(left) / denom
        }
      } else {
        left = asNumber(left) * asNumber(right)
      }
    }

    return left
  }

  private parseUnary(): FormulaValue {
    if (this.peek()?.type === 'op' && (this.peek() as { value: string }).value === '-') {
      this.take()

      return -asNumber(this.parseUnary())
    }

    if (this.peek()?.type === 'op' && (this.peek() as { value: string }).value === '+') {
      this.take()

      return this.parseUnary()
    }

    return this.parsePrimary()
  }

  private parsePrimary(): FormulaValue {
    const token = this.peek()

    if (!token) {
      throw new FormulaError()
    }

    if (token.type === 'number') {
      this.take()

      return token.value
    }

    if (token.type === 'string') {
      this.take()

      return token.value
    }

    if (token.type === 'ref') {
      this.take()

      return this.ctx.get(token.sheet, token.col, token.row)
    }

    if (token.type === 'range') {
      this.take()

      return this.readRange(token)
    }

    if (token.type === 'func') {
      this.take()
      this.expect('lparen')
      const args: FormulaValue[] = []

      if (this.peek()?.type !== 'rparen') {
        args.push(this.parseExpr())

        while (this.peek()?.type === 'comma') {
          this.take()
          args.push(this.parseExpr())
        }
      }

      this.expect('rparen')

      return callFunction(token.name, args)
    }

    if (token.type === 'lparen') {
      this.take()
      const value = this.parseExpr()
      this.expect('rparen')

      return value
    }

    throw new FormulaError()
  }

  private expect(type: Token['type']) {
    if (this.peek()?.type !== type) {
      throw new FormulaError()
    }

    this.take()
  }

  private readRange(token: Extract<Token, { type: 'range' }>): Array<boolean | number | string> {
    const values: Array<boolean | number | string> = []

    for (let row = token.r1; row <= token.r2; row += 1) {
      for (let col = token.c1; col <= token.c2; col += 1) {
        const value = this.ctx.get(token.sheet, col, row)
        values.push(Array.isArray(value) ? (value[0] ?? 0) : value)
      }
    }

    return values
  }
}

function callFunction(name: string, args: FormulaValue[]): FormulaValue {
  switch (name) {
    case 'IF':
      return isTruthy(args[0]) ? (args[1] ?? 0) : (args[2] ?? 0)

    case 'IFERROR':
      return isError(args[0]) ? (args[1] ?? 0) : (args[0] ?? 0)

    case 'ABS':
      return Array.isArray(args[0]) ? args[0].map(value => Math.abs(asNumber(value))) : Math.abs(asNumber(args[0]))

    case 'SUM':
      return flatten(args).reduce((sum: number, value) => sum + asNumber(value), 0)
    case 'AVERAGE': {
      const values = flatten(args).map(asNumber)

      return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0
    }

    case 'MIN':
      return Math.min(...flatten(args).map(asNumber))

    case 'MAX':
      return Math.max(...flatten(args).map(asNumber))

    case 'COUNTIFS':
      return countIfs(args)

    case 'SUMIFS':
      return sumIfs(args)

    case 'SUMPRODUCT':
      return sumProduct(args)

    default:
      throw new FormulaError()
  }
}

function flatten(args: FormulaValue[]): FormulaValue[] {
  return args.flatMap(arg => (Array.isArray(arg) ? arg : [arg]))
}

function countIfs(args: FormulaValue[]): number {
  if (args.length < 2 || args.length % 2 !== 0) {
    throw new FormulaError()
  }

  const pairs: { range: FormulaValue[]; criteria: FormulaValue }[] = []

  for (let index = 0; index < args.length; index += 2) {
    pairs.push({ range: asArray(args[index]), criteria: args[index + 1] ?? '' })
  }

  const length = pairs[0]?.range.length || 0
  let count = 0

  for (let index = 0; index < length; index += 1) {
    if (pairs.every(pair => matchesCriteria(pair.range[index], pair.criteria))) {
      count += 1
    }
  }

  return count
}

function sumIfs(args: FormulaValue[]): number {
  if (args.length < 3 || args.length % 2 === 0) {
    throw new FormulaError()
  }

  const sumRange = asArray(args[0])
  const pairs: { range: FormulaValue[]; criteria: FormulaValue }[] = []

  for (let index = 1; index < args.length; index += 2) {
    pairs.push({ range: asArray(args[index]), criteria: args[index + 1] ?? '' })
  }

  let total = 0

  for (let index = 0; index < sumRange.length; index += 1) {
    if (pairs.every(pair => matchesCriteria(pair.range[index], pair.criteria))) {
      total += asNumber(sumRange[index])
    }
  }

  return total
}

function sumProduct(args: FormulaValue[]): number {
  const arrays = args.map(asArray)
  const length = arrays[0]?.length || 0
  let total = 0

  for (let index = 0; index < length; index += 1) {
    total += arrays.reduce((product, values) => product * asNumber(values[index]), 1)
  }

  return total
}

function asArray(value: FormulaValue | undefined): FormulaValue[] {
  if (value === undefined) {
    return []
  }

  return Array.isArray(value) ? value : [value]
}

function matchesCriteria(value: FormulaValue | undefined, criteria: FormulaValue): boolean {
  const cell = primitive(value ?? '')
  const rule = primitive(criteria)

  if (typeof rule === 'string') {
    const inequality = /^(<>|<=|>=|<|>)(.*)$/.exec(rule)

    if (inequality) {
      const op = inequality[1]
      const target = inequality[2]

      return compareValues(cell, op, coerceLike(cell, target)) === true
    }

    if (rule === '') {
      return cell === '' || cell === 0
    }

    return String(cell).toLowerCase() === rule.toLowerCase() || (isNumeric(cell) && isNumeric(rule) && Number(cell) === Number(rule))
  }

  return compareValues(cell, '=', rule) === true
}

function coerceLike(cell: FormulaValue, raw: string): FormulaValue {
  if (raw === '') {
    return ''
  }

  if (isNumeric(cell) && raw !== '' && Number.isFinite(Number(raw))) {
    return Number(raw)
  }

  return raw
}

function compareValues(left: FormulaValue, op: string, right: FormulaValue): boolean {
  if (op === '=') {
    if (left === '' && (right === '' || right === 0)) {
      return true
    }

    if (right === '' && (left === '' || left === 0)) {
      return true
    }

    if (typeof left === 'string' || typeof right === 'string') {
      return String(left).toLowerCase() === String(right).toLowerCase()
    }

    return asNumber(left) === asNumber(right)
  }

  if (op === '<>') {
    return !compareValues(left, '=', right)
  }

  const l = asNumber(left)
  const r = asNumber(right)

  if (op === '<') {
    return l < r
  }

  if (op === '>') {
    return l > r
  }

  if (op === '<=') {
    return l <= r
  }

  return l >= r
}

function isTruthy(value: FormulaValue | undefined): boolean {
  if (Array.isArray(value)) {
    return isTruthy(value[0] ?? 0)
  }

  if (typeof value === 'boolean') {
    return value
  }

  if (typeof value === 'number') {
    return value !== 0
  }

  return value !== '' && value !== 'FALSE'
}

function isError(value: FormulaValue | undefined): boolean {
  return typeof value === 'string' && value.startsWith('#')
}

function primitive(value: FormulaValue): Exclude<FormulaValue, number[]> {
  return Array.isArray(value) ? (value[0] ?? 0) : value
}

function isNumeric(value: FormulaValue): boolean {
  const primitiveValue = primitive(value)

  return typeof primitiveValue === 'number' || (typeof primitiveValue === 'string' && primitiveValue !== '' && Number.isFinite(Number(primitiveValue)))
}

function asNumber(value: FormulaValue | undefined): number {
  if (value === undefined || value === '' || value === false) {
    return 0
  }

  if (value === true) {
    return 1
  }

  if (Array.isArray(value)) {
    return asNumber(value[0])
  }

  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new FormulaError('#DIV/0!')
    }

    return value
  }

  if (typeof value === 'string' && value.startsWith('#')) {
    throw new FormulaError(value)
  }

  const number = Number(value)

  return Number.isFinite(number) ? number : 0
}
