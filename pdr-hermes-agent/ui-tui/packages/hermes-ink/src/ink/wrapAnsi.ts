import wrapAnsiNpm from 'wrap-ansi'

type WrapAnsiOptions = {
  hard?: boolean
  wordWrap?: boolean
  trim?: boolean
}

const wrapAnsiBun = typeof Bun !== 'undefined' && typeof Bun.wrapAnsi === 'function' ? Bun.wrapAnsi : null

const wrapAnsi: (input: string, columns: number, options?: WrapAnsiOptions) => string = wrapAnsiBun ?? wrapAnsiNpm

export { wrapAnsi }
