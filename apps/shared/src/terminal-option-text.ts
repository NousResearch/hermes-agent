// macOS Option in an xterm.js terminal that runs with `macOptionIsMeta: true`.
//
// Meta mode turns every Option chord into ESC + base key, which is what shell
// users on a US layout want (Option+B / Option+F move by word). On many other
// layouts Option is how ASCII punctuation is typed at all: German Option+L is
// `@`, Option+7 is `|`, Option+5 / 6 / 8 / 9 are `[` `]` `{` `}`. Meta mode
// sends ESC l instead of `@`, so those characters cannot be typed.
//
// Rule: an Option chord whose composed character is printable ASCII
// punctuation is typed as that character; every other Option chord stays Meta.
// A US layout never composes ASCII punctuation with Option (it yields glyphs
// like `∫` or `ß`, or a dead key), so US behaviour does not change. Letters
// and digits are excluded so a layout echoing the base key keeps Meta.

export interface OptionKeyEventLike {
  type: string
  key: string
  altKey: boolean
  ctrlKey: boolean
  metaKey: boolean
  isComposing?: boolean
}

const ASCII_PUNCTUATION = /^[!-/:-@[-`{-~]$/

export function macOptionComposedText(event: OptionKeyEventLike, isMac: boolean): string | null {
  if (!isMac || event.type !== 'keydown' || event.isComposing) {
    return null
  }

  if (!event.altKey || event.ctrlKey || event.metaKey) {
    return null
  }

  return ASCII_PUNCTUATION.test(event.key) ? event.key : null
}
