// First-strong direction detection, same heuristic as dir="auto" but over a
// caller-chosen slice of the text. Callers strip code spans before asking:
// a technical RTL message usually *starts* with a command or identifier,
// which would flip first-strong to LTR even though the sentence is
// Hebrew/Arabic. U+0590-U+08FF is RTL scripts end to end (Hebrew, Arabic,
// Syriac, Thaana, NKo, Samaritan, Mandaic and their extensions).
const RTL_CHAR = /[\u0590-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]/
const FIRST_LETTER = /\p{L}/u

export function textDirection(text: string): 'ltr' | 'rtl' | null {
  const first = text.match(FIRST_LETTER)

  if (!first) {
    return null
  }

  return RTL_CHAR.test(first[0]) ? 'rtl' : 'ltr'
}
