import { expect, it } from 'vitest'

import { chosenChatFontFamily, resolveChatFontFamily } from './chat-font'

const THEME = "'Theme Sans', sans-serif"

it('leads the theme stack with the chosen face, quoted, and leaves the theme alone when none is chosen', () => {
  expect(resolveChatFontFamily(' OpenDyslexic ', THEME)).toBe(`'OpenDyslexic', ${THEME}`)
  expect(resolveChatFontFamily('', THEME)).toBe(THEME)
})

it('names the choice on its own for a shell with faces of its own, and nothing when none is chosen', () => {
  expect(chosenChatFontFamily('Atkinson Hyperlegible')).toBe("'Atkinson Hyperlegible'")
  // An authored stack is taken as written.
  expect(chosenChatFontFamily('Lexend, "Noto Sans"')).toBe('Lexend, "Noto Sans"')
  expect(chosenChatFontFamily('   ')).toBeNull()
  expect(chosenChatFontFamily(undefined)).toBeNull()
})
