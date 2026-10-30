import { expect, it } from 'vitest'

import { toChatMessages } from './index'

const BLOCK = [
  '[adviser — advisory only]',
  '- [concern · tool_choice · Architecture] re-read the failing test — decide: fix it',
  '- [nit · general] tidy the helper',
  '[end adviser]'
].join('\n')

it('hydrates an adviser row as a system card, not a user prompt', () => {
  const [message] = toChatMessages([{ role: 'user', content: BLOCK, display_kind: 'adviser', timestamp: 1 }])

  expect(message.role).toBe('system')
  expect(message.adviser?.severity).toBe('concern')
  expect(message.adviser?.label).toBe('concern · tool_choice · Architecture')
  expect(message.adviser?.body).toContain('re-read the failing test')
  expect(message.adviser?.body).not.toContain('[adviser')
  expect(message.adviser?.body).not.toContain('[end adviser]')
  expect(message.adviser?.items).toHaveLength(2)
})

it('lets a blocker listed second raise the card severity', () => {
  const block = [
    '[adviser — advisory only]',
    '- [concern · tool_choice · Architecture] check the import',
    '- [blocker · ai_self_representation_check · default] the test never ran',
    '[end adviser]'
  ].join('\n')
  const [message] = toChatMessages([{ role: 'user', content: block, display_kind: 'adviser', timestamp: 4 }])

  expect(message.adviser?.severity).toBe('blocker')
  expect(message.adviser?.label).toBe('blocker · ai_self_representation_check · default')
  expect(message.adviser?.items.map(item => item.severity)).toEqual(['concern', 'blocker'])
})

it('keeps an unparseable adviser row visible instead of dropping it', () => {
  const [message] = toChatMessages([
    { role: 'user', content: 'plain advice from an older host', display_kind: 'adviser', timestamp: 2 }
  ])

  expect(message.role).toBe('system')
  expect(message.adviser?.severity).toBe('advisory')
  expect(message.adviser?.body).toBe('plain advice from an older host')
})

it('leaves a normal user row alone', () => {
  const [message] = toChatMessages([{ role: 'user', content: 'real prompt', timestamp: 3 }])

  expect(message.role).toBe('user')
  expect(message.adviser).toBeUndefined()
})
