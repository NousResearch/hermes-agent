import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createComposerSelectionMenuItem } from './composer-selection'

test('non-editable selection is forwarded to the composer unchanged', () => {
  const sent: string[] = []
  const selection = 'first line\nsecond line'

  const item = createComposerSelectionMenuItem(
    { canCompose: true, isEditable: false, selectionText: selection },
    text => sent.push(text),
    () => assert.fail('selection send should not fail')
  )

  assert.equal(item?.label, 'Send Selection to Composer')
  item?.click()
  assert.deepEqual(sent, [selection])
})

test('editable selections retain the native edit menu without a composer action', () => {
  const item = createComposerSelectionMenuItem(
    { canCompose: true, isEditable: true, selectionText: 'selected input text' },
    () => assert.fail('editable selection must not be forwarded'),
    () => assert.fail('editable selection must not report a send failure')
  )

  assert.equal(item, null)
})

test('windows without a composer do not offer the selection action', () => {
  assert.equal(
    createComposerSelectionMenuItem(
      {
        canCompose: false,
        isEditable: false,
        selectionText: 'selected message text'
      },
      () => assert.fail('watch selection must not be forwarded'),
      () => assert.fail('watch selection must not report a send failure')
    ),
    null
  )
})
