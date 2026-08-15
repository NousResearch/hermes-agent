import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createComposerSelectionMenuItem } from './composer-selection'

test('non-editable selection is forwarded to the composer unchanged', () => {
  const sent: Array<{ text: string; x: number; y: number }> = []
  const selection = 'first line\nsecond line'

  const item = createComposerSelectionMenuItem(
    { canCompose: true, isEditable: false, selectionText: selection, x: 42, y: 17 },
    payload => sent.push(payload),
    () => assert.fail('selection send should not fail')
  )

  assert.equal(item?.label, 'Send Selection to Composer')
  item?.click()
  assert.deepEqual(sent, [{ text: selection, x: 42, y: 17 }])
})

test('editable selections retain the native edit menu without a composer action', () => {
  const item = createComposerSelectionMenuItem(
    { canCompose: true, isEditable: true, selectionText: 'selected input text', x: 0, y: 0 },
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
        selectionText: 'selected message text',
        x: 0,
        y: 0
      },
      () => assert.fail('watch selection must not be forwarded'),
      () => assert.fail('watch selection must not report a send failure')
    ),
    null
  )
})
