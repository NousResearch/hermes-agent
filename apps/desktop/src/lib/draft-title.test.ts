import { describe, expect, it } from 'vitest'

import { deriveDraftTitle } from './draft-title'

// Twin of tests/agent/test_title_generator.py::TestDeriveTitleReadsPastMarkdown: the draft
// label the tab shows before the first send must match the backend's instant title.
describe('deriveDraftTitle', () => {
  it.each([
    ['```json\n{"a": 1}\n```', '{"a": 1}'],
    ['```\n```\nwhy is this failing?', 'why is this failing?'],
    ['![screenshot](https://x.example/a.png)\nwhy does this crash?', 'screenshot'],
    ['![](https://img.example/a.png)', ''],
    ['![](https://img.example/a.png)\nwhy does this crash?', 'why does this crash?'],
    ['# Plan for the refactor\nbody', 'Plan for the refactor'],
    ['- [ ] first item\n- second', 'first item'],
    ['**urgent**: fix the `build` step in [CI](https://ci.example/run/1) please', 'urgent: fix the build step in CI please'],
    ['/skin ***really*** dark', 'really dark'],
  ])('reads past markup: %j', (text, expected) => {
    expect(deriveDraftTitle(text)).toBe(expected)
  })

  it.each(['a * b = c and 2*3', 'use *args and **kwargs', 'snake_case_name and file_name.py', '#hashtag not heading'])(
    'keeps non-markup delimiters: %j',
    text => {
      expect(deriveDraftTitle(text)).toBe(text)
    },
  )
})
