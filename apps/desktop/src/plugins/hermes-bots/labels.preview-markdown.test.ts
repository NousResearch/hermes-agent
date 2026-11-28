import { describe, expect, it } from 'vitest'

import { stripPreviewMarkdown } from './labels'

// Roster previews arrive already truncated (the gateway sends an 80-char
// excerpt + "..."). When the cut lands inside an emphasis span, the closing
// marker is gone and the paired-marker rules cannot match, so the raw opener
// leaked into the row: "**Yes. Your existing credits can...".
describe('stripPreviewMarkdown on truncated previews', () => {
  it('drops a bold opener whose closing marker was truncated away', () => {
    expect(stripPreviewMarkdown('**Yes. Your existing credits can be used for the new plan, and they carry over a...')).toBe(
      'Yes. Your existing credits can be used for the new plan, and they carry over a...'
    )
    expect(stripPreviewMarkdown("**This isn't a port conflict. Tailscale serve is already bound to...")).toBe(
      "This isn't a port conflict. Tailscale serve is already bound to..."
    )
  })

  it('drops other unpaired openers left by a cut', () => {
    expect(stripPreviewMarkdown('__Heads up: the deploy is still runn...')).toBe('Heads up: the deploy is still runn...')
    expect(stripPreviewMarkdown('~~old plan was to migrate every...')).toBe('old plan was to migrate every...')
    expect(stripPreviewMarkdown('run `hermes update --channel be...')).toBe('run hermes update --channel be...')
    expect(stripPreviewMarkdown('Done. **Next: rotate the keys and re...')).toBe('Done. Next: rotate the keys and re...')
  })

  it('keeps balanced markdown and plain arithmetic working as before', () => {
    expect(stripPreviewMarkdown('**Yes.** Your credits carry over.')).toBe('Yes. Your credits carry over.')
    expect(stripPreviewMarkdown('**Plan**: ship the `thing`')).toBe('Plan: ship the thing')
    expect(stripPreviewMarkdown('2 * 3 = 6')).toBe('2 * 3 = 6')
  })
})
