import { describe, expect, it } from 'vitest'

import { localPreviewTarget } from './local-preview'

// [previewKind, language] of the fallback target, the one remote-gateway files use.
function classify(path: string) {
  const target = localPreviewTarget(path)

  return target?.kind === 'file' ? [target.previewKind, target.language] : null
}

describe('local preview targets for native paths that contain # or ?', () => {
  it('keeps the extension of a native path with a literal # or ?', () => {
    expect(classify('/srv/out/report #2.html')).toEqual(['html', 'html'])
    expect(classify('/srv/notes/#ideas.png')).toEqual(['image', 'text'])
    expect(classify('/srv/q?a/chart.png')).toEqual(['image', 'text'])
    expect(classify('C:/Users/me/Docs/#1 plan.pdf')).toEqual(['pdf', 'text'])
    expect(classify('/srv/out/v#2/main.py')).toEqual(['text', 'python'])
  })

  it('still drops a fragment or query that follows the file name', () => {
    expect(classify('/srv/out/report.html#section')).toEqual(['html', 'html'])
    expect(classify('/srv/notes/guide.md?raw=1')).toEqual(['text', 'markdown'])
  })

  it('reads the extension from the file name, not from a dotted folder', () => {
    expect(classify('/srv/v1.2/README')).toEqual(['text', 'text'])
  })
})
