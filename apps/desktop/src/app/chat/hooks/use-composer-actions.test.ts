import { describe, expect, it } from 'vitest'

import { type DroppedFile, partitionDroppedFiles } from './use-composer-actions'

// A Finder/Explorer drop carries a native File handle; an in-app drag (project
// tree, gutter line ref) is path-only. The split decides whether a drop becomes
// an inline @file: ref (in-app, workspace-relative, gateway-resolvable) or goes
// through the upload pipeline (OS drop — absolute local path a remote gateway
// can't read, plus image bytes for vision).
const osDrop = (path: string): DroppedFile => ({ file: new File(['x'], path.split('/').pop() || 'f'), path })
const inAppRef = (path: string, extra: Partial<DroppedFile> = {}): DroppedFile => ({ path, ...extra })

describe('partitionDroppedFiles', () => {
  it('routes File-bearing OS drops to osDrops and path-only in-app drags to inAppRefs', () => {
    const finderPdf = osDrop('/Users/mahmoud/Downloads/DEVIS_signed.pdf')
    const projectFile = inAppRef('src/index.ts')

    const { inAppRefs, osDrops } = partitionDroppedFiles([finderPdf, projectFile])

    expect(osDrops).toEqual([finderPdf])
    expect(inAppRefs).toEqual([projectFile])
  })

  it('treats an OS screenshot drop as an upload target (so it gets byte upload + vision)', () => {
    const screenshot = osDrop('/var/folders/tmp/Screenshot 2026-06-09.png')

    const { inAppRefs, osDrops } = partitionDroppedFiles([screenshot])

    expect(osDrops).toEqual([screenshot])
    expect(inAppRefs).toEqual([])
  })

  it('keeps gutter line-range drags inline (no File handle)', () => {
    const lineRef = inAppRef('src/app.ts', { line: 10, lineEnd: 20 })

    const { inAppRefs, osDrops } = partitionDroppedFiles([lineRef])

    expect(osDrops).toEqual([])
    expect(inAppRefs).toEqual([lineRef])
  })

  it('splits a mixed drop and preserves order within each group', () => {
    const a = inAppRef('a.ts')
    const b = osDrop('/abs/b.pdf')
    const c = inAppRef('c.ts')
    const d = osDrop('/abs/d.png')

    const { inAppRefs, osDrops } = partitionDroppedFiles([a, b, c, d])

    expect(inAppRefs).toEqual([a, c])
    expect(osDrops).toEqual([b, d])
  })

  it('returns empty groups for an empty drop', () => {
    expect(partitionDroppedFiles([])).toEqual({ inAppRefs: [], osDrops: [] })
  })
})
