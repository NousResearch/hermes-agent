import { describe, expect, it } from 'vitest'

import { linkifyFilePaths } from './linkify-file-paths'

describe('linkifyFilePaths', () => {
  it('links an absolute path with an extension', () => {
    expect(linkifyFilePaths('见 docs/requirements/R-040 请看 /Users/echo/notes.md 文件')).toContain(
      '[/Users/echo/notes.md](#media:%2FUsers%2Fecho%2Fnotes.md)'
    )
  })

  it('links multiple paths in one chunk', () => {
    const out = linkifyFilePaths('/tmp/a.ts 与 /tmp/b.json 都要看')
    expect(out).toContain('[/tmp/a.ts](#media:%2Ftmp%2Fa.ts)')
    expect(out).toContain('[/tmp/b.json](#media:%2Ftmp%2Fb.json)')
  })

  it('URL-encodes non-ASCII in paths', () => {
    const out = linkifyFilePaths('/Users/echo/我的文件.md')
    expect(out).toContain('#media:%2FUsers%2Fecho%2F%E6%88%91%E7%9A%84%E6%96%87%E4%BB%B6.md')
  })

  it('leaves code fences untouched', () => {
    const src = '前置说明\n```\n/usr/bin/evil.sh\n```\n后置 /usr/bin/ok.sh'
    const out = linkifyFilePaths(src)
    expect(out).toContain('/usr/bin/evil.sh') // raw inside fence
    expect(out).not.toContain('#media:%2Fusr%2Fbin%2Fevil.sh')
    expect(out).toContain('[/usr/bin/ok.sh](#media:%2Fusr%2Fbin%2Fok.sh)')
  })

  it('does not relink an existing markdown link target', () => {
    const src = '[文档](/Users/echo/doc.md)'
    expect(linkifyFilePaths(src)).toBe(src)
  })

  it('ignores paths without an extension and relative paths', () => {
    const src = '目录 /Users/echo/notes 与相对路径 docs/README.md 不动'
    expect(linkifyFilePaths(src)).toBe(src)
  })
})
