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

  it('links project-relative paths when cwd is provided', () => {
    const src = '见 docs/requirements/R-039.md 和 openspec/roadmap.md'
    const out = linkifyFilePaths(src, '/Users/echo/Coding/sdata/sdata-tempo')
    expect(out).toContain(
      '[docs/requirements/R-039.md](#media:%2FUsers%2Fecho%2FCoding%2Fsdata%2Fsdata-tempo%2Fdocs%2Frequirements%2FR-039.md)'
    )
    expect(out).toContain(
      '[openspec/roadmap.md](#media:%2FUsers%2Fecho%2FCoding%2Fsdata%2Fsdata-tempo%2Fopenspec%2Froadmap.md)'
    )
  })

  it('resolves ./ and ../ relative paths against cwd', () => {
    expect(linkifyFilePaths('./src/main.ts', '/repo/pkg')).toContain(
      '[./src/main.ts](#media:%2Frepo%2Fpkg%2Fsrc%2Fmain.ts)'
    )
    expect(linkifyFilePaths('../shared/types.ts', '/repo/pkg/src')).toContain(
      '[../shared/types.ts](#media:%2Frepo%2Fpkg%2Fshared%2Ftypes.ts)'
    )
  })

  it('does not link relative paths without cwd, and skips URL path segments', () => {
    expect(linkifyFilePaths('docs/guide.md')).toBe('docs/guide.md')
    expect(linkifyFilePaths('见 https://github.com/user/repo/file.md', '/repo')).toBe(
      '见 https://github.com/user/repo/file.md'
    )
  })

  it('leaves inline code (backtick) paths untouched', () => {
    const src = '运行 `docs/guide.md` 和 `/tmp/a.md` 都不动'
    expect(linkifyFilePaths(src, '/repo')).toBe(src)
  })

  it('links plain-text relative paths even next to inline-code ones', () => {
    const src = '看 docs/guide.md 与代码里的 `docs/guide.md`'
    const out = linkifyFilePaths(src, '/repo')
    expect(out).toContain('[docs/guide.md](#media:%2Frepo%2Fdocs%2Fguide.md)')
    expect(out).toContain('`docs/guide.md`')
  })
})
