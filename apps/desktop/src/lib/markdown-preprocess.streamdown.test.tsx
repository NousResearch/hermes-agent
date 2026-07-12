// Streamdown-integration regression tests for issue #38786: a markdown image
// whose src is a raw local path (image_generate on Windows returns C:\… paths
// the model embeds verbatim) can never render as <img> — Streamdown's
// sanitize pass strips the non-http(s) src and its harden pass then replaces
// the node with an "[Image blocked: …]" placeholder. preprocessMarkdown
// rewrites such embeds to `#media:` links, and these tests pin the assumption
// that form survives the real sanitize + harden rehype pipeline and reaches
// the `a` component (mapped to MarkdownLink → MediaAttachment in the app).
import { render } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { Streamdown } from 'streamdown'
import { describe, expect, it } from 'vitest'

import { preprocessMarkdown } from '@/lib/markdown-preprocess'
import { mediaPathFromMarkdownHref } from '@/lib/media'

const ISSUE_MARKDOWN =
  'Here you go!\n\n![cute kitten](C:\\Users\\me\\AppData\\Local\\hermes\\cache\\images\\generated.jpg)'

describe('issue #38786 end-to-end', () => {
  it('reproduces the blocked placeholder without the preprocess rewrite', () => {
    const { container } = render(<Streamdown mode="static">{ISSUE_MARKDOWN}</Streamdown>)

    expect(container.textContent).toContain('[Image blocked: cute kitten]')
  })

  it('delivers the intact #media: href to the a component after the rewrite', () => {
    const seen: (string | undefined)[] = []

    const CaptureLink = ({ children, href }: ComponentProps<'a'>) => {
      seen.push(href)

      return <a href={href}>{children}</a>
    }

    // Same cast the app uses for its own overrides (markdown-text.tsx casts to
    // StreamdownTextComponents) — the Components type's index signature and
    // element-specific props don't unify for plain function components.
    const components = { a: CaptureLink } as ComponentProps<typeof Streamdown>['components']

    const { container } = render(
      <Streamdown components={components} mode="static">
        {preprocessMarkdown(ISSUE_MARKDOWN)}
      </Streamdown>
    )

    expect(container.textContent).not.toContain('[Image blocked')
    expect(container.textContent).toContain('Image: generated.jpg')
    expect(seen).toHaveLength(1)
    expect(mediaPathFromMarkdownHref(seen[0])).toBe(
      'C:\\Users\\me\\AppData\\Local\\hermes\\cache\\images\\generated.jpg'
    )
  })
})
