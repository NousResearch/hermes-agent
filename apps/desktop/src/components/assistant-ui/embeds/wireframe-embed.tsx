import DOMPurify from 'dompurify'
import { useMemo } from 'react'

import type { RichFenceProps } from './types'

const WIREFRAME_CLASS_PREFIX = 'wf-'

function sanitizeWireframeHtml(code: string): string {
  const clean = DOMPurify.sanitize(code, {
    FORBID_ATTR: ['style'],
    FORBID_TAGS: ['script', 'iframe', 'object', 'embed'],
    USE_PROFILES: { html: true }
  })

  if (typeof document === 'undefined') {
    return clean.replace(/\sclass=(?:"[^"]*"|'[^']*')/gi, '')
  }

  const template = document.createElement('template')
  template.innerHTML = clean

  for (const element of template.content.querySelectorAll('[class]')) {
    const safeClasses = Array.from(element.classList).filter(className => className.startsWith(WIREFRAME_CLASS_PREFIX))

    if (safeClasses.length > 0) {
      element.setAttribute('class', safeClasses.join(' '))
    } else {
      element.removeAttribute('class')
    }
  }

  return template.innerHTML
}

export default function WireframeRenderer({ code }: RichFenceProps) {
  const clean = useMemo(() => sanitizeWireframeHtml(code), [code])

  if (!clean.trim()) {
    return (
      <div className="my-2 rounded-lg border border-dashed border-(--ui-stroke-secondary) bg-muted/20 p-3 text-sm text-muted-foreground">
        Empty wireframe block
      </div>
    )
  }

  return (
    <div className="my-2 overflow-hidden rounded-xl border border-(--ui-stroke-secondary) bg-background shadow-xs">
      <div className="border-b border-(--ui-stroke-tertiary) bg-muted/35 px-3 py-2 text-xs font-semibold tracking-wide text-muted-foreground uppercase">
        Wireframe
      </div>
      <div
        className="max-h-[60dvh] overflow-auto p-3 text-sm [&_.wf-button]:inline-flex [&_.wf-button]:rounded-md [&_.wf-button]:bg-primary/10 [&_.wf-button]:px-2 [&_.wf-button]:py-1 [&_.wf-card]:rounded-lg [&_.wf-card]:border [&_.wf-card]:border-(--ui-stroke-secondary) [&_.wf-card]:bg-card [&_.wf-card]:p-3 [&_.wf-col]:grid [&_.wf-col]:gap-2 [&_.wf-row]:flex [&_.wf-row]:items-center [&_.wf-row]:gap-2"
        dangerouslySetInnerHTML={{ __html: clean }}
        data-testid="wireframe-embed"
      />
    </div>
  )
}
