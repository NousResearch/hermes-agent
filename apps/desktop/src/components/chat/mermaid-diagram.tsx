'use client'

import type { SyntaxHighlighterProps } from '@assistant-ui/react-streamdown'
import { type FC, memo, useEffect, useId, useRef, useState } from 'react'

import {
  CodeCard,
  CodeCardBody,
  CodeCardHeader,
  CodeCardIcon,
  CodeCardSubtitle,
  CodeCardTitle
} from '@/components/chat/code-card'
import { SyntaxHighlighter } from '@/components/chat/shiki-highlighter'
import { Codicon } from '@/components/ui/codicon'
import { CopyButton } from '@/components/ui/copy-button'
import { useI18n } from '@/i18n'
import { useTheme } from '@/themes/context'

/**
 * Renders ```mermaid fences as diagrams. Registered per-language via
 * `componentsByLanguage` on StreamdownTextPrimitive — the global
 * `SyntaxHighlighter` override replaces streamdown's built-in code dispatcher
 * (the only place its mermaid plugin is consulted), so a plugins-only setup
 * never renders diagrams; this component is the supported escape hatch.
 *
 * While streaming (`defer`) or when the source fails to parse we fall back to
 * the regular code card, so the user always at least sees the source.
 */
interface MermaidDiagramProps extends SyntaxHighlighterProps {
  defer?: boolean
}

const MermaidDiagramImpl: FC<MermaidDiagramProps> = props => {
  const { code, defer = false } = props
  const { t } = useI18n()
  const { renderedMode } = useTheme()
  const [svg, setSvg] = useState('')
  const [error, setError] = useState<string | null>(null)
  // mermaid.render() needs a document-unique element id per call. useId is unique
  // per component instance (no colons — they break SVG id/CSS selectors); the
  // counter keeps successive re-renders within one instance unique too.
  const baseId = useId().replace(/:/g, '')
  const renderCount = useRef(0)
  const source = (code ?? '').trim()

  useEffect(() => {
    if (defer || !source) {
      return
    }

    let cancelled = false
    setError(null)

    // Import mermaid only when a diagram actually renders. The desktop build is
    // a single chunk, so this doesn't shrink the bundle, but it defers the heavy
    // library's module-level initialization from app boot to first diagram use.
    void import('@streamdown/mermaid')
      .then(({ mermaid }) => {
        const instance = mermaid.getMermaid({ theme: renderedMode === 'dark' ? 'dark' : 'default' })

        return instance.render(`${baseId}-${++renderCount.current}`, source)
      })
      .then(({ svg: rendered }) => {
        if (!cancelled) {
          setSvg(rendered)
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setSvg('')
          setError(err instanceof Error ? err.message : String(err))
        }
      })

    return () => {
      cancelled = true
    }
  }, [source, defer, renderedMode, baseId])

  if (defer || !source) {
    return <SyntaxHighlighter {...props} defer={defer} />
  }

  // Parse/render failure: show the source with a visible error badge so the
  // user knows *why* no diagram appeared instead of a silent fall-through.
  if (error) {
    return (
      <div className="space-y-1">
        <div
          className="flex items-start gap-1.5 px-0.5 text-[0.7rem] leading-snug text-destructive"
          role="alert"
          title={error}
        >
          <Codicon className="mt-px shrink-0" name="warning" size={12} />
          <span>
            {t.preview.diagramError}: {error.split('\n')[0]}
          </span>
        </div>
        <SyntaxHighlighter {...props} />
      </div>
    )
  }

  // Async render pending: keep showing the source, swap in the diagram when ready.
  if (!svg) {
    return <SyntaxHighlighter {...props} defer />
  }

  return (
    <CodeCard>
      <CodeCardHeader>
        <CodeCardTitle>
          <CodeCardIcon name="graph" />
          {t.assistant.tool.code}
          <CodeCardSubtitle> · mermaid</CodeCardSubtitle>
        </CodeCardTitle>
        <CopyButton
          appearance="inline"
          className="-my-1 -mr-1 h-5 px-1 opacity-55 hover:opacity-100"
          iconClassName="size-2.5"
          label={t.assistant.tool.copyCode}
          showLabel={false}
          text={source}
        />
      </CodeCardHeader>
      <CodeCardBody>
        {/* Mermaid output is sanitized by the library (securityLevel: strict). */}
        <div
          aria-label="mermaid diagram"
          className="flex justify-center overflow-x-auto bg-background p-3 [&_svg]:h-auto [&_svg]:max-w-full"
          dangerouslySetInnerHTML={{ __html: svg }}
          role="img"
        />
      </CodeCardBody>
    </CodeCard>
  )
}

export const MermaidDiagram = memo(MermaidDiagramImpl)
