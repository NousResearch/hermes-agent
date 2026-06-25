'use client'

import { type FC, useEffect, useMemo, useState } from 'react'
import mermaid from 'mermaid'

import { CopyButton } from '@/components/ui/copy-button'
import { cn } from '@/lib/utils'

let mermaidInit = false
function ensureMermaidInit() {
  if (mermaidInit) return
  mermaidInit = true
  mermaid.initialize({
    startOnLoad: false,
    theme: 'default',
    securityLevel: 'strict',
    flowchart: { useMaxWidth: true, htmlLabels: true },
    sequence: { useMaxWidth: true },
  })
}

interface MermaidBlockProps {
  code: string
}

export const MermaidBlock: FC<MermaidBlockProps> = ({ code }) => {
  const [svg, setSvg] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const id = useMemo(() => `mermaid-${Math.random().toString(36).slice(2, 10)}`, [])

  useEffect(() => {
    let cancelled = false
    ensureMermaidInit()
    void (async () => {
      try {
        const trimmed = code.trim()
        if (!trimmed) {
          if (!cancelled) setSvg(null)
          return
        }
        const result = await mermaid.render(`${id}-svg`, trimmed)
        if (!cancelled) {
          setSvg(result.svg)
          setError(null)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Mermaid render failed')
          setSvg(null)
        }
      }
    })()
    return () => { cancelled = true }
  }, [code, id])

  if (svg) {
    return (
      <div className="my-2 rounded-[0.5rem] border border-border bg-muted/20 p-3">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-[0.6875rem] font-medium text-muted-foreground uppercase tracking-wider">
            Mermaid Diagram
          </span>
          <CopyButton
            appearance="inline"
            className="-my-1 -mr-1 h-5 px-1 opacity-55 hover:opacity-100"
            iconClassName="size-2.5"
            label="Copy source"
            showLabel={false}
            text={code}
          />
        </div>
        <div
          className={cn('mermaid-container flex justify-center overflow-x-auto', '[&_svg]:max-w-full [&_svg]:h-auto')}
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      </div>
    )
  }

  if (!error) {
    return (
      <div className="my-2 rounded-[0.5rem] border border-border bg-muted/10 p-4 text-center">
        <span className="text-[0.75rem] text-muted-foreground animate-pulse">Rendering diagram…</span>
      </div>
    )
  }

  return (
    <div className="my-2 rounded-[0.5rem] border border-border bg-muted/20 p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[0.6875rem] font-medium text-muted-foreground uppercase tracking-wider">
          Mermaid (render failed)
        </span>
        <CopyButton
          appearance="inline"
          className="-my-1 -mr-1 h-5 px-1 opacity-55 hover:opacity-100"
          iconClassName="size-2.5"
          label="Copy source"
          showLabel={false}
          text={code}
        />
      </div>
      <pre className="overflow-x-auto whitespace-pre text-[0.75rem] font-mono text-foreground/80 p-2">
        <code>{code}</code>
      </pre>
    </div>
  )
}
