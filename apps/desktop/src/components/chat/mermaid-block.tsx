import { useEffect, useRef, useState } from 'react'

let mermaidPromise: Promise<typeof import('mermaid')> | null = null

function loadMermaid() {
  if (!mermaidPromise) {
    mermaidPromise = import('mermaid').then(async mod => {
      const m = mod.default

      m.initialize({
        securityLevel: 'strict',
        startOnLoad: false,
        theme: document.documentElement.classList.contains('dark') ? 'dark' : 'default'
      })

      return mod
    })
  }

  return mermaidPromise
}

interface MermaidBlockProps {
  chart: string
}

export function MermaidBlock({ chart }: MermaidBlockProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [svg, setSvg] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    async function render() {
      try {
        const { default: mermaid } = await loadMermaid()
        const id = `mermaid-${Math.random().toString(36).slice(2, 10)}`
        const { svg: rendered } = await mermaid.render(id, chart)

        if (!cancelled) {
          setSvg(rendered)
          setError(null)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err))
          setSvg(null)
        }
      }
    }

    void render()

    return () => {
      cancelled = true
    }
  }, [chart])

  if (error) {
    return (
      <div className="my-3 rounded-md border border-border bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
        <div className="mb-1 font-medium text-foreground">Mermaid diagram error</div>
        <pre className="whitespace-pre-wrap break-words text-[0.75rem]">{error}</pre>
        <details className="mt-2">
          <summary className="cursor-pointer text-muted-foreground/70 hover:text-foreground">View source</summary>
          <pre className="mt-1 whitespace-pre-wrap break-words text-[0.75rem]">{chart}</pre>
        </details>
      </div>
    )
  }

  if (!svg) {
    return <div className="my-3 text-xs text-muted-foreground">Rendering diagram…</div>
  }

  return (
    <div
      ref={containerRef}
      className="my-3 overflow-auto rounded-md border border-border bg-background p-3 [&_svg]:max-w-full"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  )
}
