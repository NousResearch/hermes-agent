import { useCallback, useEffect, useRef, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { Copy, Search } from '@/lib/icons'
import { notify } from '@/store/notifications'

import { AnsiText } from './ansi-text'

interface TerminalOutputProps {
  className?: string
  stderr?: string
  stdout: string
}

export function TerminalOutput({ className, stderr, stdout }: TerminalOutputProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const [searchOpen, setSearchOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchIndex, setSearchIndex] = useState(0)
  const lines = stdout.split('\n')
  const totalCount = lines.length + (stderr ? stderr.split('\n').length : 0)

  // Auto-scroll to bottom on new content
  useEffect(() => {
    if (autoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [stdout, stderr, autoScroll])

  const handleScroll = useCallback(() => {
    const el = containerRef.current
    if (!el) return
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 30
    setAutoScroll(atBottom)
  }, [])

  const handleCopy = useCallback(() => {
    const text = stderr ? `${stdout}\n${stderr}` : stdout
    void navigator.clipboard.writeText(text)
    notify({ kind: 'success', title: 'Copied', message: 'Output copied to clipboard.' })
  }, [stdout, stderr])

  const matchCount = searchQuery
    ? lines.filter(l => l.toLowerCase().includes(searchQuery.toLowerCase())).length
    : 0

  return (
    <div className={className}>
      {/* Toolbar */}
      <div className="flex items-center gap-1 border-b border-border/40 px-2 py-1">
        <button
          className="rounded p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          onClick={() => setSearchOpen(s => !s)}
          title="Search output"
          type="button"
        >
          <Search size={12} />
        </button>
        <button
          className="rounded p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          onClick={handleCopy}
          title="Copy output"
          type="button"
        >
          <Copy size={12} />
        </button>
        <span className="ml-auto text-[0.6rem] text-muted-foreground">{totalCount} lines</span>
      </div>

      {/* Search bar */}
      {searchOpen && (
        <div className="flex items-center gap-1 border-b border-border/40 px-2 py-1">
          <input
            autoFocus
            className="flex-1 bg-transparent text-xs text-foreground outline-none placeholder:text-muted-foreground"
            onChange={e => {
              setSearchQuery(e.target.value)
              setSearchIndex(0)
            }}
            placeholder="Search..."
            value={searchQuery}
          />
          {searchQuery && (
            <span className="text-[0.6rem] text-muted-foreground">
              {searchIndex + 1}/{matchCount}
            </span>
          )}
        </div>
      )}

      {/* Output */}
      <div
        className="max-h-[400px] overflow-auto bg-transparent px-2 py-1.5 font-mono text-[0.7rem] leading-relaxed"
        onScroll={handleScroll}
        ref={containerRef}
      >
        {stdout && <AnsiText text={stdout} />}
        {stderr && (
          <div className="mt-1 border-t border-border/40 pt-1">
            <div className="mb-1 text-[0.6rem] font-medium uppercase tracking-wider text-muted-foreground">
              stderr
            </div>
            <AnsiText text={stderr} />
          </div>
        )}
      </div>

      {/* Scroll to bottom */}
      {!autoScroll && (
        <button
          className="absolute bottom-2 right-2 rounded-full bg-background/80 px-2 py-0.5 text-[0.6rem] text-muted-foreground shadow-sm backdrop-blur transition-colors hover:bg-muted hover:text-foreground"
          onClick={() => {
            containerRef.current?.scrollTo({ top: containerRef.current.scrollHeight, behavior: 'smooth' })
            setAutoScroll(true)
          }}
          type="button"
        >
          <Codicon name="arrow-down" size={10} /> Bottom
        </button>
      )}
    </div>
  )
}
