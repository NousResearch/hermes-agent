import { useMemo } from 'react'

interface QuoteRun {
  body: string
  header: string | null
  quoted: boolean
}

function splitQuoteRuns(text: string): QuoteRun[] {
  const runs: QuoteRun[] = []

  for (const line of text.split('\n')) {
    const match = /^> ?(.*)$/.exec(line)
    const quoted = Boolean(match)
    const value = match ? match[1] : line
    const last = runs.at(-1)

    if (last?.quoted === quoted) {
      last.body += `\n${value}`
    } else {
      runs.push({ body: value, header: null, quoted })
    }
  }

  for (const run of runs) {
    if (!run.quoted) {
      continue
    }

    const newline = run.body.indexOf('\n')
    const first = (newline === -1 ? run.body : run.body.slice(0, newline)).replace(/^\*\*(.*)\*\*$/, '$1')

    if (/^\[(?:assistant|user) \d{2}:\d{2}\]$/.test(first)) {
      run.header = first
      run.body = newline === -1 ? '' : run.body.slice(newline + 1)
    }
  }

  return runs
}

/** Render message quotes without loading the assistant's markdown pipeline. */
export function UserQuoteText({ text, renderText }: { text: string; renderText: (text: string) => React.ReactNode }) {
  const runs = useMemo(() => splitQuoteRuns(text), [text])

  return runs.map((run, index) =>
    run.quoted ? (
      <blockquote
        className="my-2 border-l-2 border-(--ui-stroke-tertiary) pl-3 text-(--ui-text-secondary)"
        data-slot="aui_user-quote"
        key={index}
      >
        {run.header && <span className="block text-xs text-(--ui-text-tertiary)">{run.header}</span>}
        {renderText(run.body)}
      </blockquote>
    ) : (
      <span key={index}>{renderText(run.body)}</span>
    )
  )
}
