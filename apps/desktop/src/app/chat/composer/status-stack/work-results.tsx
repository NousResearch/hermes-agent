import { useSessionSlice } from '@/lib/use-session-slice'
import { $subagentsBySession } from '@/store/subagents'

export function WorkResults({ sessionId }: { sessionId: string }) {
  const items = useSessionSlice($subagentsBySession, sessionId)
  const results = items.filter(item => ['completed', 'failed', 'interrupted'].includes(item.status)).slice(-3)

  if (!results.length) {
    return null
  }

  return (
    <div aria-label="Wyniki pracy współpracowników" className="space-y-2 p-3">
      {results.map(item => (
        <details className="rounded-lg border border-(--stroke-nous) p-3 text-sm" key={item.id}>
          <summary className="cursor-pointer font-medium">
            {item.goal} ·{' '}
            {item.status === 'completed'
              ? item.summary
                ? 'Raport gotowy'
                : 'Zakończono bez raportu'
              : item.status === 'failed'
                ? 'Wymaga uwagi'
                : 'Przerwano'}
          </summary>
          <p className="mt-2 whitespace-pre-wrap text-muted-foreground">
            {item.summary || 'Backend nie dostarczył raportu wyniku. Otwórz rozmowę, aby sprawdzić szczegóły.'}
          </p>
          <p className="mt-2 font-medium">Gdzie jest wynik?</p>
          {item.filesWritten.length ? (
            <ul className="mt-1 space-y-1">
              {item.filesWritten.map(file => (
                <li className="break-all" key={file}>
                  {file}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-muted-foreground">Raport w tej rozmowie; brak zgłoszonych plików.</p>
          )}
        </details>
      ))}
    </div>
  )
}
