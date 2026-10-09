import type { FC } from 'react'

// tui_gateway/start_chat.py::_setup_learned: the block a setup handoff appends
// under the user's ask. The model reads all of it; the bubble shows only the ask.
const SETUP_LEARNED_MARKER = '\n\nWhat setup learned about me:\n'

/** `[ask, block]`: the block is null for every message that has none. */
export function splitSetupLearned(text: string): [string, null | string] {
  const index = text.indexOf(SETUP_LEARNED_MARKER)

  return index < 0 ? [text, null] : [text.slice(0, index), text.slice(index + 2)]
}

export const SetupLearnedNote: FC<{ text: string }> = ({ text }) => {
  const [title = '', ...lines] = text.split('\n')

  return (
    <details className="mb-2 text-[0.75rem] text-muted-foreground" data-slot="aui_setup-learned">
      <summary className="cursor-pointer select-none hover:text-foreground/70">{title.replace(/:$/, '')}</summary>
      <div className="mt-1 whitespace-pre-line leading-5 text-foreground/75">{lines.join('\n')}</div>
    </details>
  )
}
