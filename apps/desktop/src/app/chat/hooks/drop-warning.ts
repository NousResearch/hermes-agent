/** A drop item the composer could not attach: what was dropped, and why when it is known. */
export interface DropFailure {
  name: string
  reason?: string
}

/** How many failed drop items the warning names before it counts the rest. */
const DROP_FAILURES_SHOWN = 3

/** The longest name the warning shows whole; a longer one keeps its start and its end. */
const DROP_FAILURE_NAME_CHARS = 48

/** The warning names failures until this many characters, then counts the rest. */
const DROP_WARNING_CHARS = 240

function middleEllipsis(text: string, max: number): string {
  if (text.length <= max) {
    return text
  }

  const head = Math.ceil((max - 1) / 2)

  return `${text.slice(0, head)}…${text.slice(text.length - (max - 1 - head))}`
}

function failureText({ name, reason }: DropFailure, maxName = Infinity): string {
  const label = `Could not attach ${middleEllipsis(name, maxName)}`

  return reason ? `${label}: ${reason}` : label
}

/** One bounded warning for a drop's failures: shortened names, a few of them, and a count of the
 *  rest, so a large drop or long paths can't grow the toast off screen. Whenever anything is
 *  shortened or left out, the full list rides in the collapsible detail. */
export function dropWarning(
  failures: DropFailure[],
  more: (count: number) => string
): { detail?: string; message: string } {
  const shown: string[] = []

  for (const failure of failures.slice(0, DROP_FAILURES_SHOWN)) {
    const text = failureText(failure, DROP_FAILURE_NAME_CHARS)

    if (shown.length && [...shown, text].join('; ').length > DROP_WARNING_CHARS) {
      break
    }

    shown.push(text)
  }

  const hidden = failures.length - shown.length
  const message = hidden > 0 ? `${shown.join('; ')}; ${more(hidden)}` : shown.join('; ')
  const full = failures.map(failure => failureText(failure))

  return message === full.join('; ') ? { message } : { detail: full.join('\n'), message }
}
