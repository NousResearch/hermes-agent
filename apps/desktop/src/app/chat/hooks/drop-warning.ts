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

function failureText({ name, reason }: DropFailure, maxName = Infinity, joined = false): string {
  const label = `Could not attach ${middleEllipsis(name, maxName)}`
  // A reason ends with a period, which reads '….;' before the '; ' that joins the next item.
  const text = joined ? reason?.replace(/\.$/, '') : reason

  return text ? `${label}: ${text}` : label
}

/** The failures joined into one line, then the count of the rest when there is one. */
function summaryText(failures: DropFailure[], maxName: number, rest?: string): string {
  const items = failures.map((failure, index) =>
    failureText(failure, maxName, rest !== undefined || index < failures.length - 1)
  )

  return (rest === undefined ? items : [...items, rest]).join('; ')
}

/** One bounded warning for a drop's failures: shortened names, a few of them, and a count of the
 *  rest, so a large drop or long paths can't grow the toast off screen. Whenever anything is
 *  shortened or left out, the full list, each reason whole, rides in the collapsible detail. */
export function dropWarning(
  failures: DropFailure[],
  more: (count: number) => string
): { detail?: string; message: string } {
  const shown: DropFailure[] = []

  for (const failure of failures.slice(0, DROP_FAILURES_SHOWN)) {
    if (shown.length && summaryText([...shown, failure], DROP_FAILURE_NAME_CHARS).length > DROP_WARNING_CHARS) {
      break
    }

    shown.push(failure)
  }

  const hidden = failures.length - shown.length
  const message = summaryText(shown, DROP_FAILURE_NAME_CHARS, hidden > 0 ? more(hidden) : undefined)

  return message === summaryText(failures, Infinity)
    ? { message }
    : { detail: failures.map(failure => failureText(failure)).join('\n'), message }
}
