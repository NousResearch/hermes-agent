import { describe, expect, it } from 'vitest'

import {
  formatTurnTokens,
  imperativeToActive,
  latestReasoningStatus,
  latestSentence,
  pickSpinnerLabel,
  SPINNER_LABEL_MAX,
  SPINNER_VERBS,
  spinnerVerb,
  statusFromSentence,
  todoStatus,
  toGerund
} from './spinner-label'

describe('toGerund', () => {
  it.each([
    ['check', 'checking'],
    ['write', 'writing'],
    ['run', 'running'],
    ['see', 'seeing'],
    ['tie', 'tying'],
    ['map', 'mapping'],
    ['grep', 'grepping'],
    ['Read', 'Reading'],
    ['loading', 'loading']
  ])('%s -> %s', (verb, expected) => {
    expect(toGerund(verb)).toBe(expected)
  })

  it('leaves non-words alone', () => {
    expect(toGerund('S3')).toBe('S3')
  })
})

describe('statusFromSentence (real Opus 5.5 narration)', () => {
  it.each([
    [
      "I'm checking whether Hermes' todo tool exposes an activeForm.",
      "Checking whether Hermes' todo tool exposes an activeForm"
    ],
    [
      "Next I'll check what Hermes's Anthropic adapter sends for its thinking display settings.",
      "Checking what Hermes's Anthropic adapter sends for its thinking display settings"
    ],
    ['Let me write the plan to disk.', 'Writing the plan to disk'],
    [
      "I've found that Anthropic's official approach uses the updates beta.",
      "Found that Anthropic's official approach uses the updates beta"
    ],
    ['I confirmed the config change stuck.', 'Confirmed the config change stuck'],
    ["I'm going to run the vitest shard.", 'Running the vitest shard'],
    ['I need to map the gateway events.', 'Mapping the gateway events'],
    ['Okay, now I am weighing two designs.', 'Weighing two designs'],
    ['**Checking** the `status.tsx` row.', 'Checking the status.tsx row'],
    ['Confirmed, so I will keep the reply brief.', 'Keeping the reply brief'],
    ["Nothing has replied yet, so I'll search the channel history.", 'Searching the channel history'],
    ['I also need to check whether the robot is back.', 'Checking whether the robot is back'],
    ['Running the FF scan through `execute_code` now.', 'Running the FF scan through execute_code now'],
    ['Separately, the Encord thread is quiet.', 'The Encord thread is quiet'],
    ["That's the only thing left to pass along.", "That's the only thing left to pass along"]
  ])('%s', (sentence, expected) => {
    expect(statusFromSentence(sentence)).toBe(expected)
  })

  it('stops a long label at its first clause break', () => {
    expect(
      statusFromSentence(
        "I'm checking whether any pipeline files were changed in the last 40 minutes, since a stale build would explain it."
      )
    ).toBe('Checking whether any pipeline files were changed in the last 40 minutes')
  })

  it('clips a run-on sentence at a word boundary', () => {
    const long = `I'm checking ${'a very long clause '.repeat(20)}end.`
    const label = statusFromSentence(long)

    expect(label.length).toBeLessThanOrEqual(SPINNER_LABEL_MAX)
    expect(label.endsWith(' ')).toBe(false)
    expect(label.startsWith('Checking a very long clause')).toBe(true)
  })
})

describe('latestSentence', () => {
  it('takes the newest finished sentence of the newest paragraph', () => {
    expect(latestSentence("I've got the picture. Next I'll grep the adapter.")).toBe("Next I'll grep the adapter.")
  })

  it('ignores a sentence still streaming in the newest paragraph', () => {
    expect(latestSentence('I found the row. Now I am check')).toBe('I found the row.')
  })

  it('falls back to the previous paragraph while the new one has no full sentence', () => {
    expect(latestSentence("I'm reading status.tsx.\n\nNext I'll")).toBe("I'm reading status.tsx.")
  })

  it('returns empty when nothing has finished yet', () => {
    expect(latestSentence('I am starting to')).toBe('')
  })

  it('treats an earlier paragraph without terminal punctuation as finished', () => {
    expect(latestSentence('Checking the adapter\n\nStill typ')).toBe('Checking the adapter')
  })
})

describe('latestReasoningStatus', () => {
  it('labels from reasoning after the last tool call', () => {
    expect(
      latestReasoningStatus([
        { type: 'reasoning', text: "I'll read the file." },
        { type: 'tool-call' },
        { type: 'reasoning', text: "I'm comparing the two adapters." }
      ])
    ).toBe('Comparing the two adapters')
  })

  it('drops reasoning that a later tool call already acted on', () => {
    expect(latestReasoningStatus([{ type: 'reasoning', text: "I'll read the file." }, { type: 'tool-call' }])).toBe('')
  })

  it('skips empty reasoning parts (display omitted)', () => {
    expect(
      latestReasoningStatus([
        { type: 'reasoning', text: "I'm mapping the store." },
        { type: 'reasoning', text: '' }
      ])
    ).toBe('Mapping the store')
  })

  it('returns empty with no reasoning', () => {
    expect(latestReasoningStatus([{ type: 'text', text: 'Hello.' }])).toBe('')
  })
})

describe('todo fallback (Claude Code activeForm)', () => {
  it.each([
    ['Build the status row', 'Building the status row'],
    ['Write tests', 'Writing tests'],
    ['Run QA', 'Running QA'],
    ['New sidebar section', 'New sidebar section'],
    ['API audit', 'API audit'],
    ['Deployed to Mac', 'Deployed to Mac']
  ])('%s -> %s', (title, expected) => {
    expect(imperativeToActive(title)).toBe(expected)
  })

  it('uses only the in-progress item', () => {
    expect(
      todoStatus([
        { content: 'Plan it', id: '1', status: 'completed' },
        { content: 'Ship it', id: '2', status: 'in_progress' },
        { content: 'Report', id: '3', status: 'pending' }
      ])
    ).toBe('Shipping it')
    expect(todoStatus([{ content: 'Plan it', id: '1', status: 'pending' }])).toBe('')
    expect(todoStatus(undefined)).toBe('')
  })
})

describe('spinnerVerb', () => {
  it('is stable for one turn seed and always from the list', () => {
    expect(spinnerVerb(1_730_000_000_123)).toBe(spinnerVerb(1_730_000_000_123))
    expect(SPINNER_VERBS).toContain(spinnerVerb(undefined))
    expect(SPINNER_VERBS).toContain(spinnerVerb(-7))
  })
})

describe('pickSpinnerLabel precedence', () => {
  it('reasoning, then todo, then a verb', () => {
    expect(pickSpinnerLabel('Reading the adapter', 'Shipping it', 1)).toBe('Reading the adapter')
    expect(pickSpinnerLabel('', 'Shipping it', 1)).toBe('Shipping it')
    expect(SPINNER_VERBS).toContain(pickSpinnerLabel('', '', 1))
  })
})

describe('formatTurnTokens', () => {
  it.each([
    [0, ''],
    [-5, ''],
    [Number.NaN, ''],
    [812, '812 tokens'],
    [1000, '1k tokens'],
    [8_512, '8.5k tokens'],
    [42_400, '42k tokens']
  ])('%s -> %s', (n, expected) => {
    expect(formatTurnTokens(n)).toBe(expected)
  })
})
