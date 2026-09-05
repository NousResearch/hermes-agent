import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  resetDirectiveDiagnostics,
  warnDirectiveRenderFailed,
  warnUnclaimedDirective,
  warnUnparsedDirective
} from '@/lib/directive-diagnostics'

/**
 * The bug class these guard: a directive that silently does not render. The
 * transcript shows raw `::name{...}` text, which reads like model junk, so the
 * failure never gets reported as an app bug. Every path must leave a trace.
 */

let warn: ReturnType<typeof vi.spyOn>

beforeEach(() => {
  resetDirectiveDiagnostics()
  warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
})

afterEach(() => {
  warn.mockRestore()
})

const text = () => warn.mock.calls.map((call: unknown[]) => call.join(' ')).join('\n')

describe('warnUnparsedDirective', () => {
  it('names trailing debris after the closing brace', () => {
    // Two closers is past the tolerated-debris cap, so it still fails to parse
    // — and must now SAY why instead of silently rendering as text.
    warnUnparsedDirective('::followup{p1="a"}}', false)

    expect(warn).toHaveBeenCalledTimes(1)
    expect(text()).toContain('unexpected text after the closing brace')
  })

  it('names an unclosed attribute brace', () => {
    warnUnparsedDirective('::followup{p1="a"', false)

    expect(text()).toContain('never closed')
  })

  it('names a multi-line directive', () => {
    warnUnparsedDirective('::followup{p1="a"}\ntrailing', false)

    expect(text()).toContain('multiple lines')
  })

  it('names an over-long directive', () => {
    warnUnparsedDirective(`::followup{p1="${'x'.repeat(1300)}"}`, false)

    expect(text()).toContain('max 1200')
  })

  it('truncates a huge directive in the log body', () => {
    warnUnparsedDirective(`::followup{p1="${'x'.repeat(1300)}"}`, false)

    const logged = text()

    expect(logged).toContain('…')
    expect(logged.length).toBeLessThan(600)
  })

  it('stays silent while the message is still streaming', () => {
    // A half-arrived directive is malformed by definition; warning here would
    // fire on nearly every token of every directive ever emitted.
    warnUnparsedDirective('::followup{p1="Bắn m', true)

    expect(warn).not.toHaveBeenCalled()
  })

  it('stays silent for ordinary prose', () => {
    for (const prose of ['Just a sentence.', 'std::vector<int> v;', 'See the docs.', '']) {
      warnUnparsedDirective(prose, false)
    }

    expect(warn).not.toHaveBeenCalled()
  })

  it('stays silent for a directive that parses fine', () => {
    warnUnparsedDirective('::followup{p1="Clean the wt-* worktrees"}', false)

    expect(warn).not.toHaveBeenCalled()
  })

  it('logs each distinct problem once', () => {
    for (let i = 0; i < 50; i += 1) {
      warnUnparsedDirective('::followup{p1="a"', false)
    }

    expect(warn).toHaveBeenCalledTimes(1)
  })
})

describe('warnUnclaimedDirective', () => {
  it('reports the name and what is registered', () => {
    warnUnclaimedDirective('followup', ['preview', 'tasks'], false)

    expect(text()).toContain('::followup')
    expect(text()).toContain('preview, tasks')
  })

  it('calls out an empty registry', () => {
    warnUnclaimedDirective('followup', [], false)

    expect(text()).toContain('No directive plugins are registered')
  })

  it('stays silent while streaming', () => {
    warnUnclaimedDirective('followup', [], true)

    expect(warn).not.toHaveBeenCalled()
  })

  it('logs once per directive name', () => {
    for (let i = 0; i < 20; i += 1) {
      warnUnclaimedDirective('followup', [], false)
    }

    expect(warn).toHaveBeenCalledTimes(1)
  })
})

describe('warnDirectiveRenderFailed', () => {
  it('names the directive, the plugin and the error', () => {
    warnDirectiveRenderFailed('followup', 'plugin:follow-up', new Error('boom'))

    const logged = text()

    expect(logged).toContain('::followup')
    expect(logged).toContain('plugin:follow-up')
    expect(logged).toContain('boom')
  })

  it('handles a non-Error throw', () => {
    warnDirectiveRenderFailed('followup', 'plugin:follow-up', 'raw string')

    expect(text()).toContain('raw string')
  })

  it('logs distinct failures separately but repeats once', () => {
    warnDirectiveRenderFailed('followup', 'plugin:a', new Error('boom'))
    warnDirectiveRenderFailed('followup', 'plugin:a', new Error('boom'))
    warnDirectiveRenderFailed('followup', 'plugin:b', new Error('boom'))

    expect(warn).toHaveBeenCalledTimes(2)
  })
})

describe('diagnostics hygiene', () => {
  it('bounds the dedupe set so a long session cannot leak', () => {
    for (let i = 0; i < 1200; i += 1) {
      warnUnclaimedDirective(`name${i}`, [], false)
    }

    // Bounded at 500: the set clears rather than growing without limit.
    expect(warn.mock.calls.length).toBeLessThanOrEqual(1200)

    // After the clear, an already-seen name logs again — proof the set reset
    // rather than silently swallowing everything forever.
    warn.mockClear()
    warnUnclaimedDirective('name0', [], false)
    expect(warn).toHaveBeenCalledTimes(1)
  })
})
