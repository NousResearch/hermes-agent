import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { toolFixtures } from './fixtures'
import { ToolsList } from './tools-list'
import type { ToolsEditorPhase } from './types'
import { useToolsEditor } from './use-tools-editor'

const onSave = vi.fn(async () => 'saved' as const)

/** The real pairing: the hook owns the edit, the list renders it. Testing them
 *  apart would prove neither, because every rule below lives in the seam. */
function Harness({ phase = 'ready' as ToolsEditorPhase, savedDisabled = [] as string[] }) {
  const tools = toolFixtures(savedDisabled)
  const editor = useToolsEditor({ onSave, savedDisabled, tools })

  return (
    <ToolsList
      connectorName="Linear"
      counts={editor.counts}
      currentAction={editor.currentAction}
      dirty={editor.dirty}
      isOn={editor.isOn}
      onApplyQuickAction={editor.applyQuickAction}
      onDiscard={editor.discard}
      onKeepMine={editor.keepMine}
      onRefresh={() => {}}
      onReload={() => {}}
      onRemove={() => {}}
      onRetry={() => {}}
      onSave={() => void editor.save()}
      onSignIn={() => {}}
      onToggle={editor.toggle}
      phase={phase === 'ready' ? editor.phase : phase}
      tools={tools}
    />
  )
}

// cmdk scrolls its active item into view when the category picker opens; jsdom
// has no layout and therefore no `scrollIntoView`.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
})

const switchFor = (label: string) => screen.getByRole('switch', { name: label })
const footer = () => screen.queryByText(/back on$/)

afterEach(cleanup)

describe('editing the tool list', () => {
  it('counts one toggle in the dirty footer', () => {
    render(<Harness />)

    expect(footer()).toBeNull()

    fireEvent.click(switchFor('Turn Create an issue off'))

    expect(screen.getByText('1 tool off, none back on')).toBeTruthy()
    expect(switchFor('Turn Create an issue on').getAttribute('aria-checked')).toBe('false')
  })

  it('counts a tool put back on separately from one taken off', () => {
    render(<Harness savedDisabled={['LINEAR_CREATE_ISSUE']} />)

    fireEvent.click(switchFor('Turn Create an issue on'))
    fireEvent.click(switchFor('Turn Archive an issue off'))

    expect(screen.getByText('1 tool off, 1 back on')).toBeTruthy()
  })

  it('flips the switches a quick action names, and only those', () => {
    render(<Harness />)

    fireEvent.click(screen.getByRole('button', { name: 'Turn off destructive' }))

    expect(switchFor('Turn Archive an issue on').getAttribute('aria-checked')).toBe('false')
    expect(switchFor('Turn Delete a comment on').getAttribute('aria-checked')).toBe('false')
    expect(switchFor('Turn Create an issue off').getAttribute('aria-checked')).toBe('true')
    expect(screen.getByText('2 tools off, none back on')).toBeTruthy()
  })

  it('leaves a tool with an unknown effect exactly as the person left it', () => {
    render(<Harness savedDisabled={['LINEAR_PING']} />)

    fireEvent.click(screen.getByRole('button', { name: 'Read only' }))

    // Nothing the person did by hand to an unclassified tool is rewritten, so
    // the only change the footer reports is the action's own expansion.
    expect(screen.getByText('6 tools off, none back on')).toBeTruthy()
  })

  it('gives an org-locked tool no switch to press', () => {
    render(<Harness />)

    expect(screen.queryByRole('switch', { name: /Delete a project/ })).toBeNull()
    expect(screen.getAllByText('off by your organisation').length).toBeGreaterThan(0)
  })

  it('keeps a toggle made outside the current filter', () => {
    render(<Harness />)

    fireEvent.click(switchFor('Turn Create an issue off'))
    fireEvent.change(screen.getByRole('textbox', { name: /Search/ }), { target: { value: 'archive' } })

    expect(screen.queryByRole('switch', { name: /Create an issue/ })).toBeNull()

    fireEvent.click(switchFor('Turn Archive an issue off'))
    fireEvent.change(screen.getByRole('textbox', { name: /Search/ }), { target: { value: '' } })

    expect(switchFor('Turn Create an issue on').getAttribute('aria-checked')).toBe('false')
    expect(switchFor('Turn Archive an issue on').getAttribute('aria-checked')).toBe('false')
    expect(screen.getByText('2 tools off, none back on')).toBeTruthy()
  })

  it('discards back to the saved rule', () => {
    render(<Harness />)

    fireEvent.click(switchFor('Turn Create an issue off'))
    fireEvent.click(screen.getByRole('button', { name: 'Discard' }))

    expect(footer()).toBeNull()
    expect(switchFor('Turn Create an issue off').getAttribute('aria-checked')).toBe('true')
  })

  it('hides deprecated tools behind their own toggle', () => {
    render(<Harness />)

    fireEvent.change(screen.getByRole('textbox', { name: /Search/ }), { target: { value: 'legacy' } })

    expect(screen.getByText('No tool matches these filters.')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Show 1 deprecated' }))

    expect(screen.getByText('Sync the legacy board')).toBeTruthy()
  })

  it('opens one tool at a time and shows its slug behind the disclosure', () => {
    render(<Harness />)

    expect(screen.queryByText('LINEAR_CREATE_ISSUE')).toBeNull()

    fireEvent.click(screen.getByRole('button', { expanded: false, name: /Create an issue/ }))

    expect(screen.getByText('LINEAR_CREATE_ISSUE')).toBeTruthy()
  })
})

describe('what the chrome promises', () => {
  it('counts a chip over the rows it can actually show', () => {
    render(<Harness />)

    // `Write` covers five tools, one of them deprecated and hidden. A chip that
    // counted it would name a row the reader cannot reach, and could raise a
    // second chip past the two-value threshold on a row that is not on screen.
    fireEvent.click(screen.getByRole('button', { name: /^Write/ }))

    expect(screen.getByRole('button', { name: /^Write/ }).textContent).toBe('Write4')
    expect(screen.getAllByRole('switch')).toHaveLength(4)
  })

  it('keeps the scroll position and the open row through a query that matches nothing', () => {
    render(<Harness />)

    fireEvent.click(screen.getByRole('button', { expanded: false, name: /Create an issue/ }))

    const search = screen.getByRole('textbox', { name: /Search/ })

    fireEvent.change(search, { target: { value: 'quickbooks' } })

    expect(screen.getByText('No tool matches these filters.')).toBeTruthy()

    fireEvent.change(search, { target: { value: '' } })

    // The viewport owns both, so unmounting it for the empty case made one
    // keystroke behave differently from the next.
    expect(screen.getByText('LINEAR_CREATE_ISSUE')).toBeTruthy()
  })

  it('closes the category picker once a category is picked', () => {
    render(<Harness />)

    fireEvent.click(screen.getByRole('button', { name: /categories/ }))
    fireEvent.click(screen.getByText('comments'))

    // The panel opens over the rows it filters, so leaving it open hides the
    // answer to the question the reader just asked.
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(screen.getAllByRole('switch')).toHaveLength(2)
  })

  it('paints rows after a filter shortens the list under a scrolled viewport', () => {
    const { container } = render(<Harness />)
    const viewport = container.querySelector('[data-slot="tools-viewport"]')!

    fireEvent.scroll(viewport, { target: { scrollTop: 400 } })
    fireEvent.change(screen.getByRole('textbox', { name: /Search/ }), { target: { value: 'archive' } })

    // An offset the browser has not clamped yet must not slice the window past
    // the end of the new list and paint a blank scroller.
    expect(screen.getByText('Archive an issue')).toBeTruthy()
  })
})

describe('the states that replace the list', () => {
  it('names the problem and the way out, one per phase', () => {
    const cases: [ToolsEditorPhase, string, string][] = [
      ['unavailable', 'Could not load the tool list.', 'Retry'],
      ['gone', 'Linear left the catalog.', 'Remove'],
      ['signedOut', 'Sign in to Nous to read the tool list.', 'Sign in to Nous'],
      ['conflict', 'Someone changed this rule while you were editing.', 'Reload their version']
    ]

    for (const [phase, title, action] of cases) {
      render(<Harness phase={phase} />)

      expect(screen.getByText(title)).toBeTruthy()
      expect(screen.getByRole('button', { name: action })).toBeTruthy()

      cleanup()
    }
  })

  it('shows a static wash while it loads, with no list and no filters', () => {
    render(<Harness phase="loading" />)

    expect(screen.getByRole('status')).toBeTruthy()
    expect(screen.queryByRole('switch')).toBeNull()
  })
})
