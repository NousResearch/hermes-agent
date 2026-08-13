import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { $visibleModels } from '@/store/model-visibility'
import type { ModelOptionProvider } from '@/types/hermes'

import { ProviderModelList } from './provider-model-list'

// Radix DropdownMenu uses pointer capture, which jsdom does not implement.
beforeAll(() => {
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.releasePointerCapture ??= () => undefined
  Element.prototype.setPointerCapture ??= () => undefined
  HTMLElement.prototype.scrollIntoView ??= () => undefined
})

const provider: ModelOptionProvider = {
  slug: 'openai',
  name: 'OpenAI',
  models: ['gpt-4o', 'gpt-4o-mini', 'o1-preview']
}

describe('ProviderModelList', () => {
  beforeEach(() => {
    $visibleModels.set(null)
  })

  afterEach(() => {
    cleanup()
  })

  it('renders a switch per model, all on by default', () => {
    render(<ProviderModelList provider={provider} />)

    const switches = screen.getAllByRole('switch')
    expect(switches).toHaveLength(3)
    switches.forEach(sw => expect(sw.getAttribute('aria-checked')).toBe('true'))
  })

  it('toggles a model off when its switch is clicked', () => {
    render(<ProviderModelList provider={provider} />)

    const first = screen.getAllByRole('switch')[0]
    fireEvent.click(first)
    expect(first.getAttribute('aria-checked')).toBe('false')
  })

  it('shows the all-hidden banner once every model is toggled off', () => {
    render(<ProviderModelList provider={provider} />)

    expect(screen.queryByText(/all models hidden/i)).toBeNull()

    screen.getAllByRole('switch').forEach(sw => fireEvent.click(sw))
    expect(screen.queryByText(/all models hidden/i)).not.toBeNull()
  })

  it('filters models by the search term typed into the internal search field', () => {
    render(<ProviderModelList provider={provider} />)

    const searchInput = screen.getByRole('textbox', { name: /search models/i })
    fireEvent.change(searchInput, { target: { value: 'gpt-4o-mini' } })

    expect(screen.getAllByRole('switch')).toHaveLength(1)
  })

  it('shows the empty state when search matches nothing', () => {
    render(<ProviderModelList provider={provider} />)

    const searchInput = screen.getByRole('textbox', { name: /search models/i })
    fireEvent.change(searchInput, { target: { value: 'zzz' } })

    expect(screen.queryAllByRole('switch')).toHaveLength(0)
    expect(screen.queryByText(/no models/i)).not.toBeNull()
  })

  it('shows all models hidden and a disabled banner when the provider is deactivated', () => {
    render(<ProviderModelList enabled={false} provider={provider} />)

    const switches = screen.getAllByRole('switch')
    expect(switches).toHaveLength(3)
    switches.forEach(sw => expect(sw.getAttribute('aria-checked')).toBe('false'))
    expect(screen.queryByText(/disable provider/i)).not.toBeNull()
  })

  it('renders inline letter-group headers (B1)', () => {
    render(<ProviderModelList provider={provider} />)

    // Grouping is off by default; enable it via the tag.
    fireEvent.click(screen.getByRole('button', { name: /group by letter/i }))

    // gpt-4o / gpt-4o-mini → "G"; o1-preview → "O".
    expect(screen.queryAllByText('G').length).toBeGreaterThan(0)
    expect(screen.queryAllByText('O').length).toBeGreaterThan(0)
  })

  it('filters by a capability chip (B2)', () => {
    render(<ProviderModelList provider={provider} />)

    // o1-preview is inferred as reasoning; gpt-4o is not.
    fireEvent.click(screen.getByRole('button', { name: /reasoning/i }))
    const switches = screen.getAllByRole('switch')
    expect(switches).toHaveLength(1)
  })

  it('sorts A→Z when the sort select changes (B3)', () => {
    const many: ModelOptionProvider = {
      slug: 'p',
      name: 'P',
      models: ['zebra', 'alpha', 'mango']
    }
    render(<ProviderModelList provider={many} />)

    const select = screen.getByLabelText(/active first/i) as HTMLSelectElement
    fireEvent.change(select, { target: { value: 'az' } })

    const labels = screen.getAllByRole('switch').map(sw => sw.getAttribute('aria-label'))
    expect(labels).toEqual(['Alpha', 'Mango', 'Zebra'])
  })

  it('hides inactive models with the "active only" toggle (B4)', () => {
    render(<ProviderModelList provider={provider} />)

    // Turn off the first model.
    fireEvent.click(screen.getAllByRole('switch')[0])

    fireEvent.click(screen.getByRole('checkbox'))
    // Only the 2 still-active models remain.
    expect(screen.getAllByRole('switch')).toHaveLength(2)
  })

  it('activates all models via the bulk button (B4)', () => {
    // Start with everything hidden.
    $visibleModels.set(new Set([`openai::`]))
    render(<ProviderModelList provider={provider} />)

    expect(screen.getAllByRole('switch').every(sw => sw.getAttribute('aria-checked') === 'false')).toBe(true)

  fireEvent.click(screen.getByRole('button', { name: 'Activate all' }))
    expect(screen.getAllByRole('switch').every(sw => sw.getAttribute('aria-checked') === 'true')).toBe(true)
  })

  it('surfaces pricing and unavailable models (B5)', () => {
    const priced: ModelOptionProvider = {
      slug: 'p',
      name: 'P',
      models: ['gpt-4o'],
      pricing: { 'gpt-4o': { input: '$3.00', output: '$9.00', cache: null, free: false } },
      unavailable_models: ['gpt-4o']
    }
    render(<ProviderModelList provider={priced} />)

    expect(screen.getByText(/\$3\.00 \/ \$9\.00/)).toBeTruthy()
    expect(screen.getByText(/unavailable/i)).toBeTruthy()
    // Unavailable model switch is disabled (cannot be toggled on).
    expect(screen.getAllByRole('switch')[0].hasAttribute('disabled')).toBe(true)
  })

  it('does not reorder the list when a model is toggled (non-destructive sort)', () => {
    // Default sort is activeFirst. With all models active the order is
    // alphabetical. Toggling the first model OFF must NOT make it jump to the
    // bottom instantly — the row stays put until the list itself updates.
    render(<ProviderModelList provider={provider} />)

    const orderBefore = screen.getAllByRole('switch').map(sw => sw.getAttribute('aria-label'))
    expect(orderBefore).toEqual(['GPT-4o', 'GPT-4o-mini', 'O1'])

    fireEvent.click(screen.getAllByRole('switch')[0])

    const orderAfter = screen.getAllByRole('switch').map(sw => sw.getAttribute('aria-label'))
    // Order is unchanged: the toggled model did not jump.
    expect(orderAfter).toEqual(orderBefore)
    // But its checked state did flip.
    expect(screen.getAllByRole('switch')[0].getAttribute('aria-checked')).toBe('false')
  })

  it('re-sorts active-first only on a list update, not on toggle', () => {
    // Start with the last model hidden; activeFirst keeps visible ones on top.
    $visibleModels.set(new Set(['openai::gpt-4o', 'openai::gpt-4o-mini']))
    render(<ProviderModelList provider={provider} />)

    // o1-preview is hidden → it sits at the bottom.
    let order = screen.getAllByRole('switch').map(sw => sw.getAttribute('aria-label'))
    expect(order).toEqual(['GPT-4o', 'GPT-4o-mini', 'O1'])

    // Activating o1-preview via its switch must NOT instantly promote it.
    const o1Switch = screen.getAllByRole('switch').find(sw => sw.getAttribute('aria-label') === 'O1')!
    fireEvent.click(o1Switch)

    order = screen.getAllByRole('switch').map(sw => sw.getAttribute('aria-label'))
    expect(order).toEqual(['GPT-4o', 'GPT-4o-mini', 'O1'])
    expect(o1Switch.getAttribute('aria-checked')).toBe('true')
  })

  it('does not group by letter by default; toggling the tag enables grouping', () => {
    render(<ProviderModelList provider={provider} />)

    // No letter-group headers by default.
    expect(screen.queryByText('G')).toBeNull()
    expect(screen.queryByText('O')).toBeNull()

    // Toggle "Group by letter" on.
    fireEvent.click(screen.getByRole('button', { name: /group by letter/i }))

    // Now letter headers appear (they are aria-hidden, so use queryAllByText).
    expect(screen.queryAllByText('G').length).toBeGreaterThan(0)
    expect(screen.queryAllByText('O').length).toBeGreaterThan(0)
  })

  it('emphasizes the activated model with bold text', () => {
    render(<ProviderModelList provider={provider} />)

    const rows = screen.getAllByRole('switch').map(sw => sw.closest('label'))
    // All models start active → all rows are bold.
    rows.forEach(row => expect(row?.className).toMatch(/font-bold/))

    // Deactivate the first model → its row loses the bold.
    fireEvent.click(screen.getAllByRole('switch')[0])
    expect(rows[0]!.className).not.toMatch(/font-bold/)
    // The other two remain bold.
    expect(rows[1]!.className).toMatch(/font-bold/)
    expect(rows[2]!.className).toMatch(/font-bold/)
  })

  it('places the search field and the filter chips on the same toolbar row', () => {
    render(<ProviderModelList provider={provider} />)

    // The search input and the Vision chip share a single flex row ancestor.
    const searchInput = screen.getByRole('textbox', { name: /search models/i })
    const visionChip = screen.getByRole('button', { name: /vision/i })

    const toolbar = screen.getByTestId('model-toolbar')
    expect(toolbar.contains(searchInput)).toBe(true)
    expect(toolbar.contains(visionChip)).toBe(true)
  })

  it('outlines the search field with a border', () => {
    render(<ProviderModelList provider={provider} />)

    const searchInput = screen.getByRole('textbox', { name: /search models/i })
    // The input sits inside a bordered wrapper (the outlined field container).
    const wrapper = searchInput.parentElement
    expect(wrapper?.className).toMatch(/border/)
  })

  it('renders a search icon inside the outlined search field', () => {
    render(<ProviderModelList provider={provider} />)

    const searchInput = screen.getByRole('textbox', { name: /search models/i })
    const wrapper = searchInput.parentElement
    // A leading <svg> (the Search icon) precedes the input within the wrapper.
    expect(wrapper?.querySelector('svg')).not.toBeNull()
  })

  it('renders an SVG icon inside every capability filter chip', () => {
    render(<ProviderModelList provider={provider} />)

    for (const name of [/vision/i, /multimodal/i, /reasoning/i, /fast/i]) {
      const chip = screen.getByRole('button', { name })
      expect(chip.querySelector('svg')).not.toBeNull()
    }
  })

  it('colors a resting chip with its capability hue', () => {
    render(<ProviderModelList provider={provider} />)

    // Unpressed chips still read as their capability: reasoning is amber-ish.
    const reasoning = screen.getByRole('button', { name: /reasoning/i })
    expect(reasoning.getAttribute('aria-pressed')).toBe('false')
    expect(reasoning.className).toMatch(/amber/)
  })

  it('uses the shared badge color when a chip is active', () => {
    render(<ProviderModelList provider={provider} />)

    const reasoning = screen.getByRole('button', { name: /reasoning/i })
    fireEvent.click(reasoning)

    // Active state uses CAPABILITY_ACTIVE_CLASS (amber-500/30), the same color
    // as the reasoning badge on the model rows.
    expect(reasoning.getAttribute('aria-pressed')).toBe('true')
    expect(reasoning.className).toMatch(/amber-500\/30/)
  })

  it('gives the search field a min-width', () => {
    render(<ProviderModelList provider={provider} />)

    const wrapper = screen.getByRole('textbox', { name: /search models/i }).parentElement
    expect(wrapper?.className).toMatch(/min-w-\[10rem\]/)
  })

  it('gives the search field a hover accent', () => {
    render(<ProviderModelList provider={provider} />)

    const wrapper = screen.getByRole('textbox', { name: /search models/i }).parentElement
    expect(wrapper?.className).toMatch(/hover:border-/)
  })

  it('gives the search field a focus accent (border + ring)', () => {
    render(<ProviderModelList provider={provider} />)

    const wrapper = screen.getByRole('textbox', { name: /search models/i }).parentElement
    expect(wrapper?.className).toMatch(/focus-within:border-\(--ui-accent\)/)
    expect(wrapper?.className).toMatch(/focus-within:ring-2/)
  })

  it('renders both the inline chips and the collapsed Filters trigger with responsive classes', () => {
    render(<ProviderModelList provider={provider} />)

    const toolbar = screen.getByTestId('model-toolbar')
    // The toolbar is a container-query context.
    expect(toolbar.className).toMatch(/@container/)

    // Inline chips show only at/above the breakpoint.
    const inlineChips = screen.getByRole('button', { name: /vision/i }).parentElement
    expect(inlineChips?.className).toMatch(/hidden/)
    expect(inlineChips?.className).toMatch(/@\[24rem\]:flex/)

    // The Filters trigger shows only below the breakpoint.
    const filtersTrigger = screen.getByRole('button', { name: /filters/i })
    expect(filtersTrigger.className).toMatch(/@\[24rem\]:hidden/)
  })

  it('opens a Filters dropdown with four checkbox items that share the chip state', async () => {
    render(<ProviderModelList provider={provider} />)

    // Open the collapsed Filters menu (Radix opens on pointerdown).
    const trigger = screen.getByRole('button', { name: /filters/i })
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })

    const items = await screen.findAllByRole('menuitemcheckbox')
    expect(items).toHaveLength(4)

    // Toggle "Reasoning" from the dropdown…
    const reasoningItem = items.find(item => item.textContent?.match(/reasoning/i))!
    fireEvent.click(reasoningItem)

    // …and the inline chip reflects the shared state.
    const reasoningChip = screen.getByRole('button', { name: /reasoning/i })
    expect(reasoningChip.getAttribute('aria-pressed')).toBe('true')
  })

  it('shows an active-count badge on the Filters trigger when filters are set', async () => {
    render(<ProviderModelList provider={provider} />)

    // Enable two filters via the inline chips.
    fireEvent.click(screen.getByRole('button', { name: /vision/i }))
    fireEvent.click(screen.getByRole('button', { name: /reasoning/i }))

    // The Filters trigger surfaces the count.
    const filtersTrigger = screen.getByRole('button', { name: /filters/i })
    expect(filtersTrigger.textContent).toContain('2')
  })

  describe('discover / update button', () => {
    const builtinProvider: ModelOptionProvider = {
      slug: 'openai',
      name: 'OpenAI',
      models: ['gpt-4o', 'gpt-4o-mini']
    }

    const customProvider: ModelOptionProvider = {
      slug: 'custom:lab',
      name: 'Lab',
      models: ['model-a'],
      is_user_defined: true
    }

    const emptyProvider: ModelOptionProvider = {
      slug: 'custom:empty',
      name: 'Empty',
      models: [],
      is_user_defined: true
    }

    it('renders the discover button for a built-in provider', () => {
      render(<ProviderModelList provider={builtinProvider} />)
      expect(screen.getByRole('button', { name: /update list/i })).toBeTruthy()
    })

    it('shows "Update list" when the provider has models', () => {
      render(<ProviderModelList provider={builtinProvider} />)
      expect(screen.getByRole('button', { name: /update list/i })).toBeTruthy()
      expect(screen.queryByRole('button', { name: /discover models/i })).toBeNull()
    })

    it('shows "Discover models" when the model list is empty', () => {
      render(<ProviderModelList provider={emptyProvider} />)
      expect(screen.getByRole('button', { name: /discover models/i })).toBeTruthy()
      expect(screen.queryByRole('button', { name: /update list/i })).toBeNull()
    })

    it('shows "Add model" only for custom providers', () => {
      const { unmount } = render(<ProviderModelList provider={customProvider} />)
      expect(screen.getByRole('button', { name: /add model/i })).toBeTruthy()
      unmount()

      render(<ProviderModelList provider={builtinProvider} />)
      expect(screen.queryByRole('button', { name: /add model/i })).toBeNull()
    })

    it('calls onDiscover when the discover button is clicked', () => {
      const onDiscover = vi.fn()
      render(<ProviderModelList onDiscover={onDiscover} provider={builtinProvider} />)
      fireEvent.click(screen.getByRole('button', { name: /update list/i }))
      expect(onDiscover).toHaveBeenCalledTimes(1)
    })

    it('disables the discover button when discoverWorking is true', () => {
      render(<ProviderModelList discoverWorking provider={builtinProvider} />)
      const btn = screen.getByRole('button', { name: /update list/i })
      expect((btn as HTMLButtonElement).disabled).toBe(true)
    })

    it('disables the discover button when the provider is disabled', () => {
      render(<ProviderModelList enabled={false} provider={builtinProvider} />)
      const btn = screen.getByRole('button', { name: /update list/i })
      expect((btn as HTMLButtonElement).disabled).toBe(true)
    })
  })
})
