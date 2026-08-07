import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { ModelOptionProvider } from '@/types/hermes'

import { ProviderManagerNav } from './provider-manager-nav'

const providers: ModelOptionProvider[] = [
  { slug: 'openai', name: 'OpenAI', models: ['gpt-4o', 'gpt-4o-mini'], authenticated: true },
  { slug: 'anthropic', name: 'Anthropic', models: ['claude-3-5-sonnet'], authenticated: true },
  { slug: 'deepseek', name: 'DeepSeek', models: [], authenticated: false }
]

describe('ProviderManagerNav', () => {
  it('renders every provider as a listbox option', () => {
    const { container } = render(
      <ProviderManagerNav providers={providers} onAdd={vi.fn()} selectedSlug="openai" onSelect={() => {}} />
    )

    screen.getByText('OpenAI')
    screen.getByText('Anthropic')
    screen.getByText('DeepSeek')
    expect(container.querySelectorAll('[role="option"]')).toHaveLength(3)
  })

  it('renders group headers in Local → Configured → Unconfigured order', () => {
    render(
      <ProviderManagerNav
        providers={[
          { slug: 'deepseek', name: 'DeepSeek', models: [], authenticated: false },
          { slug: 'openai', name: 'OpenAI', models: ['gpt-4o'], authenticated: true },
          { slug: 'local', name: 'Local' }
        ]}
        onAdd={vi.fn()}
        selectedSlug="openai"
        onSelect={() => {}}
      />
    )

    const headers = Array.from(document.querySelectorAll('[aria-hidden="true"]'))
      .map(el => el.textContent)
      .filter(Boolean)
    // Local, Configured, Unconfigured headers appear in that order.
    expect(headers).toEqual(['Local', 'Configured', 'Unconfigured'])
  })

  it('separates each group header with a top divider except the first', () => {
    render(
      <ProviderManagerNav
        providers={[
          { slug: 'deepseek', name: 'DeepSeek', models: [], authenticated: false },
          { slug: 'openai', name: 'OpenAI', models: ['gpt-4o'], authenticated: true },
          { slug: 'local', name: 'Local' }
        ]}
        onAdd={vi.fn()}
        selectedSlug="openai"
        onSelect={() => {}}
      />
    )

    const headers = Array.from(document.querySelectorAll('[aria-hidden="true"]')).filter(el => el.textContent)
    // First header (Local) has no top border; the following two (Configured,
    // Unconfigured) carry a border-t divider so each section reads as distinct.
    expect(headers[0].className).not.toContain('border-t')
    expect(headers[1].className).toContain('border-t')
    expect(headers[2].className).toContain('border-t')
  })

  it('colors active groups green and the unconfigured group muted', () => {
    render(
      <ProviderManagerNav
        providers={[
          { slug: 'deepseek', name: 'DeepSeek', models: [], authenticated: false },
          { slug: 'openai', name: 'OpenAI', models: ['gpt-4o'], authenticated: true }
        ]}
        onAdd={vi.fn()}
        selectedSlug="openai"
        onSelect={() => {}}
      />
    )

    const headers = Array.from(document.querySelectorAll('[aria-hidden="true"]')).filter(el => el.textContent)
    const configured = headers.find(el => el.textContent === 'Configured')
    const unconfigured = headers.find(el => el.textContent === 'Unconfigured')
    // Active (Configured) → green accent; Unconfigured → muted tertiary text.
    expect(configured?.className).toContain('text-(--ui-green)')
    expect(unconfigured?.className).toContain('text-(--ui-text-tertiary)')
    expect(unconfigured?.className).not.toContain('text-(--ui-green)')
  })

  it('marks the selected provider with aria-selected', () => {
    const { container } = render(
      <ProviderManagerNav providers={providers} onAdd={vi.fn()} selectedSlug="anthropic" onSelect={() => {}} />
    )

    const options = container.querySelectorAll('[role="option"]')
    const selected = Array.from(options).find(o => o.getAttribute('aria-selected') === 'true')
    expect(selected?.textContent).toContain('Anthropic')
  })

  it('calls onSelect when a provider is clicked', () => {
    const onSelect = vi.fn()
    render(<ProviderManagerNav providers={providers} onAdd={vi.fn()} selectedSlug="openai" onSelect={onSelect} />)

    fireEvent.click(screen.getByText('Anthropic'))
    expect(onSelect).toHaveBeenCalledWith('anthropic')
  })

  it('moves selection with ArrowDown and ArrowUp across groups', () => {
    const onSelect = vi.fn()
    const { container } = render(
      <ProviderManagerNav
        providers={[
          { slug: 'local', name: 'Local' },
          { slug: 'openai', name: 'OpenAI', models: ['gpt-4o'], authenticated: true },
          { slug: 'deepseek', name: 'DeepSeek', models: [], authenticated: false }
        ]}
        onAdd={vi.fn()}
        selectedSlug="local"
        onSelect={onSelect}
      />
    )

    const list = container.querySelector('[role="listbox"]') as HTMLElement
    // Flat order is [local, openai, deepseek] (Local → Configured → Unconfigured).
    fireEvent.keyDown(list, { key: 'ArrowDown' })
    expect(onSelect).toHaveBeenLastCalledWith('openai')

    fireEvent.keyDown(list, { key: 'ArrowUp' })
    expect(onSelect).toHaveBeenLastCalledWith('local')
  })

  it('ignores non-arrow keys', () => {
    const onSelect = vi.fn()
    const { container } = render(
      <ProviderManagerNav providers={providers} onAdd={vi.fn()} selectedSlug="openai" onSelect={onSelect} />
    )

    const list = container.querySelector('[role="listbox"]') as HTMLElement
    fireEvent.keyDown(list, { key: 'Enter' })
    expect(onSelect).not.toHaveBeenCalled()
  })
})
