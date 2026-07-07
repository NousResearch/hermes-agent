import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { HermesUiChecklistArtifact } from '@/lib/hermes-ui-artifacts'

import { ChecklistArtifactCard } from './hermes-ui-artifact'
import { RichCodeBlock } from './registry'

const artifact: HermesUiChecklistArtifact = {
  description: 'Operational source-of-truth checklist for Obsidian-backed durable context.',
  id: 'obsidian-source-of-truth-policy',
  items: [
    { id: 'obsidian-profile-vault', label: 'Active profile: office-work' },
    { id: 'obsidian-source-truth', label: 'Obsidian is the source of truth.' }
  ],
  title: 'Obsidian source-of-truth policy',
  type: 'checklist'
}

const storageKey = 'hermes-ui:checklist:obsidian-source-of-truth-policy'

beforeEach(() => {
  window.localStorage.clear()
})

afterEach(cleanup)

describe('ChecklistArtifactCard', () => {
  it('renders title, items, checkboxes, and progress', () => {
    render(<ChecklistArtifactCard artifact={artifact} />)

    expect(screen.getByText('Obsidian source-of-truth policy')).toBeTruthy()
    expect(screen.getByLabelText('Active profile: office-work')).toBeTruthy()
    expect(screen.getByLabelText('Obsidian is the source of truth.')).toBeTruthy()
    expect(screen.getByText('0 / 2')).toBeTruthy()
  })

  it('clicking a checkbox updates progress and persists localStorage', () => {
    render(<ChecklistArtifactCard artifact={artifact} />)

    fireEvent.click(screen.getByLabelText('Active profile: office-work'))

    expect(screen.getByText('1 / 2')).toBeTruthy()
    expect(JSON.parse(window.localStorage.getItem(storageKey) || '{}')).toEqual({
      'obsidian-profile-vault': true,
      'obsidian-source-truth': false
    })
  })

  it('Mark all and Clear update all items', () => {
    render(<ChecklistArtifactCard artifact={artifact} />)

    fireEvent.click(screen.getByRole('button', { name: 'Mark all' }))
    expect(screen.getByText('2 / 2')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Clear' }))
    expect(screen.getByText('0 / 2')).toBeTruthy()
  })

  it('loads persisted state and ignores stale stored ids', () => {
    window.localStorage.setItem(
      storageKey,
      JSON.stringify({
        'gone-item': true,
        'obsidian-profile-vault': true
      })
    )

    render(<ChecklistArtifactCard artifact={artifact} />)

    expect(screen.getByText('1 / 2')).toBeTruthy()
    expect(screen.getByLabelText('Active profile: office-work')).toHaveProperty('checked', true)
  })

  it('renders HTML-looking labels as text only', () => {
    render(
      <ChecklistArtifactCard
        artifact={{
          ...artifact,
          items: [{ id: 'script-label', label: '<img src=x onerror=alert(1)><script>alert(1)</script>' }]
        }}
      />
    )

    expect(screen.getByText('<img src=x onerror=alert(1)><script>alert(1)</script>')).toBeTruthy()
    expect(document.querySelector('script')).toBeNull()
    expect(document.querySelector('img')).toBeNull()
  })

  it('renders Hebrew checklists right-to-left with localized action labels', () => {
    render(
      <ChecklistArtifactCard
        artifact={{
          ...artifact,
          description: 'רשימה בעברית עם הסבר קצר.',
          items: [
            {
              description: 'הסבר נוסף שמופיע כשורה נפרדת וקריאה יותר.',
              id: 'hebrew-item',
              label: 'פרופיל פעיל: Hermes עובד בפרופיל office-work.'
            }
          ],
          title: 'מדיניות מקור האמת של Obsidian'
        }}
      />
    )

    const card = screen.getByLabelText('מדיניות מקור האמת של Obsidian')

    expect(card.getAttribute('dir')).toBe('rtl')
    expect(screen.getByRole('button', { name: 'סמן הכול' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'נקה' })).toBeTruthy()
    expect(screen.getByText('הסבר נוסף שמופיע כשורה נפרדת וקריאה יותר.')).toBeTruthy()
  })

  it('renders valid hermes-ui rich code blocks and falls back for invalid payloads', async () => {
    const { rerender } = render(
      <RichCodeBlock code={JSON.stringify(artifact)} fallback={<pre>fallback code block</pre>} language="hermes-ui" />
    )

    await waitFor(() => expect(screen.getByText('Obsidian source-of-truth policy')).toBeTruthy())
    expect(screen.queryByText('fallback code block')).toBeNull()

    rerender(<RichCodeBlock code="{ nope" fallback={<pre>fallback code block</pre>} language="hermes-ui" />)

    expect(screen.getByText('fallback code block')).toBeTruthy()
  })
})
