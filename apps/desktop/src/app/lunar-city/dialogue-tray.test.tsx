import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { I18nProvider, type Locale, TRANSLATIONS } from '@/i18n'

import { DialogueTray } from './dialogue-tray'
import type { WorldActionRunner } from './world-actions'

const localizedLocales = (Object.keys(TRANSLATIONS).filter(locale => locale !== 'en') as Locale[])

describe('DialogueTray', () => {
  it('shows source truth and only allowed actions for a blocked task', () => {
    render(
      <DialogueTray
        onAction={vi.fn<WorldActionRunner['run']>(async () => ({ kind: 'completed', ok: true }))}
        subject={{
          condition: {
            active: true,
            facts: {},
            id: 'condition-7',
            kind: 'task.blocked',
            scope: 'task',
            severity: 'warning',
            source: 'kanban',
            sourceRef: { taskId: 'task-7' },
            title: 'Fix auth'
          },
          detail: 'dependency failed',
          title: 'Fix auth'
        }}
      />
    )

    expect(screen.getByText('task.blocked')).toBeTruthy()
    expect(screen.getByText('dependency failed')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Inspect blocker' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Approve review' })).toBeNull()
  })

  it.each(localizedLocales)('renders dialogue controls in %s', locale => {
    const copy = TRANSLATIONS[locale].lunarCity.dialogue
    render(
      <I18nProvider configClient={null} initialLocale={locale as Locale}>
        <DialogueTray
          onAction={vi.fn<WorldActionRunner['run']>(async () => ({ kind: 'completed', ok: true }))}
          onClose={vi.fn()}
          subject={{
            condition: {
              active: true,
              facts: {},
              id: 'condition-7',
              kind: 'task.blocked',
              scope: 'task',
              severity: 'warning',
              source: 'kanban',
              sourceRef: { taskId: 'task-7' },
              title: 'Fix auth'
            },
            title: 'Fix auth'
          }}
        />
      </I18nProvider>
    )

    expect(screen.getByText(copy.label)).toBeTruthy()
    expect(screen.getByRole('button', { name: copy.inspectBlocker })).toBeTruthy()
    expect(screen.getByRole('button', { name: copy.close })).toBeTruthy()
    expect(copy.label).not.toBe(TRANSLATIONS.en.lunarCity.dialogue.label)
  })
})
