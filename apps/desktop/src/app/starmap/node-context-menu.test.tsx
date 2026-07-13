import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { NodeContextMenu, type NodeMenuTarget } from './node-context-menu'

vi.mock('@/hermes', () => ({
  deleteLearningNode: vi.fn(),
  editLearningNode: vi.fn(),
  getLearningNode: vi.fn(),
  setApiRequestProfile: vi.fn()
}))

vi.mock('@/store/starmap', () => ({
  evictStarmapNode: vi.fn(() => vi.fn()),
  loadStarmapGraph: vi.fn()
}))

vi.mock('../hooks/use-on-profile-switch', () => ({
  useOnProfileSwitch: vi.fn()
}))

function target(kind: NodeMenuTarget['kind']): NodeMenuTarget {
  return {
    id: `${kind}-1`,
    kind,
    label: kind === 'skill' ? 'مهارة الاختبار' : 'ذاكرة الاختبار',
    x: 24,
    y: 32
  }
}

function renderMenu(kind: NodeMenuTarget['kind']) {
  return render(
    <I18nProvider configClient={null} initialLocale="ar">
      <NodeContextMenu onClose={vi.fn()} onNodeRemoved={vi.fn()} target={target(kind)} />
    </I18nProvider>
  )
}

describe('NodeContextMenu Arabic copy', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('localizes memory actions and the destructive confirmation', () => {
    renderMenu('memory')

    expect(screen.getByRole('button', { name: 'تحرير الذاكرة…' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'حذف الذاكرة' }))

    expect(screen.getByRole('dialog', { name: 'حذف ذاكرة الاختبار؟' })).toBeTruthy()
    expect(screen.getByText('ستُحذف هذه الذاكرة نهائيا.')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'إلغاء' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'حذف' })).toBeTruthy()
  })

  it('localizes skill actions and the archive confirmation', () => {
    renderMenu('skill')

    expect(screen.getByRole('button', { name: 'تحرير المهارة…' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'أرشفة المهارة' }))

    expect(screen.getByRole('dialog', { name: 'أرشفة مهارة الاختبار؟' })).toBeTruthy()
    expect(screen.getByText(/ستُؤرشف المهارة/)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'أرشفة' })).toBeTruthy()
  })
})
