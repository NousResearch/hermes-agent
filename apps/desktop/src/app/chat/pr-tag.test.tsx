import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { HermesBranchPullRequest } from '@/global'

import { PrTag } from './pr-tag'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

const pr: HermesBranchPullRequest = {
  branch: 'fix',
  draft: false,
  number: 42,
  state: 'open',
  title: 'Restore session metadata',
  url: 'https://github.com/example/project/pull/42'
}

it('keeps PR identity and state while only the sidebar opts out of the number', () => {
  const { rerender } = render(<PrTag pr={pr} showIcon={false} />)
  const button = () => screen.getByRole('button', { name: 'Open pull request #42' })
  expect(button().textContent).toBe('#42')
  expect(button().querySelector('.codicon')).toBeNull()

  for (const [state, draft, icon, color] of [
    ['open', false, 'git-pull-request', 'green'],
    ['merged', false, 'git-merge', 'purple'],
    ['closed', false, 'git-pull-request-closed', 'red'],
    ['open', true, 'git-pull-request-draft', 'text-quaternary']
  ] as const) {
    rerender(<PrTag iconOnly pr={{ ...pr, draft, state }} />)
    expect(button().textContent).toBe('')
    expect(button().querySelector(`.codicon-${icon}`)).not.toBeNull()
    expect(button().classList.contains(`text-(--ui-${color})`)).toBe(true)
    expect(button().getAttribute('data-slot')).toBe('tooltip-trigger')
    expect(button().getAttribute('title')).toBeNull()
  }

  rerender(<PrTag pr={pr} />)
  expect(button().textContent).toBe('42')
})

it('opens an icon-only PR without selecting or pinning the containing session', () => {
  const openExternal = vi.fn()
  const rowClick = vi.fn()
  const rowPointerDown = vi.fn()
  vi.stubGlobal('hermesDesktop', { openExternal })
  render(
    <div onClick={rowClick} onPointerDown={rowPointerDown}>
      <PrTag iconOnly pr={pr} />
    </div>
  )
  const button = screen.getByRole('button', { name: 'Open pull request #42' })
  fireEvent.pointerDown(button)
  fireEvent.click(button, { shiftKey: true })
  expect(openExternal).toHaveBeenCalledWith(pr.url)
  expect(rowPointerDown).not.toHaveBeenCalled()
  expect(rowClick).not.toHaveBeenCalled()
})
