import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { ComposerTriggerPopover } from './trigger-popover'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      composer: {
        lookupLoading: 'Loading…',
        lookupNoMatches: 'No matches',
        lookupTry: 'Try',
        lookupOr: 'or'
      }
    }
  })
}))

function atItem(type: string, display: string, rawText: string, meta = '') {
  return {
    id: `${rawText}|0`,
    type,
    label: display,
    metadata: { icon: type, display, meta, rawText, insertId: display }
  }
}

function slashItem(command: string, group: string, meta = '') {
  return {
    id: `${command}|0`,
    type: 'slash',
    label: command.slice(1),
    metadata: { command, display: command, meta, group, action: '', rawText: command }
  }
}

const noop = () => {}

/** The rendered shape of one row: does it have an icon, and how is it laid out?
 *  Icons are codicons — an `<i class="codicon codicon-<name>">`, not an SVG. */
function rowShape(root: HTMLElement) {
  const row = root.querySelector('button') as HTMLElement
  const icon = row.querySelector('i.codicon')

  return {
    hasIcon: Boolean(icon),
    iconName: icon?.className.match(/codicon-([\w-]+)/)?.[1],
    classes: row.className
  }
}

describe('@ and / are one menu', () => {
  it('a slash row has an icon, just like an @ row', () => {
    const at = render(
      <ComposerTriggerPopover
        activeIndex={0}
        items={[atItem('folder', 'apps/desktop/', '@folder:apps/desktop/', 'dir')]}
        kind="@"
        loading={false}
        onHover={noop}
        onPick={noop}
      />
    )

    const atShape = rowShape(at.container)
    at.unmount()

    const slash = render(
      <ComposerTriggerPopover
        activeIndex={0}
        items={[slashItem('/work', 'Skills', 'Start in a worktree')]}
        kind="/"
        loading={false}
        onHover={noop}
        onPick={noop}
      />
    )

    const slashShape = rowShape(slash.container)

    // The whole point: `/` used to render a stacked, icon-less row.
    expect(slashShape.hasIcon).toBe(true)
    expect(atShape.hasIcon).toBe(true)
    expect(slashShape.classes).toBe(atShape.classes)

    // And the glyph reflects the kind, not one generic bullet.
    expect(slashShape.iconName).toBe('zap')
    expect(atShape.iconName).toBe('folder')
  })

  it('renders the name and description for both kinds', () => {
    const { rerender } = render(
      <ComposerTriggerPopover
        activeIndex={0}
        items={[slashItem('/work', 'Skills', 'Start in a worktree')]}
        kind="/"
        loading={false}
        onHover={noop}
        onPick={noop}
      />
    )

    expect(screen.getByText('/work')).toBeTruthy()
    expect(screen.getByText('Start in a worktree')).toBeTruthy()

    rerender(
      <ComposerTriggerPopover
        activeIndex={0}
        items={[atItem('file', 'src/main.tsx', '@file:src/main.tsx', 'src')]}
        kind="@"
        loading={false}
        onHover={noop}
        onPick={noop}
      />
    )

    expect(screen.getByText('src/main.tsx')).toBeTruthy()
    expect(screen.getByText('src')).toBeTruthy()
  })

  it('an emoji row stays icon-less — the emoji IS the icon', () => {
    const { container } = render(
      <ComposerTriggerPopover
        activeIndex={0}
        items={[{ id: ':joy:|0', type: 'emoji', label: '😂  :joy:', metadata: { display: '😂  :joy:' } }]}
        kind=":"
        loading={false}
        onHover={noop}
        onPick={noop}
      />
    )

    expect(rowShape(container).hasIcon).toBe(false)
  })

  it('labels the active browse scope from the shared vocabulary', () => {
    render(
      <ComposerTriggerPopover
        activeIndex={0}
        items={[atItem('folder', 'apps/', '@folder:apps/', 'dir')]}
        kind="@"
        loading={false}
        onHover={noop}
        onPick={noop}
        scope="folder"
      />
    )

    expect(screen.getByText('Folders')).toBeTruthy()
  })
})
