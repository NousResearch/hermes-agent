// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { en } from '@/i18n/en'

afterEach(cleanup)

// The temporary-session badge is the only thing standing between a user and a
// false belief about what is being written to disk. The backend guarantee is
// covered by tests/hermes_cli/test_ephemeral_sessions.py plus a filesystem E2E;
// what those cannot catch is the badge silently failing to *render*, which
// would leave a temporary chat visually indistinguishable from a normal one.
//
// Two regressions motivate the contrast assertions below:
//
//   1. The badge fill was originally `bg-amber-500/12` for both themes. At 12%
//      opacity over the light surface it washed out to near-invisible (measured
//      1.04:1 against the page). Light mode now gets an opaque `bg-amber-50`
//      with `text-amber-900`.
//   2. The first attempted fix over-corrected dark mode to `bg-amber-950`, a
//      dark brown that read as an error state rather than "temporary". Dark now
//      uses a translucent `dark:bg-amber-400/15` with `dark:text-amber-200`.
//
// Both themes must therefore carry their own explicit fill + text colour. A
// single shared translucent fill is the exact bug this guards against.

/** Mirror of the badge in index.tsx (~L1088). Kept in sync deliberately: the
 *  full ChatBar drags in the assistant-ui runtime, Electron IPC bridges and a
 *  session store, none of which this contract needs. */
function TemporaryBadge({ ephemeral }: { ephemeral: boolean }) {
  if (!ephemeral) {return null}

  return (
    <div
      aria-live="polite"
      className="flex w-full items-center gap-2 rounded-md border border-amber-600/55 bg-amber-50 px-2.5 py-1.5 text-[0.6875rem] font-medium text-amber-900 dark:border-amber-400/50 dark:bg-amber-400/15 dark:text-amber-200"
      data-testid="temporary-session-indicator"
    >
      <span className="font-semibold">{en.composer.temporarySessionBadge}</span>
      <span className="truncate font-normal opacity-80">{en.composer.temporarySessionHint}</span>
    </div>
  )
}

describe('temporary session indicator', () => {
  it('renders nothing for a normal session', () => {
    render(<TemporaryBadge ephemeral={false} />)
    expect(screen.queryByTestId('temporary-session-indicator')).toBeNull()
  })

  it('renders the badge for a temporary session', () => {
    render(<TemporaryBadge ephemeral />)
    expect(screen.getByTestId('temporary-session-indicator')).not.toBeNull()
  })

  it('states plainly that nothing is saved', () => {
    render(<TemporaryBadge ephemeral />)
    const badge = screen.getByTestId('temporary-session-indicator')

    // The copy is the promise. If it drifts to something vaguer the user loses
    // the only on-screen statement of what temporary mode actually does.
    expect(badge.textContent).toContain('not saved')
    // Says WHEN the chat disappears, not just that it isn't saved -- the
    // question someone deciding whether to trust the mode actually has.
    expect(badge.textContent).toContain('Gone when you close this chat.')
  })

  it('announces itself to screen readers', () => {
    render(<TemporaryBadge ephemeral />)
    // Privacy state must not be a purely visual signal.
    expect(screen.getByTestId('temporary-session-indicator').getAttribute('aria-live')).toBe(
      'polite'
    )
  })

  it('carries an opaque light-mode fill and a separate dark-mode fill', () => {
    render(<TemporaryBadge ephemeral />)
    const className = screen.getByTestId('temporary-session-indicator').className

    // Light: opaque cream, not a translucent wash that disappears on white.
    expect(className).toContain('bg-amber-50')
    expect(className).toContain('text-amber-900')

    // Dark: translucent amber, NOT the amber-950 brown that read as an error.
    expect(className).toContain('dark:bg-amber-400/15')
    expect(className).toContain('dark:text-amber-200')
    expect(className).not.toContain('bg-amber-950')

    // The original single-fill bug: one translucent fill shared by both themes.
    expect(className).not.toContain('bg-amber-500/12')
  })

  it('has not drifted from the real badge in index.tsx', async () => {
    // The mirror above is a copy, so it can rot: someone edits the real badge,
    // this file keeps asserting the old classes and passes while the shipped UI
    // regresses. Read the component source and require an exact match.
    const fs = await import('node:fs/promises')
    const url = await import('node:url')
    const path = await import('node:path')

    const here = path.dirname(url.fileURLToPath(import.meta.url))
    const source = await fs.readFile(path.join(here, 'index.tsx'), 'utf8')

    const marker = 'data-testid="temporary-session-indicator"'
    expect(source).toContain(marker)

    const idx = source.indexOf(marker)
    const preceding = source.slice(Math.max(0, idx - 1200), idx)
    const live = [...preceding.matchAll(/className="([^"]*amber[^"]*)"/g)].map((m) => m[1]).at(-1)

    render(<TemporaryBadge ephemeral />)
    expect(live).toBe(screen.getByTestId('temporary-session-indicator').className)
  })

  it('uses the incognito (spy) icon, not a padlock', async () => {
    // A padlock means "encrypted/secure", which is a different and misleading
    // promise: a temporary chat is not more secure in transit, it simply is not
    // written down. The spy glyph is the established incognito/private-browsing
    // convention, so it sets the right expectation at a glance.
    const fs = await import('node:fs/promises')
    const url = await import('node:url')
    const path = await import('node:path')

    const here = path.dirname(url.fileURLToPath(import.meta.url))
    const source = await fs.readFile(path.join(here, 'index.tsx'), 'utf8')

    const idx = source.indexOf('data-testid="temporary-session-indicator"')
    const badgeBlock = source.slice(idx, idx + 400)

    expect(badgeBlock).toContain('<IconSpy')
    expect(badgeBlock).not.toContain('name="lock"')
    expect(source).toContain("import { IconSpy } from '@tabler/icons-react'")
  })
})
