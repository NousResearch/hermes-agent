import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

/**
 * Faithful mirror of chat/sidebar/index.tsx's new-session button
 * visibility logic (issue #92070). Isolated from the full sidebar
 * component (1800+ lines, many external dependencies) -- mirrors just
 * the conditional that used to hide the button in the "All profiles"
 * aggregated view.
 *
 * `gated` toggles between the pre-fix and post-fix behavior so this
 * test can prove the difference, rather than just asserting on
 * hardcoded post-fix behavior with nothing to distinguish it from.
 */
function NewSessionButtonHarness({
  agentsGrouped,
  gated,
  onNewSessionInWorkspace,
  onOpenProjectCreate,
  showAllProfiles,
}: {
  agentsGrouped: boolean
  gated: boolean
  onNewSessionInWorkspace: (path: null | string) => void
  onOpenProjectCreate: () => void
  showAllProfiles: boolean
}) {
  const button = (
    <button
      aria-label="new-session"
      onClick={event => {
        event.stopPropagation()

        if (agentsGrouped) {
          onOpenProjectCreate()
        } else {
          onNewSessionInWorkspace(null)
        }
      }}
    >
      +
    </button>
  )

  // `gated={true}` reproduces the ORIGINAL bug: `{!showAllProfiles ? (...) : null}`.
  // `gated={false}` is the fix: the button is no longer conditional on showAllProfiles.
  return gated ? (!showAllProfiles ? button : null) : button
}

describe('sidebar new-session button visibility (issue #92070)', () => {
  it('renders in the default (single-profile) view regardless of gating', () => {
    for (const gated of [true, false]) {
      const { unmount } = render(
        <NewSessionButtonHarness
          agentsGrouped={false}
          gated={gated}
          onNewSessionInWorkspace={() => undefined}
          onOpenProjectCreate={() => undefined}
          showAllProfiles={false}
        />
      )

      expect(screen.queryByLabelText('new-session')).toBeTruthy()
      unmount()
    }
  })

  it('the pre-fix gated behavior hides the button in the "All profiles" view (sanity: proves this harness actually reproduces the reported bug)', () => {
    render(
      <NewSessionButtonHarness
        agentsGrouped={false}
        gated={true}
        onNewSessionInWorkspace={() => undefined}
        onOpenProjectCreate={() => undefined}
        showAllProfiles={true}
      />
    )

    expect(screen.queryByLabelText('new-session')).toBeNull()
  })

  it('the fix keeps the button visible in the "All profiles" aggregated view', () => {
    render(
      <NewSessionButtonHarness
        agentsGrouped={false}
        gated={false}
        onNewSessionInWorkspace={() => undefined}
        onOpenProjectCreate={() => undefined}
        showAllProfiles={true}
      />
    )

    expect(screen.queryByLabelText('new-session')).toBeTruthy()
  })

  it('clicking the button lands the new session in the active profile via onNewSessionInWorkspace(null)', async () => {
    const { fireEvent } = await import('@testing-library/react')
    let called: null | string | undefined

    render(
      <NewSessionButtonHarness
        agentsGrouped={false}
        gated={false}
        onNewSessionInWorkspace={path => {
          called = path
        }}
        onOpenProjectCreate={() => undefined}
        showAllProfiles={true}
      />
    )

    fireEvent.click(screen.getByLabelText('new-session'))

    expect(called).toBe(null)
  })
})
