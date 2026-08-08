import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { AssistantAvatar } from '@/components/assistant-ui/thread/assistant-avatar'
import { $activeProfileName, $avatarDataUrl } from '@/store/avatar'

const PNG_URL = 'data:image/png;base64,iVBORw0KGgo=='

/** Query the avatar image — it uses alt="" and aria-hidden so getByRole won't find it. */
function queryAvatarImg(): HTMLImageElement | null {
  return document.querySelector('img[src^="data:image"]') as HTMLImageElement | null
}

describe('AssistantAvatar', () => {
  beforeEach(() => {
    $avatarDataUrl.set(null)
    $activeProfileName.set('default')
  })

  afterEach(() => {
    cleanup()
  })

  // ── Custom avatar (image) ───────────────────────────────────────────────

  it('renders an <img> when an avatar data URL is loaded', () => {
    $avatarDataUrl.set(PNG_URL)
    $activeProfileName.set('default')

    render(<AssistantAvatar />)

    const img = queryAvatarImg()
    expect(img).toBeTruthy()
    expect(img!.getAttribute('src')).toBe(PNG_URL)
  })

  it('renders the image inside a rounded wrapper', () => {
    $avatarDataUrl.set(PNG_URL)

    render(<AssistantAvatar />)

    const img = queryAvatarImg()
    const container = img!.parentElement
    expect(container?.className).toContain('rounded-full')
  })

  // ── Gradient fallback with letter ──────────────────────────────────────

  it('shows the first letter of the name when no avatar is set', () => {
    render(<AssistantAvatar name="Hermes" />)

    // Should NOT have an <img>
    expect(screen.queryByRole('img', { hidden: true })).toBeNull()
    // Should show the letter 'H'
    expect(screen.getByText('H')).toBeTruthy()
  })

  it('uppercases the fallback letter', () => {
    render(<AssistantAvatar name="haena" />)

    expect(screen.getByText('H')).toBeTruthy()
  })

  it('trims whitespace from name for fallback', () => {
    render(<AssistantAvatar name="  MiMo  " />)

    expect(screen.getByText('M')).toBeTruthy()
  })

  it('shows gradient letter even for single-character names', () => {
    render(<AssistantAvatar name="X" />)

    expect(screen.getByText('X')).toBeTruthy()
  })

  // ── Emoji fallback (no name, no avatar) ─────────────────────────────────

  it('shows 🤖 emoji when no name and no avatar are provided', () => {
    render(<AssistantAvatar />)

    expect(screen.getByText('🤖')).toBeTruthy()
  })

  it('shows 🤖 emoji when name is an empty string', () => {
    render(<AssistantAvatar name="" />)

    expect(screen.getByText('🤖')).toBeTruthy()
  })

  // ── Profile-aware rendering ────────────────────────────────────────────

  it('uses per-profile avatar from localStorage when profile has an override', () => {
    // Set up a per-profile override in localStorage
    const overrideUrl = 'data:image/png;base64,PROFILE=='
    localStorage.setItem('hermes.avatar.haena', overrideUrl)

    $avatarDataUrl.set(PNG_URL) // default
    $activeProfileName.set('haena') // switch to haena

    render(<AssistantAvatar />)

    const img = queryAvatarImg()
    expect(img!.getAttribute('src')).toBe(overrideUrl)
  })
})
