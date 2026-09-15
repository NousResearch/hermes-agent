import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { setProfileColor } from '@/store/profile'

import { ProfileLead, profileLeadLabel } from './profile-lead'

afterEach(() => {
  cleanup()
  setProfileColor('inbox', null)
})

const lead = (container: HTMLElement) => container.querySelector<HTMLElement>('[data-profile-lead]') as HTMLElement

describe('profileLeadLabel', () => {
  it('prints a short name whole', () => {
    expect(profileLeadLabel('inbox')).toBe('inbox')
    expect(profileLeadLabel('fourteen-chars')).toBe('fourteen-chars')
  })

  it('elides a name past the cap so it cannot take the title room', () => {
    expect(profileLeadLabel('research-assistant')).toBe('research-assi…')
  })
})

describe('ProfileLead', () => {
  it('names the profile and hides the separator from assistive tech', () => {
    const { container } = render(<ProfileLead profile="inbox" selected={false} />)

    expect(lead(container).dataset.profileLead).toBe('inbox')
    expect(lead(container).textContent?.startsWith('inbox')).toBe(true)
    expect(lead(container).querySelector('[aria-hidden="true"]')?.textContent?.trim()).toBe('›')
  })

  it('carries the deterministic profile colour for the hover and selected states', () => {
    const { container } = render(<ProfileLead profile="inbox" selected={false} />)

    expect(lead(container).style.getPropertyValue('--profile-lead-color')).toMatch(/^hsl\(/)
  })

  it('prefers a user-picked colour override, the same one the rail square wears', () => {
    setProfileColor('inbox', 'hsl(30 68% 58%)')

    const { container } = render(<ProfileLead profile="inbox" selected={false} />)

    expect(lead(container).style.getPropertyValue('--profile-lead-color')).toBe('hsl(30 68% 58%)')
  })

  it('is quiet at rest and only colours on hover', () => {
    const { container } = render(<ProfileLead profile="inbox" selected={false} />)

    expect(lead(container).className).toContain('text-(--ui-text-quaternary)')
    expect(lead(container).className).toContain('group-hover:text-(--profile-lead-color)')
    expect(lead(container).className).not.toContain('font-medium')
  })

  it('keeps the colour while selected, with a little weight so it survives the pointer leaving', () => {
    const { container } = render(<ProfileLead profile="inbox" selected />)

    expect(lead(container).className).toContain('font-medium')
    expect(lead(container).className).not.toContain('group-hover:')
    expect(lead(container).className).not.toContain('text-(--ui-text-quaternary)')
  })
})
