import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $freeTierSignIn } from '@/store/free-tier-sign-in'

import { HostedNotice } from './hosted-notice'

afterEach(() => {
  cleanup()
  $freeTierSignIn.set({ status: 'closed' })
})

describe('the Connectors line for a guest', () => {
  it('offers a Sign in that starts the sign-in', () => {
    render(<HostedNotice hasGuest phase="ready" />)

    expect(screen.getByText('Connections stay on this computer until you sign in.')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Sign in' }))

    expect($freeTierSignIn.get().status).toBe('requested')
  })
})
