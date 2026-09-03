import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { ProcessNotificationNote } from './user-message'

afterEach(cleanup)

describe('ProcessNotificationNote', () => {
  it('keeps completed background output behind one compact disclosure', () => {
    const { container } = render(
      <ProcessNotificationNote
        text={'[IMPORTANT: Background process proc_123 completed normally (exit code 0).\nCommand: hermes chat\nOutput: done]'}
      />
    )

    const details = container.querySelector('details')

    expect(details?.open).toBe(false)
    expect(screen.getByText('Agent work completed')).toBeTruthy()
    expect(screen.getByText('Show details')).toBeTruthy()

    fireEvent.click(container.querySelector('summary') as HTMLElement)

    expect(details?.open).toBe(true)
    expect(screen.getByText(/Command: hermes chat/)).toBeTruthy()
  })
})
