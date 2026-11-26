// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { en } from '@/i18n/en'

import { InterfaceModeSettings } from './interface-mode-settings'

afterEach(cleanup)

describe('InterfaceModeSettings', () => {
  it('mounts outside its feedback provider without throwing', () => {
    expect(() => render(<InterfaceModeSettings />)).not.toThrow()
    expect(screen.getByText(en.simplifiedAgents.title)).toBeTruthy()
  })
})
