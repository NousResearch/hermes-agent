import { afterEach, describe, expect, it, vi } from 'vitest'

import { host } from '@/sdk'

const desktopWindow = window as unknown as { hermesDesktop?: Partial<Window['hermesDesktop']> }

afterEach(() => {
  delete desktopWindow.hermesDesktop
})

describe('host.spellcheck dictionary verbs', () => {
  it("routes one word through the app's dictionary IPC and reports main's verdict", async () => {
    const spellcheckDictionary = vi.fn(async () => true)

    desktopWindow.hermesDesktop = { spellcheckDictionary }

    expect(await host.spellcheck.addWord('Hermes')).toBe(true)
    expect(await host.spellcheck.removeWord('Hermes')).toBe(true)

    expect(spellcheckDictionary.mock.calls).toEqual([
      ['add', 'Hermes'],
      ['remove', 'Hermes']
    ])

    spellcheckDictionary.mockResolvedValueOnce(false)

    expect(await host.spellcheck.addWord('two words')).toBe(false)
  })

  it('resolves false instead of throwing without a desktop shell or with a non-string', async () => {
    expect(await host.spellcheck.addWord('Hermes')).toBe(false)

    const spellcheckDictionary = vi.fn(async () => {
      throw new Error('older shell')
    })

    desktopWindow.hermesDesktop = { spellcheckDictionary }

    expect(await host.spellcheck.removeWord('Hermes')).toBe(false)
    expect(await host.spellcheck.addWord(42 as unknown as string)).toBe(false)
    expect(spellcheckDictionary).toHaveBeenCalledTimes(1)
  })
})
