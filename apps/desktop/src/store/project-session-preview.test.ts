import { beforeEach, describe, expect, it } from 'vitest'

import { $projectSessionPreview, setProjectSessionPreview } from './project-session-preview'

describe('project session preview setting', () => {
  beforeEach(() => {
    window.localStorage.clear()
    setProjectSessionPreview(3)
  })

  it('defaults to the shipped preview count of 3', () => {
    expect($projectSessionPreview.get()).toBe(3)
  })

  it('persists a chosen numeric count', () => {
    setProjectSessionPreview(8)

    expect($projectSessionPreview.get()).toBe(8)
    expect(window.localStorage.getItem('hermes.desktop.projectSessionPreview')).toBe('8')
  })

  it('persists the uncapped "all" value', () => {
    setProjectSessionPreview('all')

    expect($projectSessionPreview.get()).toBe('all')
    expect(window.localStorage.getItem('hermes.desktop.projectSessionPreview')).toBe('all')
  })

  it('decodes corrupt stored values back to the default', () => {
    window.localStorage.setItem('hermes.desktop.projectSessionPreview', '9999')

    expect($projectSessionPreview.get()).toBe(3)
  })
})
