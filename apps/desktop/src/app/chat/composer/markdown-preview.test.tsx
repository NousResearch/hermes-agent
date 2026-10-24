// @vitest-environment jsdom
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { ComposerMarkdownPreview } from './markdown-preview'

describe('ComposerMarkdownPreview', () => {
  it('renders the current draft as static Markdown without promoting artifacts', async () => {
    render(
      <ComposerMarkdownPreview
        text={'# Release notes\n\n- **preview** the [draft](https://example.com/draft)\n- keep editing'}
      />
    )

    expect(await screen.findByRole('heading', { name: 'Release notes' })).toBeTruthy()
    expect(screen.getByText('preview')).toBeTruthy()
    expect(screen.getByText('draft').tagName).toBe('SPAN')
    expect(screen.getByText('keep editing')).toBeTruthy()
  })
})
