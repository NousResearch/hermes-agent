import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import FileTreeRenderer from './file-tree-embed'

afterEach(cleanup)

describe('FileTreeRenderer', () => {
  it('renders file entries from valid JSON', () => {
    render(
      <FileTreeRenderer
        code={JSON.stringify({
          title: 'Touched files',
          entries: [{ path: 'apps/desktop/src/store/workstream.ts', change: 'modified', note: 'metadata overlay' }]
        })}
      />
    )

    expect(screen.getByText('Touched files')).toBeTruthy()
    expect(screen.getByText('apps/desktop/src/store/workstream.ts')).toBeTruthy()
    expect(screen.getByText('modified')).toBeTruthy()
    expect(screen.getByText('metadata overlay')).toBeTruthy()
  })

  it('falls back on invalid JSON', () => {
    render(<FileTreeRenderer code='[]' />)

    expect(screen.getByText('Invalid file-tree block')).toBeTruthy()
  })

  it('renders attacker strings as text, not markup', () => {
    const { container } = render(
      <FileTreeRenderer
        code={JSON.stringify({ entries: [{ path: '<img src=x onerror=alert(1)>', change: '<script>alert(1)</script>' }] })}
      />
    )

    expect(screen.getByText('<img src=x onerror=alert(1)>')).toBeTruthy()
    expect(screen.getByText('<script>alert(1)</script>')).toBeTruthy()
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('script')).toBeNull()
  })
})
