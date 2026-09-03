import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { NotebookPreview } from './preview-notebook'

function notebook(cells: unknown[]) {
  return JSON.stringify({ nbformat: 4, cells })
}

describe('NotebookPreview', () => {
  afterEach(cleanup)

  it('renders markdown, code prompts, stdout, and saved figures', () => {
    const { container } = render(
      <NotebookPreview
        text={notebook([
          { cell_type: 'markdown', source: ['## Why XRP lags'] },
          {
            cell_type: 'code',
            execution_count: 1,
            source: ['print(1)'],
            outputs: [
              { output_type: 'stream', name: 'stdout', text: ['hello-stdout\n'] },
              { output_type: 'display_data', data: { 'image/png': 'QUJD' } }
            ]
          }
        ])}
      />
    )

    expect(screen.getByText('Why XRP lags')).toBeTruthy()
    expect(screen.getByText('In [1]:')).toBeTruthy()
    expect(screen.getByText('hello-stdout')).toBeTruthy()
    expect(container.querySelector('img')?.getAttribute('src')).toBe('data:image/png;base64,QUJD')
  })

  it('strips scripts from HTML outputs and names widgets', () => {
    render(
      <NotebookPreview
        text={notebook([
          {
            cell_type: 'code',
            execution_count: 2,
            source: '',
            outputs: [
              {
                output_type: 'display_data',
                data: { 'text/html': '<p>safe</p><script>window.pwned = 1</script>' }
              },
              {
                output_type: 'display_data',
                data: { 'application/vnd.jupyter.widget-view+json': { version_major: 2 } }
              }
            ]
          }
        ])}
      />
    )

    expect(screen.getByText('safe')).toBeTruthy()
    expect(screen.queryByText(/pwned/)).toBeNull()
    expect(screen.getByText('Interactive widget (not shown)')).toBeTruthy()
  })

  it('shows a parse error for non-notebook JSON', () => {
    render(<NotebookPreview text={'{"hello": 1}'} />)

    expect(screen.getByText('Not a Jupyter notebook')).toBeTruthy()
  })
})
