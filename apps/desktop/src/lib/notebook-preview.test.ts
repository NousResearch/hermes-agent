import { describe, expect, it } from 'vitest'

import { cleanStreamText, isNotebookPath, parseNotebook } from './notebook-preview'

describe('isNotebookPath', () => {
  it('matches .ipynb regardless of case and query strings', () => {
    expect(isNotebookPath('/tmp/analysis.ipynb')).toBe(true)
    expect(isNotebookPath('C:\\Users\\a\\Why.IPYNB')).toBe(true)
    expect(isNotebookPath('/tmp/analysis.ipynb?x=1')).toBe(true)
    expect(isNotebookPath('/tmp/analysis.py')).toBe(false)
    expect(isNotebookPath('/tmp/notebook.json')).toBe(false)
  })
})

describe('parseNotebook', () => {
  it('returns null for invalid JSON or a non-notebook object', () => {
    expect(parseNotebook('not json')).toBeNull()
    expect(parseNotebook('[]')).toBeNull()
    expect(parseNotebook('{"nbformat": 4}')).toBeNull()
  })

  it('joins source lists and renders markdown, code, stdout, and images', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        metadata: { kernelspec: { language: 'python' } },
        nbformat: 4,
        cells: [
          { cell_type: 'markdown', source: ['# Title\n', 'body'] },
          {
            cell_type: 'code',
            execution_count: 3,
            source: ['print(1)\n'],
            outputs: [
              { output_type: 'stream', name: 'stdout', text: ['1\n'] },
              {
                output_type: 'display_data',
                data: { 'image/png': 'QUJD\n' }
              }
            ]
          }
        ]
      })
    )

    expect(notebook?.language).toBe('python')
    expect(notebook?.cells).toHaveLength(2)
    expect(notebook?.cells[0]).toMatchObject({ kind: 'markdown', source: '# Title\nbody' })
    expect(notebook?.cells[1]).toMatchObject({ kind: 'code', executionCount: 3, source: 'print(1)\n' })
    expect(notebook?.cells[1].outputs).toEqual([
      { type: 'stream', name: 'stdout', text: '1\n' },
      { type: 'image', mime: 'image/png', dataUrl: 'data:image/png;base64,QUJD' }
    ])
  })

  it('prefers a figure over text/plain twins and sanitizes stream rewrites', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        cells: [
          {
            cell_type: 'code',
            source: '',
            outputs: [
              {
                output_type: 'display_data',
                data: {
                  'text/plain': '<Figure>',
                  'image/png': 'QUJD'
                }
              },
              { output_type: 'stream', name: 'stdout', text: 'aa\rbb\n' }
            ]
          }
        ]
      })
    )

    expect(notebook?.cells[0].outputs[0].type).toBe('image')
    expect(notebook?.cells[0].outputs[1]).toEqual({ type: 'stream', name: 'stdout', text: 'bb\n' })
  })

  it('keeps errors, HTML, widgets, and nbformat v3 cells', () => {
    const notebook = parseNotebook(
      JSON.stringify({
        nbformat: 3,
        worksheets: [
          {
            cells: [
              {
                cell_type: 'code',
                source: ['1'],
                outputs: [
                  { output_type: 'pyerr', ename: 'ValueError', evalue: 'boom', traceback: ['\u001b[31mboom\u001b[0m'] },
                  { output_type: 'pyout', html: ['<b>ok</b>'] },
                  {
                    output_type: 'display_data',
                    data: { 'application/vnd.jupyter.widget-view+json': { version_major: 2 } }
                  }
                ]
              }
            ]
          }
        ]
      })
    )

    expect(notebook?.cells).toHaveLength(1)
    expect(notebook?.cells[0].outputs[0]).toMatchObject({ type: 'error', ename: 'ValueError', traceback: 'boom' })
    expect(notebook?.cells[0].outputs[1]).toEqual({ type: 'html', html: '<b>ok</b>' })
    expect(notebook?.cells[0].outputs[2]).toEqual({ type: 'widget' })
  })
})

describe('cleanStreamText', () => {
  it('keeps the last carriage-return frame and strips ANSI', () => {
    expect(cleanStreamText('progress\r100%\n')).toBe('100%\n')
    expect(cleanStreamText('\u001b[31mred\u001b[0m')).toBe('red')
  })
})
