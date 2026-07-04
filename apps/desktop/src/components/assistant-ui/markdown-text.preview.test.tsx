import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $previewTarget } from '@/store/preview'

import { MarkdownTextContent } from './markdown-text'

const marker = '[Preview: report.html](#preview/%2Ftmp%2Fhermes-preview%2Freport.html)'

const target = '/tmp/hermes-preview/report.html'

afterEach(() => {
  cleanup()
  $previewTarget.set(null)
})

describe('MarkdownTextContent preview markers', () => {
  it('renders #preview markdown links as native preview cards and opens the in-app preview', async () => {
    render(<MarkdownTextContent isRunning={false} text={marker} />)

    const button = await screen.findByRole('button', { name: /open preview/i })
    expect(screen.getByText('report.html')).toBeTruthy()

    fireEvent.click(button)

    await waitFor(() => {
      expect($previewTarget.get()).toMatchObject({
        path: target,
        previewKind: 'html',
        renderMode: 'preview',
        source: target,
        url: `file://${target}`
      })
    })
  })

  it('defers preview cards while text is still streaming', () => {
    render(<MarkdownTextContent isRunning={true} text={marker} />)

    expect(screen.queryByRole('button', { name: /open preview/i })).toBeNull()
    expect($previewTarget.get()).toBeNull()
  })
})
