import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $artifactRegistry, $artifactVersionSelection, artifactPreviewTarget, upsertArtifact } from '@/store/artifacts'

import { ArtifactPreview } from './preview-artifact'

function register(title: string, kind: 'code' | 'html' | 'svg', content: string) {
  const result = upsertArtifact('session-1', { kind, language: kind === 'code' ? 'python' : kind, title }, content)

  if (!result) {
    throw new Error('artifact did not register')
  }

  return result
}

async function renderArtifact(artifactId: string) {
  const record = $artifactRegistry.get()['session-1']!.find(item => item.id === artifactId)!

  await act(async () => {
    render(<ArtifactPreview target={artifactPreviewTarget(record)} />)
  })
}

describe('ArtifactPreview', () => {
  afterEach(() => {
    cleanup()
    $artifactRegistry.set({})
    $artifactVersionSelection.set({})
  })

  it('renders html in a scripts-only sandboxed frame the parent app is unreachable from', async () => {
    const { artifactId } = register('Dashboard', 'html', '<h1>Hi</h1>')
    await renderArtifact(artifactId)

    const frame = screen.getByTitle('Dashboard') as HTMLIFrameElement

    expect(frame.getAttribute('sandbox')).toBe('allow-scripts')
    expect(frame.srcdoc).toContain('<h1>Hi</h1>')
    // No allow-same-origin: scripts inside cannot reach the renderer's origin.
    expect(frame.getAttribute('sandbox')).not.toContain('same-origin')
  })

  it('strips scripts out of svg before it renders inline', async () => {
    const { artifactId } = register(
      'Logo',
      'svg',
      '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
    )

    await renderArtifact(artifactId)

    expect(document.querySelector('svg')).not.toBeNull()
    expect(document.querySelector('svg script')).toBeNull()
  })

  it('offers only the source view for code, which has nothing to render', async () => {
    const { artifactId } = register('Solver', 'code', 'print("hi")')
    await renderArtifact(artifactId)

    expect(screen.queryByRole('button', { name: /rendered/i })).toBeNull()
  })

  it('shows the version stepper once an artifact has history, and follows the selection', async () => {
    register('Dashboard', 'html', '<h1>v1</h1>')
    const { artifactId } = register('Dashboard', 'html', '<h1>v2</h1>')
    await renderArtifact(artifactId)

    expect(screen.getByText('v2 of 2')).toBeTruthy()
    expect((screen.getByTitle('Dashboard') as HTMLIFrameElement).srcdoc).toContain('v2')

    await act(async () => {
      $artifactVersionSelection.set({ [artifactId]: 0 })
    })

    expect(screen.getByText('v1 of 2')).toBeTruthy()
    expect((screen.getByTitle('Dashboard') as HTMLIFrameElement).srcdoc).toContain('v1')
  })

  it('hides the stepper for a single-version artifact', async () => {
    const { artifactId } = register('Dashboard', 'html', '<h1>only</h1>')
    await renderArtifact(artifactId)

    expect(screen.queryByText('v1 of 1')).toBeNull()
  })

  it('picks up a new version in an already-open tab', async () => {
    const { artifactId } = register('Dashboard', 'html', '<h1>v1</h1>')
    await renderArtifact(artifactId)

    await act(async () => {
      register('Dashboard', 'html', '<h1>v2</h1>')
    })

    expect((screen.getByTitle('Dashboard') as HTMLIFrameElement).srcdoc).toContain('v2')
  })

  it('falls back to an empty state when the registry no longer has the artifact', async () => {
    const { artifactId } = register('Dashboard', 'html', '<h1>gone</h1>')
    const record = $artifactRegistry.get()['session-1']!.find(item => item.id === artifactId)!
    const target = artifactPreviewTarget(record)

    $artifactRegistry.set({})

    await act(async () => {
      render(<ArtifactPreview target={target} />)
    })

    expect(screen.queryByTitle('Dashboard')).toBeNull()
  })
})
