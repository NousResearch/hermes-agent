import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $connection } from '@/store/session'

import { FileTreeBody } from './index'

describe('FileTreeBody remote gateway guard', () => {
  afterEach(() => {
    $connection.set(null)
  })

  it('shows remote gateway message when connected to a remote gateway', () => {
    $connection.set({ mode: 'remote' } as never)

    render(
      <FileTreeBody
        collapseNonce={0}
        cwd="/root/hermes"
        data={[]}
        error={null}
        loading={false}
        onActivateFile={() => {}}
        onActivateFolder={() => {}}
        onLoadChildren={() => {}}
        onNodeOpenChange={() => {}}
        openState={{}}
      />
    )

    expect(screen.getByText('Remote gateway')).toBeTruthy()
    expect(screen.getByText(/File browser is not available/)).toBeTruthy()
  })

  it('shows unreadable error in local mode when error is present', () => {
    $connection.set({ mode: 'local' } as never)

    render(
      <FileTreeBody
        collapseNonce={0}
        cwd="/some/path"
        data={[]}
        error="ENOENT"
        loading={false}
        onActivateFile={() => {}}
        onActivateFolder={() => {}}
        onLoadChildren={() => {}}
        onNodeOpenChange={() => {}}
        openState={{}}
      />
    )

    expect(screen.getByText('Unreadable')).toBeTruthy()
  })
})
