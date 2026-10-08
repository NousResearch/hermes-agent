import type { ConnectorAccountRow, ConnectorAccountsRenameParams, ConnectorAccountsResult } from '@hermes/shared'
import { JsonRpcGatewayError } from '@hermes/shared'
import { QueryClientProvider, useQuery } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { setPrimaryGateway } from '@/store/gateway'

import { type AccountEdit, AccountsSection } from './accounts-section'
import { connectorsAccountsQueryKey } from './data/keys'
import { useRenameAccount } from './data/mutations'

const SCOPE = 'default'

const ACCOUNTS: ConnectorAccountsResult = {
  accounts: [
    {
      active: true,
      alias: 'work',
      connection_id: 'ca_work',
      connector: 'gmail',
      created_at: '2026-10-05T00:00:00Z',
      label: 'Gmail account',
      status: 'active',
      updated_at: '2026-10-05T00:00:00Z'
    },
    {
      active: true,
      alias: 'home',
      connection_id: 'ca_home',
      connector: 'gmail',
      created_at: '2026-10-01T00:00:00Z',
      label: 'Gmail account',
      status: 'active',
      updated_at: '2026-10-01T00:00:00Z'
    }
  ]
}

function Harness() {
  // A refetch never answers, so only the rename's own reconcile or rollback can change the row.
  const accounts = useQuery({
    queryFn: () => new Promise<ConnectorAccountsResult>(() => undefined),
    queryKey: connectorsAccountsQueryKey(SCOPE),
    staleTime: Number.POSITIVE_INFINITY
  })

  const renamer = useRenameAccount(SCOPE)
  const [edit, setEdit] = useState<AccountEdit | null>(null)

  return (
    <AccountsSection
      accounts={accounts.data?.accounts ?? []}
      connecting={false}
      edit={edit}
      onAdd={async () => ({ ok: true })}
      onEditChange={setEdit}
      onReconnect={() => undefined}
      onRemove={() => undefined}
      onRename={renamer.rename}
      reconnecting={false}
      retired={[]}
    />
  )
}

function renderSection() {
  queryClient.setQueryData(connectorsAccountsQueryKey(SCOPE), ACCOUNTS)

  return render(
    <QueryClientProvider client={queryClient}>
      <I18nProvider>
        <Harness />
      </I18nProvider>
    </QueryClientProvider>
  )
}

async function renameWork(to: string) {
  fireEvent.click(screen.getAllByRole('button', { name: 'Rename' })[0])
  fireEvent.change(screen.getByRole('textbox', { name: 'New name for work' }), { target: { value: to } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
}

/** The primary socket answers `connectors.accounts.rename` with `rename`; every other method stays unanswered. */
function serveRename(rename: () => Promise<ConnectorAccountRow>) {
  const request = vi.fn((method: string, _params: ConnectorAccountsRenameParams) =>
    method === 'connectors.accounts.rename' ? rename() : new Promise(() => undefined)
  )

  // SAFETY: the connector RPC layer calls only `request` on the primary socket in these tests.
  setPrimaryGateway({ request } as never)

  return request
}

const renameCalls = (request: ReturnType<typeof serveRename>) =>
  request.mock.calls.filter(([method]) => method === 'connectors.accounts.rename').map(([, params]) => params)

afterEach(() => {
  cleanup()
  queryClient.clear()
  setPrimaryGateway(null)
})

describe('AccountsSection rename', () => {
  it('sends the account id and the new name, and shows the new name before the answer', async () => {
    let answer: (row: ConnectorAccountRow) => void = () => undefined

    const request = serveRename(
      () =>
        new Promise(resolve => {
          answer = resolve
        })
    )

    renderSection()
    await renameWork('office')

    await waitFor(() => expect(screen.getByText('office')).toBeTruthy())
    expect(renameCalls(request)).toEqual([{ alias: 'office', connection_id: 'ca_work' }])

    answer({ ...ACCOUNTS.accounts[0], alias: 'office' })
    await waitFor(() => expect(screen.getByText('office')).toBeTruthy())
  })

  it('restores the old name and says why when the gateway refuses the name', async () => {
    serveRename(() =>
      Promise.reject(
        new JsonRpcGatewayError('That name is already used.', { code: 4090, data: { reason: 'ALIAS_TAKEN' } })
      )
    )

    renderSection()
    await renameWork('office')

    await waitFor(() => expect(screen.getByRole('alert').textContent).toBe('That name is already used.'))
    expect(
      queryClient.getQueryData<ConnectorAccountsResult>(connectorsAccountsQueryKey(SCOPE))?.accounts[0].alias
    ).toBe('work')
    expect(screen.queryByText('office')).toBeNull()
  })

  it('a slower refusal of an older rename does not undo a newer one', async () => {
    const answers: Array<(outcome: 'ok' | 'refused') => void> = []

    serveRename(
      () =>
        new Promise((resolve, reject) => {
          answers.push(outcome =>
            outcome === 'ok'
              ? resolve({ ...ACCOUNTS.accounts[0], alias: 'desk' })
              : reject(new JsonRpcGatewayError('busy', { code: 5034, data: { reason: 'ACCOUNTS_UNAVAILABLE' } }))
          )
        })
    )

    renderSection()
    await renameWork('office')
    await waitFor(() => expect(screen.getByText('office')).toBeTruthy())
    fireEvent.click(screen.getAllByRole('button', { name: 'Rename' })[0])
    fireEvent.change(screen.getByRole('textbox', { name: 'New name for office' }), { target: { value: 'desk' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(answers).toHaveLength(2))

    answers[1]('ok')
    await waitFor(() => expect(screen.getByText('desk')).toBeTruthy())
    answers[0]('refused')

    await waitFor(() => expect(screen.getByRole('alert')).toBeTruthy())
    expect(
      queryClient.getQueryData<ConnectorAccountsResult>(connectorsAccountsQueryKey(SCOPE))?.accounts[0].alias
    ).toBe('desk')
  })

  it('refuses a duplicate or malformed name without calling the gateway', async () => {
    const request = serveRename(() => Promise.resolve(ACCOUNTS.accounts[0]))

    renderSection()
    await renameWork('home')

    expect(screen.getByRole('alert').textContent).toBe('That name is already used.')

    fireEvent.change(screen.getByRole('textbox', { name: 'New name for work' }), { target: { value: '-work' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    expect(screen.getByRole('alert').textContent).toMatch(/lowercase letters/)
    expect(renameCalls(request)).toEqual([])
  })
})
