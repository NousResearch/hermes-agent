import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render } from '@testing-library/react'

import { type CanonicalGroupOrigin, useGatewayGroupRooms } from './canonical-group-registry'
import { CanonicalGroupRow } from './canonical-group-row'

function GatewayRooms({ onOpen }: { onOpen: (key: string, origin: CanonicalGroupOrigin) => void }) {
  const rooms = useGatewayGroupRooms()

  return (
    <>
      {rooms.map(room => (
        <CanonicalGroupRow active={false} key={room.key} onOpen={() => onOpen(room.key, 'key')} room={room} />
      ))}
    </>
  )
}

/** The Bots roster's gateway Group Chat rows, on a fresh query cache. */
export function renderGatewayRooms(onOpen: (key: string, origin: CanonicalGroupOrigin) => void = () => undefined) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={client}>
      <GatewayRooms onOpen={onOpen} />
    </QueryClientProvider>
  )
}
