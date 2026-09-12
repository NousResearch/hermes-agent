# Federated Apps API/Data Contract

## Overview

This document defines the API surface and data shapes for the Federated Apps Wiring/Operator Topology system (Slice 06: Normalized State Artifact). This system enables distributed applications to register themselves, expose operators (capabilities), and establish wiring connections between operators across app boundaries.

## Data Types

### ConnectionState

Represents the operational state of operators and wires:

```typescript
type ConnectionState =
  | 'available'      // Ready to accept work
  | 'busy'           // Currently processing
  | 'error'          // Error state (retryable)
  | 'unavailable'    // Unavailable (non-retryable)
  | 'connecting'     // Initial connection in progress
  | 'disconnecting'  // Graceful shutdown in progress
```

### WireProtocol

Communication protocols supported for operator wiring:

```typescript
type WireProtocol = 'ipc' | 'ws' | 'http' | 'grpc' | 'internal'
```

### WireDirection

Directionality of data flow through a wire:

```typescript
type WireDirection = 'unidirectional' | 'bidirectional'
```

### OperatorCapabilities

Feature flags describing operator capabilities:

```typescript
interface OperatorCapabilities {
  canStream: boolean        // Supports streaming responses
  canBatch: boolean         // Supports batch processing
  canDelegate: boolean      // Can delegate work to other operators
  supportsPriority: boolean // Supports priority queuing
  maxConcurrent: number     // Maximum concurrent operations
}
```

### FederatedApp

Represents a registered application in the federation:

```typescript
interface FederatedApp {
  id: string                    // Unique identifier
  name: string                  // Human-readable name
  version: string               // App version (semver)
  url?: string                  // Optional remote URL for external apps
  protocol: WireProtocol        // Communication protocol
  capabilities: {
    canHostOperators: boolean   // Can host operator nodes
    canProxy: boolean           // Can proxy requests
    maxOperators: number        // Maximum operators this app can host
  }
  state: ConnectionState        // Current operational state
  registeredAt: number          // Unix timestamp (ms)
  lastHeartbeatAt: number       // Last heartbeat timestamp (ms)
  metadata: Record<string, unknown>  // Extensible metadata
}
```

### OperatorNode

Represents a capability/provider within an app:

```typescript
interface OperatorNode {
  id: string                    // Unique identifier
  appId: string                 // Reference to parent FederatedApp
  name: string                  // Human-readable name
  type: string                  // Operator type (e.g., 'llm', 'tool', 'gateway')
  capabilities: OperatorCapabilities
  state: ConnectionState        // Current operational state
  currentLoad: number           // 0-100 percentage
  queueDepth: number            // Current queued items
  errorMessage?: string         // Present when state is 'error'
  lastActivityAt: number        // Last activity timestamp (ms)
  metadata: Record<string, unknown>  // Extensible metadata
}
```

### OperatorWire

Represents a connection (edge) between two operators:

```typescript
interface OperatorWire {
  id: string                    // Unique identifier
  sourceAppId: string           // Source app reference
  sourceOperatorId: string      // Source operator reference
  targetAppId: string           // Target app reference
  targetOperatorId: string      // Target operator reference
  protocol: WireProtocol        // Wire protocol
  direction: WireDirection      // Data flow direction
  state: ConnectionState        // Connection health
  latencyMs?: number            // Last measured latency
  throughputBps?: number        // Last measured throughput (bytes/sec)
  establishedAt: number         // Connection establishment timestamp
  lastActivityAt: number        // Last activity timestamp
  metadata: Record<string, unknown>  // Extensible metadata
}
```

### FederatedAppsState

Normalized state artifact for the entire topology:

```typescript
interface FederatedAppsState {
  apps: Record<string, FederatedApp>      // Keyed by app id
  operators: Record<string, OperatorNode>  // Keyed by operator id
  wires: Record<string, OperatorWire>     // Keyed by wire id
  version: number                          // State version for optimistic updates
}
```

### WireEndpoint

Reference to an endpoint for creating wires:

```typescript
interface WireEndpoint {
  appId: string
  operatorId: string
}
```

## API Operations

### App Management

#### Register App
```typescript
registerApp(app: Omit<FederatedApp, 'registeredAt' | 'lastHeartbeatAt'>): FederatedApp
```
Registers a new federated app. Automatically sets `registeredAt` and `lastHeartbeatAt` to current time.

#### Unregister App
```typescript
unregisterApp(appId: string): boolean
```
Unregisters an app and all its operators and wires. Returns `true` if app existed and was removed.

#### Update App State
```typescript
updateAppState(appId: string, state: ConnectionState, metadata?: Record<string, unknown>): boolean
```
Updates app state and heartbeat timestamp. Returns `true` if app existed.

### Operator Management

#### Register Operator
```typescript
registerOperator(operator: Omit<OperatorNode, 'lastActivityAt'>): OperatorNode
```
Registers a new operator node. Automatically sets `lastActivityAt` to current time.

#### Unregister Operator
```typescript
unregisterOperator(operatorId: string): boolean
```
Unregisters an operator and all its connected wires. Returns `true` if operator existed.

#### Update Operator State
```typescript
updateOperatorState(
  operatorId: string,
  updates: Partial<Pick<OperatorNode, 'state' | 'currentLoad' | 'queueDepth' | 'errorMessage'>>
): boolean
```
Updates operator state and metrics. Automatically updates `lastActivityAt`. Returns `true` if operator existed.

### Wire Management

#### Create Wire
```typescript
createWire(
  id: string,
  source: WireEndpoint,
  target: WireEndpoint,
  protocol: WireProtocol,
  direction: WireDirection = 'bidirectional'
): OperatorWire
```
Creates a wire between two operators. Initial state is `'connecting'`.

#### Remove Wire
```typescript
removeWire(wireId: string): boolean
```
Removes a wire. Returns `true` if wire existed.

#### Update Wire State
```typescript
updateWireState(
  wireId: string,
  updates: Partial<Pick<OperatorWire, 'state' | 'latencyMs' | 'throughputBps'>>
): boolean
```
Updates wire state and metrics. Automatically updates `lastActivityAt`. Returns `true` if wire existed.

### Query Operations

#### Get App Operators
```typescript
getAppOperators(appId: string): OperatorNode[]
```
Returns all operators belonging to a specific app.

#### Get Operator Wires
```typescript
getOperatorWires(operatorId: string): OperatorWire[]
```
Returns all wires connected to a specific operator (as source or target).

#### Get Topology Graph
```typescript
getTopologyGraph(): {
  nodes: Array<{ id: string; type: 'app' | 'operator'; data: FederatedApp | OperatorNode }>
  edges: Array<{ id: string; source: string; target: string; data: OperatorWire }>
}
```
Returns the complete topology graph for visualization.

#### Get Available Operators
```typescript
getAvailableOperators(): OperatorNode[]
```
Returns operators that are `'available'` and have load < 90%.

#### Has Path
```typescript
hasPath(sourceId: string, targetId: string): boolean
```
Returns `true` if a routing path exists between two operators (BFS traversal).

#### Get Routing Path
```typescript
getRoutingPath(sourceId: string, targetId: string): string[] | null
```
Returns the operator ID path from source to target, or `null` if no path exists.

### State Persistence

#### Export State
```typescript
exportState(): FederatedAppsState
```
Exports the complete state as a JSON-serializable object.

#### Import State
```typescript
importState(state: FederatedAppsState): void
```
Imports state from a JSON artifact. Automatically increments version.

#### Clear State
```typescript
clearFederatedApps(): void
```
Resets all federated apps state (for testing/reset).

## State Management

### Normalized State

All entities are stored in flat maps keyed by ID:
- `apps: Record<string, FederatedApp>`
- `operators: Record<string, OperatorNode>`
- `wires: Record<string, OperatorWire>`

This enables O(1) lookups and prevents data duplication.

### Immutable Updates

All mutations create new state objects:
```typescript
$ federatedApps.set({
  ...state,
  apps: { ...state.apps, [app.id]: newApp },
  version: state.version + 1
})
```

### Version Tracking

The `version` field increments on every mutation for optimistic concurrency control.

### Referential Integrity

- Unregistering an app automatically removes its operators and wires
- Unregistering an operator automatically removes its connected wires

## Server Normalization Rules

1. **App ID Uniqueness**: App IDs must be globally unique within the federation
2. **Operator ID Uniqueness**: Operator IDs must be globally unique (not just within app)
3. **Wire Endpoint Validation**: Both source and target operators must exist when creating a wire
4. **State Transitions**: Valid state transitions are enforced (e.g., 'connecting' → 'available', not directly to 'busy')
5. **Heartbeat Timeouts**: Apps should transition to 'unavailable' if heartbeat is not received within timeout window
6. **Load Balancing**: Operators with `currentLoad >= 90` are excluded from `getAvailableOperators()`

## Browser/Client Normalization Rules

1. **Optimistic Updates**: Clients may apply optimistic updates while awaiting server confirmation
2. **Conflict Resolution**: Version mismatches trigger re-fetch of current state
3. **Local-First**: Client state is the source of truth for UI until server confirms otherwise
4. **Retry Logic**: Failed operations are retried with exponential backoff

## TypeScript Data Shapes

```typescript
// Re-export types for consumers
export type {
  ConnectionState,
  WireProtocol,
  WireDirection,
  OperatorCapabilities,
  FederatedApp,
  OperatorNode,
  OperatorWire,
  FederatedAppsState,
  WireEndpoint
}

// Main state atom (nanostores)
export const $federatedApps: Atom<FederatedAppsState>

// State version accessor
export function getStateVersion(): number
```

## Integration Notes

The federated apps store integrates with the Hermes desktop app via nanostores, following the same pattern as `subagents.ts`. Components can subscribe to `$federatedApps` atom for reactive updates.

```typescript
import { $federatedApps, getAvailableOperators } from './federated-apps'

// Subscribe to state changes
$ federatedApps.subscribe(state => {
  console.log('Apps:', Object.keys(state.apps))
  console.log('Operators:', Object.keys(state.operators))
})

// Use query functions
const available = getAvailableOperators()
```
