/**
 * Which way the flow runs, for the parts that can't be told by a prop.
 *
 * Node and edge components are rendered by React Flow from a type map, so the
 * page can't pass them anything — but a handle has to face the way the ranks
 * run or every wire leaves the wrong side of its card. Dagre already takes the
 * direction as `rankdir`; this is the same value reaching the components that
 * have to agree with it.
 */

import { Position } from '@xyflow/react'
import { createContext, useContext } from 'react'

import { DEFAULT_DIR, type FlowDir } from './layout'

const Ctx = createContext<FlowDir>(DEFAULT_DIR)

export const FlowDirProvider = Ctx.Provider

export const useFlowDir = () => useContext(Ctx)

/** Where a step's input and output sit for the current direction. */
export function usePorts(): { source: Position; target: Position } {
  return useFlowDir() === 'TB'
    ? { source: Position.Bottom, target: Position.Top }
    : { source: Position.Right, target: Position.Left }
}
