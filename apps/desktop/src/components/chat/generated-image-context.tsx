'use client'

import { createContext, type ReactNode, useContext, useMemo, useState } from 'react'

type Value = {
  isPending: boolean
  setPending: (pending: boolean) => void
}

const Ctx = createContext<Value | null>(null)

export function GeneratedImageProvider({ children }: { children: ReactNode }) {
  const [isPending, setPending] = useState(false)
  const value = useMemo(() => ({ isPending, setPending }), [isPending])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export const useGeneratedImageContext = () => useContext(Ctx)
