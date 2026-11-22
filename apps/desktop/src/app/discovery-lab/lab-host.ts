import { useMemo } from 'react'
import { useNavigate } from 'react-router'

import type { LabHost } from './lab-states'
import { pressWhenShown, reloadAt } from './ui-driver'

/** TESTING ONLY: the lab's way into the real UI from a page under the router. */
export function useLabHost(): LabHost {
  const navigate = useNavigate()

  return useMemo(() => ({ navigate: to => navigate(to), load: reloadAt, press: pressWhenShown }), [navigate])
}
