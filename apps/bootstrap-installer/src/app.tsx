import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import Failure from './routes/failure'
import Location from './routes/location'
import Progress from './routes/progress'
import Success from './routes/success'
import Welcome from './routes/welcome'
import { $bootstrap, $route, initialize } from './store'

/*
 * App shell — North Forge Setup.
 *
 * No header chrome (the OS title bar already says "North Forge Setup"; an
 * in-window repeat of the mark + words was redundant slop).
 *
 * Route state lives in a single $route atom — 5 screens, no react-router.
 */
export default function App() {
  const route = useStore($route)
  const bootstrap = useStore($bootstrap)

  useEffect(() => {
    void initialize()
  }, [])

  return (
    <div className="relative flex h-full flex-col overflow-hidden bg-background text-foreground">
      <main className="relative z-10 flex flex-1 flex-col overflow-hidden">
        {route === 'welcome' && <Welcome />}
        {route === 'location' && <Location />}
        {route === 'progress' && <Progress bootstrap={bootstrap} />}
        {route === 'success' && <Success />}
        {route === 'failure' && <Failure bootstrap={bootstrap} />}
      </main>
    </div>
  )
}
