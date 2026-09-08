import { type CSSProperties } from 'react'

import { HackeryButton } from '../components/hackery-button'
import { $route } from '../store'

/*
 * Welcome screen.
 *
 * NORTH FORGE wordmark (Collapse Bold, uppercase, tracked), a one-line
 * description of what setup does, and one bracket button that advances to
 * the location screen (which resolves the checkout + shows the sibling
 * -venv / -data paths before anything runs).
 */
export default function Welcome() {
  return (
    <div className="nf-fade-in flex h-full flex-col items-center justify-center gap-10 px-12 py-10">
      <div className="w-full max-w-2xl min-w-0 text-center">
        <p
          className="fit-text mx-auto mb-4 w-full font-['Collapse'] font-bold uppercase leading-[0.9] tracking-[0.08em] text-midground mix-blend-plus-lighter dark:text-foreground/90"
          style={
            {
              '--fit-text-line-height': '0.9',
              '--fit-text-max': '6rem',
              '--fit-text-min': '2.5rem'
            } as CSSProperties
          }
        >
          <span>
            <span>NORTH FORGE</span>
          </span>
          <span aria-hidden="true">NORTH FORGE</span>
        </p>

        <p className="m-0 text-center text-base leading-normal tracking-tight text-muted-foreground">
          This sets up the North Forge checkout that&rsquo;s already on your drive &mdash;
          a Python environment and a data folder next to it. One time, a few minutes.
        </p>
      </div>

      <HackeryButton label="Get started" onClick={() => $route.set('location')} />
    </div>
  )
}
