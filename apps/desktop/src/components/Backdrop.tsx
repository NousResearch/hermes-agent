import { type CSSProperties, useEffect } from 'react'

const assetPath = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`

const BACKDROP = {
  opacity: 0.025,
  blendMode: 'difference' as CSSProperties['mixBlendMode'],
  invert: true,
  saturate: 1,
  brightness: 1,
  objectPosition: 'top left',
  scale: 160,
  radiusScalar: 0.2
}

export function Backdrop() {
  useEffect(() => {
    document.documentElement.style.setProperty('--radius-scalar', String(BACKDROP.radiusScalar))
  }, [])

  return (
    <div
      aria-hidden
      className="pointer-events-none absolute inset-0 z-2"
      style={{
        mixBlendMode: BACKDROP.blendMode,
        opacity: BACKDROP.opacity
      }}
    >
      <img
        alt=""
        className="w-auto min-w-dvw object-cover"
        fetchPriority="low"
        src={assetPath('ds-assets/filler-bg0.jpg')}
        style={{
          height: `${BACKDROP.scale}dvh`,
          objectPosition: BACKDROP.objectPosition,
          filter: `invert(calc(${BACKDROP.invert ? 1 : 0} * var(--backdrop-invert-mul, 1))) saturate(${BACKDROP.saturate}) brightness(${BACKDROP.brightness})`
        }}
      />
    </div>
  )
}
