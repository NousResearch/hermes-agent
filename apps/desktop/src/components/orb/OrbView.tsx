import { useStore } from '@nanostores/react'
import { type ComponentProps, useEffect, useRef, useState } from 'react'

import { Loader, type LoaderType } from '@/components/ui/loader'
import { cn } from '@/lib/utils'
import { $orbEnabled, $orbParams } from '@/store/orb'

import { type OrbParams } from './orb-params'
import { createOrbRenderer, isOrbWebGpuSupported } from './orb-renderer'

interface OrbViewProps extends Omit<ComponentProps<'div'>, 'children'> {
  /** Render params — the built-in default or a parsed configurator URL. */
  params: OrbParams
  /** Loader shown while WebGPU is unavailable or the renderer fails. */
  fallbackType?: LoaderType
  label?: string
}

/**
 * Thinking-state orb preference: whether the orb replaces the standard
 * thinking indicators, and the params to render (custom URL or default).
 */
export function useOrbThinking(): { enabled: boolean; params: OrbParams } {
  const enabled = useStore($orbEnabled)
  const params = useStore($orbParams)

  return { enabled, params }
}

/**
 * WebGPU liquid-glass orb (vendored lersent001/orb renderer). Degrades
 * gracefully: without WebGPU — or when the renderer errors — it renders the
 * standard `Loader` instead of a broken canvas.
 */
export function OrbView({ params, fallbackType = 'original-thinking', label, className, ...rest }: OrbViewProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const paramsRef = useRef(params)
  paramsRef.current = params
  const [failed, setFailed] = useState(() => !isOrbWebGpuSupported())

  useEffect(() => {
    const canvas = canvasRef.current

    if (!canvas || failed) {
      return
    }

    return createOrbRenderer({
      canvas,
      getParams: () => paramsRef.current,
      onError: () => setFailed(true)
    })
  }, [failed])

  if (failed) {
    return <Loader aria-hidden="true" className={className} label={label} type={fallbackType} {...rest} />
  }

  return (
    <div aria-label={label} className={cn('relative overflow-hidden', className)} role="img" {...rest}>
      <canvas className="absolute inset-0 h-full w-full" ref={canvasRef} />
    </div>
  )
}
