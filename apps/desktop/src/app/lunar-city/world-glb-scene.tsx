import { useEffect, useRef, useState } from 'react'
import { AmbientLight, Box3, Color, DirectionalLight, PerspectiveCamera, Raycaster, Scene, Vector2, Vector3, WebGLRenderer } from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

import { LUNAR_CITY_ASSET_MANIFEST } from './world-assets'

export interface WorldGlbSelection {
  objectName: string
}

interface WorldGlbSceneProps {
  className?: string
  enabled: boolean
  onSelect?: (selection: WorldGlbSelection) => void
}

function frameCamera(camera: PerspectiveCamera, box: Box3): void {
  const center = box.getCenter(new Vector3())
  const size = box.getSize(new Vector3())
  const radius = Math.max(size.x, size.y, size.z, 1)

  camera.position.set(center.x + radius * 0.8, center.y - radius * 1.0, center.z + radius * 0.72)
  camera.lookAt(center)
  camera.near = 0.1
  camera.far = radius * 8
  camera.updateProjectionMatrix()
}

export function WorldGlbScene({ className, enabled, onSelect }: WorldGlbSceneProps) {
  const hostRef = useRef<HTMLDivElement | null>(null)
  const [status, setStatus] = useState<'failed' | 'loading' | 'ready'>('loading')

  useEffect(() => {
    const host = hostRef.current

    if (!enabled || !host) {
      return
    }

    let disposed = false
    let frame = 0
    const scene = new Scene()
    scene.background = new Color(0x030712)
    const camera = new PerspectiveCamera(45, 1, 0.1, 500)
    let renderer: WebGLRenderer

    if (typeof window.WebGLRenderingContext === 'undefined') {
      setStatus('failed')

      return
    }

    try {
      renderer = new WebGLRenderer({ alpha: false, antialias: true, powerPreference: 'high-performance' })
    } catch {
      setStatus('failed')

      return
    }

    const raycaster = new Raycaster()
    const pointer = new Vector2()
    const loader = new GLTFLoader()

    const resize = () => {
      const rect = host.getBoundingClientRect()
      const width = Math.max(1, Math.floor(rect.width))
      const height = Math.max(1, Math.floor(rect.height))

      renderer.setSize(width, height, false)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
    }

    const animate = () => {
      if (disposed || document.hidden) {
        return
      }

      scene.rotation.z += 0.0009
      renderer.render(scene, camera)
      frame = window.requestAnimationFrame(animate)
    }

    const onPointerDown = (event: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect()

      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
      raycaster.setFromCamera(pointer, camera)

      const hit = raycaster.intersectObjects(scene.children, true)[0]

      if (hit?.object.name) {
        onSelect?.({ objectName: hit.object.name })
      }
    }

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.domElement.setAttribute('aria-label', 'Interactive Lunar City 3D scene')
    renderer.domElement.setAttribute('data-testid', 'lunar-city-glb-canvas')
    host.append(renderer.domElement)
    scene.add(new AmbientLight(0xb8d7ff, 0.95))

    const key = new DirectionalLight(0xffffff, 2.2)
    key.position.set(12, -18, 24)
    scene.add(key)

    resize()
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(resize)
    observer?.observe(host)
    renderer.domElement.addEventListener('pointerdown', onPointerDown)

    loader.load(
      `${import.meta.env.BASE_URL}${LUNAR_CITY_ASSET_MANIFEST.glb}`,
      gltf => {
        if (disposed) {
          return
        }

        scene.add(gltf.scene)
        frameCamera(camera, new Box3().setFromObject(gltf.scene))
        setStatus('ready')
        animate()
      },
      undefined,
      () => {
        if (!disposed) {
          setStatus('failed')
        }
      }
    )

    return () => {
      disposed = true
      window.cancelAnimationFrame(frame)
      observer?.disconnect()
      renderer.domElement.removeEventListener('pointerdown', onPointerDown)
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [enabled, onSelect])

  return (
    <div className={className} data-renderer-status={status} data-testid="lunar-city-glb-host" ref={hostRef}>
      {status !== 'ready' && (
        <img
          alt="Lunar City Blender baseline with grounded roads and concave terrain"
          className="block aspect-[16/9] w-full object-cover"
          src={`${import.meta.env.BASE_URL}lunar-city/lunar-city-baseline.png`}
        />
      )}
    </div>
  )
}
