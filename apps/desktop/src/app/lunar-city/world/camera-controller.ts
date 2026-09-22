import type { CameraControlState, CameraIntent, CameraLandmark, EntityKey, Vec3, WorldBounds } from '../model'

export interface MutableVector3 extends Vec3 {
  set?(x: number, y: number, z: number): void
}

export interface CameraLike {
  alpha: number
  beta: number
  inertia?: number
  radius: number
  target: MutableVector3
}

export interface CameraPose {
  alpha: number
  beta: number
  radius: number
  target: Vec3
}

export interface CameraControllerOptions {
  focusAnchors?: ReadonlyMap<EntityKey, () => Vec3 | undefined>
  followOffset?: Vec3
  focusRadius?: (key: EntityKey) => number | undefined
  focusBeta?: (key: EntityKey) => number | undefined
  transitionMs?: number
}

export interface CameraController {
  dispatch(intent: CameraIntent): void
  getState(): CameraControlState
  isTransitioning(): boolean
  setReducedMotion(reduced: boolean): void
  setIdleEnabled(enabled: boolean): void
  isIdleActive(): boolean
  isIdlePending(): boolean
  update(elapsedMs: number): void
}

export interface CameraPickTarget {
  kind: 'entity'
  entityKey: EntityKey
}

export interface CameraInputBindings {
  dispatch(intent: CameraIntent): void
  pick(canvasX: number, canvasY: number): CameraPickTarget | undefined
}

export interface CameraInputRelease {
  (): void
  activeListenerCount(): number
}

interface FocusTransition {
  elapsedMs: number
  from: CameraPose
  to: CameraPose
}

const FULL_TURN = Math.PI * 2
const DEFAULT_TRANSITION_MS = 260
const DRAG_THRESHOLD_PX = 3
const IDLE_START_MS = 5_000
const IDLE_ORBIT_RADIANS_PER_MS = 0.000018

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value))
}

function normalizeAngle(value: number): number {
  const normalized = ((((value + Math.PI) % FULL_TURN) + FULL_TURN) % FULL_TURN) - Math.PI

  return normalized === -Math.PI && value > 0 ? Math.PI : normalized
}

function clampTarget(target: Vec3, bounds: WorldBounds): Vec3 {
  return {
    x: clamp(target.x, bounds.min.x, bounds.max.x),
    y: clamp(target.y, bounds.min.y, bounds.max.y),
    z: clamp(target.z, bounds.min.z, bounds.max.z)
  }
}

function setTarget(target: MutableVector3, value: Vec3): void {
  if (typeof target.set === 'function') {
    target.set(value.x, value.y, value.z)

    return
  }

  target.x = value.x
  target.y = value.y
  target.z = value.z
}

function easeInOut(progress: number): number {
  return progress < 0.5 ? 2 * progress * progress : 1 - (-2 * progress + 2) ** 2 / 2
}

function interpolateVector(from: Vec3, to: Vec3, progress: number): Vec3 {
  return {
    x: from.x + (to.x - from.x) * progress,
    y: from.y + (to.y - from.y) * progress,
    z: from.z + (to.z - from.z) * progress
  }
}

function interpolatePose(from: CameraPose, to: CameraPose, progress: number): CameraPose {
  return {
    alpha: from.alpha + (to.alpha - from.alpha) * progress,
    beta: from.beta + (to.beta - from.beta) * progress,
    radius: from.radius + (to.radius - from.radius) * progress,
    target: interpolateVector(from.target, to.target, progress)
  }
}

function sameVector(left: Vec3 | undefined, right: Vec3): boolean {
  return left?.x === right.x && left.y === right.y && left.z === right.z
}

function writePose(camera: CameraLike, pose: CameraPose): void {
  camera.alpha = pose.alpha
  camera.beta = pose.beta
  camera.radius = pose.radius
  setTarget(camera.target, pose.target)
}

export function readPose(camera: CameraLike): CameraPose {
  return {
    alpha: camera.alpha,
    beta: camera.beta,
    radius: camera.radius,
    target: { x: camera.target.x, y: camera.target.y, z: camera.target.z }
  }
}

export function poseFromLandmark(landmark: CameraLandmark): CameraPose {
  return {
    alpha: landmark.alpha,
    beta: landmark.beta,
    radius: landmark.radius,
    target: { ...landmark.target }
  }
}

export function createCameraController(
  camera: CameraLike,
  overview: CameraLandmark,
  bounds: WorldBounds,
  options: CameraControllerOptions = {}
): CameraController {
  camera.inertia = 0
  const focusAnchors = options.focusAnchors ?? new Map<EntityKey, () => Vec3 | undefined>()
  const followOffset = options.followOffset
  const transitionMs = options.transitionMs ?? DEFAULT_TRANSITION_MS
  let focusedEntityKey: EntityKey | undefined
  let following = false
  let transition: FocusTransition | undefined
  let lastFollowAnchor: Vec3 | undefined
  let reducedMotion = false
  let idleEnabled = false
  let idleElapsedMs = 0
  let idleActive = false

  const resetIdle = (): void => {
    idleElapsedMs = 0
    idleActive = false
  }

  const applyBounds = (): void => {
    camera.alpha = normalizeAngle(camera.alpha)
    camera.beta = clamp(camera.beta, overview.minBeta, overview.maxBeta)
    camera.radius = clamp(camera.radius, overview.minRadius, overview.maxRadius)
    setTarget(camera.target, clampTarget(camera.target, bounds))
  }

  const startFocusTransition = (anchor: Vec3, follow: boolean): void => {
    const from = readPose(camera)
    const offsetLength = followOffset ? Math.hypot(followOffset.x, followOffset.y, followOffset.z) : 0

    const to =
      follow && followOffset && offsetLength > 0
        ? {
            alpha: Math.atan2(followOffset.z, followOffset.x),
            beta: Math.acos(clamp(followOffset.y / offsetLength, -1, 1)),
            radius: clamp(offsetLength, overview.minRadius, overview.maxRadius),
            target: clampTarget(anchor, bounds)
          }
        : {
            ...from,
            beta: focusedEntityKey ? (options.focusBeta?.(focusedEntityKey) ?? from.beta) : from.beta,
            radius: focusedEntityKey ? (options.focusRadius?.(focusedEntityKey) ?? from.radius) : from.radius,
            target: clampTarget(anchor, bounds)
          }

    if (reducedMotion) {
      writePose(camera, to)
      applyBounds()
      transition = undefined

      return
    }

    transition = { elapsedMs: 0, from, to }
  }

  const clearFocus = (): void => {
    focusedEntityKey = undefined
    following = false
    transition = undefined
    lastFollowAnchor = undefined
  }

  return {
    dispatch(intent) {
      resetIdle()

      if (intent.kind === 'orbit') {
        transition = undefined
        camera.alpha += intent.deltaAlpha
        camera.beta += intent.deltaBeta
        applyBounds()

        return
      }

      if (intent.kind === 'pan') {
        transition = undefined
        setTarget(camera.target, {
          x: camera.target.x + intent.deltaX,
          y: camera.target.y,
          z: camera.target.z + intent.deltaZ
        })
        applyBounds()

        return
      }

      if (intent.kind === 'zoom') {
        transition = undefined
        camera.radius += intent.delta
        applyBounds()

        return
      }

      if (intent.kind === 'focus') {
        const anchor = focusAnchors.get(intent.entityKey)?.()

        if (!anchor) {
          clearFocus()

          return
        }

        focusedEntityKey = intent.entityKey
        following = intent.follow
        startFocusTransition(anchor, following)
        lastFollowAnchor = { ...anchor }

        return
      }

      if (intent.kind === 'clear-focus') {
        clearFocus()

        return
      }

      clearFocus()
      writePose(camera, poseFromLandmark(overview))
    },
    getState() {
      return { focusedEntityKey, following }
    },
    isTransitioning() {
      return transition !== undefined
    },
    setReducedMotion(reduced) {
      reducedMotion = reduced

      if (reduced) {
        resetIdle()
      }

      if (reduced && transition) {
        writePose(camera, transition.to)
        applyBounds()
        transition = undefined
      }
    },
    setIdleEnabled(enabled) {
      idleEnabled = enabled

      if (!enabled) {
        resetIdle()
      }
    },
    isIdleActive() {
      return idleActive
    },
    isIdlePending() {
      return idleEnabled && !reducedMotion && !focusedEntityKey && !idleActive
    },
    update(elapsedMs) {
      if (following && focusedEntityKey) {
        const anchor = focusAnchors.get(focusedEntityKey)?.()

        if (!anchor) {
          clearFocus()

          return
        }

        if (!sameVector(lastFollowAnchor, anchor)) {
          startFocusTransition(anchor, true)
          lastFollowAnchor = { ...anchor }
        }
      }

      if (!transition) {
        const delta = Math.max(0, elapsedMs)

        if (idleEnabled && !reducedMotion && !focusedEntityKey) {
          idleElapsedMs += delta

          if (idleElapsedMs >= IDLE_START_MS) {
            idleActive = true
            camera.alpha = normalizeAngle(camera.alpha + delta * IDLE_ORBIT_RADIANS_PER_MS)
            applyBounds()
          }
        }

        return
      }

      transition.elapsedMs += Math.max(0, elapsedMs)
      const progress = clamp(transition.elapsedMs / transitionMs, 0, 1)
      writePose(camera, interpolatePose(transition.from, transition.to, easeInOut(progress)))
      applyBounds()

      if (progress === 1) {
        transition = undefined
      }
    }
  }
}

interface PointerSample {
  clientX: number
  clientY: number
  gesture: boolean
  moved: boolean
  pointerType: string
}

function distance(first: PointerSample, second: PointerSample): number {
  return Math.hypot(first.clientX - second.clientX, first.clientY - second.clientY)
}

function touchPointers(activePointers: ReadonlyMap<number, PointerSample>): PointerSample[] {
  return [...activePointers.values()].filter(pointer => pointer.pointerType === 'touch')
}

function canvasLocalCoordinates(canvas: HTMLCanvasElement, clientX: number, clientY: number): { x: number; y: number } {
  const rect = canvas.getBoundingClientRect()

  // Babylon's InputManager derives scene.pointerX/Y from client coordinates by
  // subtracting this same input-element rectangle; Scene.pick consumes those
  // CSS-pixel local coordinates rather than the backing-store dimensions.
  return { x: clientX - rect.left, y: clientY - rect.top }
}

function dispatchPick(
  canvas: HTMLCanvasElement,
  bindings: CameraInputBindings,
  clientX: number,
  clientY: number
): void {
  const coordinates = canvasLocalCoordinates(canvas, clientX, clientY)
  const target = bindings.pick(coordinates.x, coordinates.y)

  if (target) {
    bindings.dispatch({ kind: 'focus', entityKey: target.entityKey, follow: true })

    return
  }

  bindings.dispatch({ kind: 'clear-focus' })
}

export function bindCameraInput(canvas: HTMLCanvasElement, bindings: CameraInputBindings): CameraInputRelease {
  const activePointers = new Map<number, PointerSample>()
  let pinchDistance: number | undefined

  const pointerDown = (event: PointerEvent): void => {
    activePointers.set(event.pointerId, {
      clientX: event.clientX,
      clientY: event.clientY,
      gesture: false,
      moved: false,
      pointerType: event.pointerType
    })

    const touches = touchPointers(activePointers)

    if (touches.length >= 2) {
      for (const touch of touches) {
        touch.gesture = true
      }

      pinchDistance = distance(touches[0]!, touches[1]!)
    }
  }

  const pointerMove = (event: PointerEvent): void => {
    const pointer = activePointers.get(event.pointerId)

    if (!pointer) {
      return
    }

    const deltaX = event.clientX - pointer.clientX
    const deltaY = event.clientY - pointer.clientY
    pointer.moved ||= Math.hypot(deltaX, deltaY) >= DRAG_THRESHOLD_PX
    pointer.clientX = event.clientX
    pointer.clientY = event.clientY

    const touches = touchPointers(activePointers)

    if (touches.length >= 2) {
      for (const touch of touches) {
        touch.gesture = true
      }

      const nextDistance = distance(touches[0]!, touches[1]!)

      if (pinchDistance !== undefined) {
        bindings.dispatch({ kind: 'zoom', delta: (pinchDistance - nextDistance) * 0.08 })
      }

      pinchDistance = nextDistance

      return
    }

    if (pointer.gesture && pointer.pointerType === 'touch') {
      return
    }

    if (event.button === 2 || event.buttons === 2) {
      bindings.dispatch({ kind: 'pan', deltaX: -deltaX * 0.25, deltaZ: deltaY * 0.25 })

      return
    }

    if (event.button === 0 || event.buttons === 1 || pointer.pointerType === 'touch') {
      bindings.dispatch({ kind: 'orbit', deltaAlpha: -deltaX * 0.01, deltaBeta: -deltaY * 0.01 })
    }
  }

  const pointerUp = (event: PointerEvent): void => {
    const pointer = activePointers.get(event.pointerId)
    activePointers.delete(event.pointerId)
    const touches = touchPointers(activePointers)

    pinchDistance = touches.length >= 2 ? distance(touches[0]!, touches[1]!) : undefined

    if (pointer && !pointer.gesture && !pointer.moved && event.button !== 2) {
      dispatchPick(canvas, bindings, event.clientX, event.clientY)
    }
  }

  const pointerCancel = (event: PointerEvent): void => {
    activePointers.delete(event.pointerId)
    const touches = touchPointers(activePointers)

    pinchDistance = touches.length >= 2 ? distance(touches[0]!, touches[1]!) : undefined
  }

  const wheel = (event: WheelEvent): void => {
    event.preventDefault()
    bindings.dispatch({ kind: 'zoom', delta: event.deltaY * 0.05 })
  }

  const contextMenu = (event: Event): void => event.preventDefault()

  canvas.addEventListener('pointerdown', pointerDown)
  canvas.addEventListener('pointermove', pointerMove)
  canvas.addEventListener('pointerup', pointerUp)
  canvas.addEventListener('pointercancel', pointerCancel)
  canvas.addEventListener('wheel', wheel, { passive: false })
  canvas.addEventListener('contextmenu', contextMenu)

  let activeListenerCount = 6

  const release = (): void => {
    if (activeListenerCount === 0) {
      return
    }

    canvas.removeEventListener('pointerdown', pointerDown)
    canvas.removeEventListener('pointermove', pointerMove)
    canvas.removeEventListener('pointerup', pointerUp)
    canvas.removeEventListener('pointercancel', pointerCancel)
    canvas.removeEventListener('wheel', wheel)
    canvas.removeEventListener('contextmenu', contextMenu)
    activeListenerCount = 0
  }

  release.activeListenerCount = () => activeListenerCount

  return release
}
