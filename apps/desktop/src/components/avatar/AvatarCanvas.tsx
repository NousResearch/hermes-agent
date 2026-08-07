// AvatarCanvas — dedicated Canvas for the mascot in Hermes Desktop.
// The main pipeline is framed for abstract visualization; a character is not.
// Deliberately does NOT call animController.setInvalidate(): that is a single
// slot owned by the pipeline and a second writer would steal it.

import { Canvas } from '@react-three/fiber'
import { AvatarStage } from './AvatarStage'

export interface AvatarCanvasProps {
  /** Simple state to drive the avatar */
  state?: AvatarStageProps['state']
  /** Canvas size in pixels */
  width?: number
  height?: number
  /** Background color (CSS) */
  background?: string
}

import type { AvatarStageProps } from './AvatarStage'

export function AvatarCanvas({
  state = 'idle',
  width = 200,
  height = 300,
  background = 'transparent',
}: AvatarCanvasProps) {
  return (
    <Canvas
      camera={{ position: [0, 0, 3], fov: 35 }}
      style={{ width, height, background }}
      frameloop="demand"
      gl={{ antialias: true, alpha: true, preserveDrawingBuffer: false }}
    >
      <ambientLight intensity={0.8} />
      <directionalLight position={[2, 4, 3]} intensity={1.2} />
      <AvatarStage state={state} />
    </Canvas>
  )
}