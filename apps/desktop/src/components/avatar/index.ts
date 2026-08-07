// Avatar components for Hermes Desktop
// Mirrors FlowTec Commander's FVE avatar system (commit cd3412f)

export { AvatarStage, type AvatarStageProps, type AvatarState } from './AvatarStage'
export { AvatarCanvas, type AvatarCanvasProps } from './AvatarCanvas'
export { Avatar3DMascot } from './Avatar3DMascot'
export {
  AVATAR_GLB_URL,
  type AvatarClipName,
  type AvatarEmotionName,
  type AvatarRuntimeEvent,
  clipForEvent,
  emotionForEvent,
  isLoopingClip,
  applyEmotion,
  FACIAL_MORPHS,
} from './clipMap'