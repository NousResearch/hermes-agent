import { Animation, AnimationGroup, Vector3 } from './babylon.mjs'

const FRAME_RATE = 30

export function vectorClip(scene, name, target, property, middle, { duration = 30, start = null } = {}) {
  const initial = start ?? target[property].clone()
  const animation = new Animation(
    `${name}:${target.name}`,
    property,
    FRAME_RATE,
    Animation.ANIMATIONTYPE_VECTOR3,
    Animation.ANIMATIONLOOPMODE_CYCLE
  )
  animation.setKeys([
    { frame: 0, value: initial.clone() },
    { frame: duration / 2, value: new Vector3(...middle) },
    { frame: duration, value: initial.clone() }
  ])
  const animationGroup = new AnimationGroup(name, scene)
  animationGroup.addTargetedAnimation(animation, target)
  animationGroup.normalize(0, duration)
  return animationGroup
}

export function scalarClip(scene, name, target, property, middle, { duration = 30, start = 0 } = {}) {
  const animation = new Animation(
    `${name}:${target.name}`,
    property,
    FRAME_RATE,
    Animation.ANIMATIONTYPE_FLOAT,
    Animation.ANIMATIONLOOPMODE_CYCLE
  )
  animation.setKeys([
    { frame: 0, value: start },
    { frame: duration / 2, value: middle },
    { frame: duration, value: start }
  ])
  const animationGroup = new AnimationGroup(name, scene)
  animationGroup.addTargetedAnimation(animation, target)
  animationGroup.normalize(0, duration)
  return animationGroup
}

export function poseClip(scene, name, channels, { duration = 30 } = {}) {
  const animationGroup = new AnimationGroup(name, scene)
  channels.forEach(({ middle, property = 'rotation', start = null, target }, index) => {
    const initial = start ? new Vector3(...start) : target[property].clone()
    const animation = new Animation(
      `${name}:${index}:${target.name}`,
      property,
      FRAME_RATE,
      Animation.ANIMATIONTYPE_VECTOR3,
      Animation.ANIMATIONLOOPMODE_CYCLE
    )
    animation.setKeys([
      { frame: 0, value: initial.clone() },
      { frame: duration / 2, value: new Vector3(...middle) },
      { frame: duration, value: initial.clone() }
    ])
    animationGroup.addTargetedAnimation(animation, target)
  })
  animationGroup.normalize(0, duration)
  return animationGroup
}

export function stateClips(scene, target, names) {
  return names.map((name, index) => {
    const amplitude = 0.025 + (index % 5) * 0.0125
    return scalarClip(scene, name, target, 'rotation.y', amplitude, { duration: 24 + (index % 4) * 6 })
  })
}
