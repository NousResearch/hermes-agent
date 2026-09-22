import type { EntityKey, LunarCitySnapshot } from '../model'
import { createWorkerSocialController } from '../worker-social'

import type { createEntityRegistry } from './entities'
import { createWorkerCompletionController } from './worker-completion'

export function createWorkerSocialRuntime(registry: ReturnType<typeof createEntityRegistry>, enabled: () => boolean) {
  const controller = createWorkerSocialController()
  const completion = createWorkerCompletionController()

  const environment = () => ({
    enabled: enabled(),
    supports: (key: EntityKey, animation: string) => registry.entity(key)?.visual?.supportsAnimation?.(animation) ?? false,
    position: (key: EntityKey) => registry.entity(key)?.position,
    available: (key: EntityKey) => {
      const record = registry.entity(key)

      return !!record?.visual && !record.moving && record.nearby && record.lodIndex === 0
    }
  })

  return {
    update: (snapshot: LunarCitySnapshot) => {controller.update(snapshot);completion.update(snapshot)},
    encounters: () => controller.encounters(),
    nextWakeDelay: () => controller.nextWakeDelay(environment()),
    request: (key: EntityKey, gesture?: string) => gesture
      ? controller.requestGesture(key, gesture, environment())
      : controller.requestGreeting(key, environment()),
    tick(elapsedMs: number) {
      const env=environment(), poses=new Map(controller.tick(elapsedMs,env))

      for(const [key,pose] of completion.tick(elapsedMs,{...env,moving:key=>registry.entity(key)?.moving??false})) {poses.set(key,pose)}
      registry.setSocialPoses(poses)

      return controller.active() || completion.isActive()
    }
  }
}
