class MemoryStorage {
  #values = new Map()

  get length() {
    return this.#values.size
  }

  clear() {
    this.#values.clear()
  }

  getItem(key) {
    return this.#values.get(String(key)) ?? null
  }

  key(index) {
    return [...this.#values.keys()][index] ?? null
  }

  removeItem(key) {
    this.#values.delete(String(key))
  }

  setItem(key, value) {
    this.#values.set(String(key), String(value))
  }
}

Object.defineProperty(globalThis, 'localStorage', {
  configurable: true,
  value: new MemoryStorage(),
  writable: true
})

const core = await import('@babylonjs/core')
const serializers = await import('@babylonjs/serializers')

core.Logger.LogLevels = core.Logger.NoneLogLevel

export const {
  Animation,
  AnimationGroup,
  Bone,
  Color3,
  Matrix,
  Mesh,
  MeshBuilder,
  NullEngine,
  PBRMaterial,
  Quaternion,
  Scene,
  Skeleton,
  TransformNode,
  Vector3,
  VertexBuffer
} = core

export const { GLTF2Export } = serializers
