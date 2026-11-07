/** A registered llama.cpp engine — the shape of `GET /api/local-models/engines`.
 *
 *  Split out of `types/hermes.ts` because that file sits at the 2000-line health
 *  target: new engine types live here, next to the API that returns them.
 *  `LocalModelsStatus.active_engine_id` (types/hermes.ts) names one of these `id`s. */
export interface LocalEngine {
  id: string
  name: string
  /** `managed` is Hermes' own PM-managed llama.cpp build (install/update flows);
   *  `external` is a binary the user registered themselves. */
  kind: 'managed' | 'external'
  path: string
  backend: string
  args: string[]
  env: Record<string, string>
  builtin: boolean
  active: boolean
  /** Whether the path currently points at a runnable file. */
  installed: boolean
}
