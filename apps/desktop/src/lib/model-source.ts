/** One upstream behind a provider entry that is really several providers.
 *
 *  A user-defined proxy (a local gateway, a resold relay) is wired into Hermes
 *  as ONE provider row whose catalog is flat `upstream/model` ids — and every
 *  menu strips that prefix for display (`modelBaseId`), so the same model name
 *  served by two upstreams renders as two identical rows. The prefix IS the
 *  source; grouping by it makes the distinction visible again.
 *
 *  Built-in catalogs (OpenRouter, Nous Portal, AI Gateway) also carry
 *  `author/model` ids, but there the prefix is the model's author — already
 *  implied by the name — so relabeling those rows would only add noise. The
 *  split therefore applies to USER-DEFINED provider rows only: an endpoint the
 *  user wired up themselves, where the prefix names a real upstream behind it.
 *  Callers gate on `is_user_defined`; the helpers here stay pure over ids. */

/** One source's models, in the provider's catalog order. */
export interface ModelSourceGroup {
  models: string[]
  source: string
}

/** The `upstream/` prefix of a model id, or null when it carries none. */
export function modelSource(model: string): null | string {
  const slash = model.indexOf('/')

  return slash > 0 ? model.slice(0, slash) : null
}

/** Group a provider's model ids by their `upstream/` prefix — sources in
 *  first-appearance order, each keeping the catalog's order inside — or null
 *  when the split does not apply: a single model, any id without a prefix (a
 *  mixed catalog would invent a bogus unlabeled group), or one distinct prefix
 *  (a sub-label that only repeats the provider adds nothing). */
export function groupModelSources(models: readonly string[]): ModelSourceGroup[] | null {
  if (models.length < 2) {
    return null
  }

  const groups: ModelSourceGroup[] = []
  const bySource = new Map<string, ModelSourceGroup>()

  for (const model of models) {
    const source = modelSource(model)

    if (source === null) {
      return null
    }

    let group = bySource.get(source)

    if (!group) {
      group = { models: [], source }
      bySource.set(source, group)
      groups.push(group)
    }

    group.models.push(model)
  }

  return groups.length >= 2 ? groups : null
}
