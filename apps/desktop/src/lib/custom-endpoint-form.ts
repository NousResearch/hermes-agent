import type { ReactNode } from 'react'

/**
 * Custom-endpoint editor decoration surface — the seam a plugin can augment
 * the Settings ▸ Providers ▸ Custom Endpoints editor through, with the SAME
 * registry schema as every other surface (statusbar, composer, session rows):
 *
 *   render area (`data`):  customEndpoint.form — inline inside the editor
 *                                               card, between the option
 *                                               checkboxes and the action
 *                                               buttons.
 *
 * Core keeps ownership of the form fields, validation, and the save path —
 * this seam AUGMENTS the editor (e.g. "also create this endpoint on other
 * profiles"), it never replaces a field or the Save button. Contributions
 * receive a read-only snapshot of the draft and the last successful save;
 * they render `null` when they have nothing to offer.
 *
 * Credential hygiene: the draft snapshot carries `hasApiKey`, never the key
 * value. A plugin that needs the credential resolves it through Hermes' own
 * secret scope on the backend, like every other credential path.
 */

export const CUSTOM_ENDPOINT_FORM_AREA = 'customEndpoint.form'

/** Read-only snapshot of the editor draft handed to contributions. */
export interface CustomEndpointFormDraft {
  id: string
  name: string
  baseUrl: string
  model: string
  apiMode: string
  contextLength: string
  makeDefault: boolean
  discoverModels: boolean
  /** Whether the draft carries a key value — never the value itself. */
  hasApiKey: ***
}

/** The endpoint core just saved, handed to contributions after a successful
 *  save so they can react (e.g. replicate it to other profiles). */
export interface CustomEndpointSaved {
  id: string
  name: string
  base_url: string
  model: string
  has_api_key: ***
  /** Monotonic per save (`Date.now()`), so repeated saves of the same
   *  endpoint re-notify contributions. */
  savedAt: number
}

/** Props handed to a `customEndpoint.form` contribution's `render`. */
export interface CustomEndpointFormProps {
  /** The profile the editor is scoped to; `null` = the app's active profile. */
  profile: string | null
  draft: CustomEndpointFormDraft
  /** Last successful save in this editor session; `null` before the first. */
  saved: CustomEndpointSaved | null
  /** True while core's save request is in flight. */
  saving: boolean
}

/** Payload of a `customEndpoint.form` contribution's `data`. */
export interface CustomEndpointFormContribution {
  /** Renders the augmentation, or `null` to leave the editor untouched.
   *  Mounted as a component inside the contribution error boundary, so it can
   *  subscribe to its own stores; a throw degrades to an inline error, not a
   *  dead editor. */
  render: (props: CustomEndpointFormProps) => ReactNode
}
