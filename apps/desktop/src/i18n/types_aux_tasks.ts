/** Copy for one built-in auxiliary-model slot row (Settings → Models → Auxiliary models). */
export interface AuxTaskCopy {
  label: string
  hint: string
}

/** Built-in slot key → row copy; plugin slots bring their own label/hint from the backend. */
export type AuxTaskCopyMap = Record<string, AuxTaskCopy>
