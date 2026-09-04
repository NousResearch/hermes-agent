interface ProductIdentity {
  /** True when this artifact is Hermes Light (remote-only client). */
  light: boolean
  /** True for a Store-submission build (Windows Store packaging identity). */
  store: boolean
  /** Display name. e.g. "Hermes Light" */
  displayName: string
  /** OS-level app identity. e.g. "com.nousresearch.hermes-light" */
  appId: string
  /** app name in pascal case. e.g. "HermesLight" */
  appNamePascal: string
  /** OS-level app identity w/ org prefix. e.g. "NousResearch.HermesLight" */
  msixAppIdWithOrg: string
  /** electron-updater feed channel this build publishes to. Stable tags:
   *  "latest" | "light"; canary tags: "canary" | "light-canary". Null
   *  for a store build (the Store owns its updates; no release feed). */
  channel: string | null
  /** Store-submission MSIX packaging identity. Present only when `store`. */
  storeMsix?: {
    identityName: string
    publisher: string
    publisherDisplayName: string
  }
}

declare const identity: ProductIdentity
export = identity
