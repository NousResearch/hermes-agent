import 'i18next'

// Hermes Desktop uses flat i18n keys with a `namespace:section.subsection`
// convention for grep-friendliness (e.g. `settings:mcp.configured`).
//
// i18next's default `CustomTypeOptions.resources` infers nested objects from
// the JSON shape, which would require either nesting the JSON (`settings: {
// mcp: { configured: "..." } }`) or providing a literal-union of every
// dotted key — both fragile and ugly for our grep-friendly flat layout.
//
// We deliberately do NOT augment `resources` so `t()` keeps its default
// `TFunctionNonStrict` signature (any string key, any options object). The
// parity test (`src/locales/locales.test.ts`) is what actually guards key
// drift between locales.
declare module 'i18next' {
  interface CustomTypeOptions {
    defaultNS: 'translation'
    returnNull: false
  }
}

export {}
