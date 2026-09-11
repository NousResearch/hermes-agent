export const EASE = 'cubic-bezier(0.22, 1, 0.36, 1)'

// Hermes blue — the app's --theme-primary (#0053fd), lifted for dark ground.
export const BLUE = '#4d8dff'
export const BLUE_DIM = 'rgba(77, 141, 255, 0.55)'
export const BLUE_FAINT = 'rgba(77, 141, 255, 0.4)'

// One shadow for every floating surface — --shadow-nous's recipe (single top
// light, layered contact→ambient, x=0, negative spread pulling each layer
// inward) restated for a dark ground at LOW opacity, so cards sit on the
// frost instead of dragging black halos across it.
export const NOUS_SHADOW =
  '0 2px 4px -2px rgba(0,0,0,0.3), 0 8px 12px -6px rgba(0,0,0,0.24), 0 20px 28px -14px rgba(0,0,0,0.2), 0 36px 48px -28px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.05)'
