export const EASE = 'cubic-bezier(0.22, 1, 0.36, 1)'

// Hermes blue: the app's --theme-primary (#0053fd), lightened for a dark background.
export const BLUE = '#4d8dff'
export const BLUE_DIM = 'rgba(77, 141, 255, 0.55)'
export const BLUE_FAINT = 'rgba(77, 141, 255, 0.4)'

// One shadow for every floating surface. It follows --shadow-nous (single top
// light, layered contact to ambient, x = 0, negative spread on each layer) at
// lower opacity for the dark background, so cards do not cast black halos over
// the frost.
export const NOUS_SHADOW =
  '0 2px 4px -2px rgba(0,0,0,0.3), 0 8px 12px -6px rgba(0,0,0,0.24), 0 20px 28px -14px rgba(0,0,0,0.2), 0 36px 48px -28px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.05)'
