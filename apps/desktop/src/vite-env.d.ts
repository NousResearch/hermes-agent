/// <reference types="vite/client" />

declare module '@tabler/icons-react/dist/esm/icons/*.mjs' {
  import type { Icon } from '@tabler/icons-react'

  const icon: Icon
  export default icon
}
