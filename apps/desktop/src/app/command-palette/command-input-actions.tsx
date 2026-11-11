import { Dialog as DialogPrimitive } from 'radix-ui'
import type { ReactNode } from 'react'

import { SurfaceCloseButton } from '@/components/ui/surface-close'

export function CommandInputActions({ extra }: { extra?: ReactNode }) {
  return (
    <>
      {extra}
      <DialogPrimitive.Close asChild>
        <SurfaceCloseButton />
      </DialogPrimitive.Close>
    </>
  )
}
