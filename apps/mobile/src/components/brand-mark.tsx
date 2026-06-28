import { cn } from '@/lib/utils'

import { baseUrl } from '@/lib/env'

const assetPath = (path: string) => `${baseUrl}${path.replace(/^\/+/, '')}`

// Brand badge: nous-girl mark on a white tile, identical in light/dark.
// Fills the tile (softly rounded); size via className (default size-14).
export function BrandMark({ className, ...props }: React.ComponentProps<'span'>) {
  return (
    <span
      className={cn(
        'inline-flex size-14 shrink-0 items-center justify-center overflow-hidden rounded-md bg-white',
        className
      )}
      {...props}
    >
      <img alt="" className="size-full object-contain" src={assetPath('nous-girl.jpg')} />
    </span>
  )
}
