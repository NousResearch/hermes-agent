import { cn } from '@/lib/utils'

const assetPath = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`

// Brand badge: the nous-girl mark shown as-is (black line-art on its own white
// card), identical in light and dark. Replaces the generic Sparkles hero glyph.
// The art fills the tile (no padding, no radius); size comes from the caller's
// className (default size-14).
export function BrandMark({ className, ...props }: React.ComponentProps<'span'>) {
  return (
    <span
      className={cn(
        'inline-flex size-14 shrink-0 items-center justify-center bg-white',
        className
      )}
      {...props}
    >
      <img alt="" className="size-full object-contain" src={assetPath('nous-girl.jpg')} />
    </span>
  )
}
