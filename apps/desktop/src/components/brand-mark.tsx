import { cn } from '@/lib/utils'

const assetPath = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`

// Brand badge: the Hermes logo, shown as-is over a transparent tile in light
// mode and a near-black (#222) tile in dark. Replaces the generic Sparkles
// glyph used as a hero mark in overlays. The logo fills the tile (no padding,
// no radius); size comes from the caller's className (default size-14).
export function BrandMark({ className, ...props }: React.ComponentProps<'span'>) {
  return (
    <span
      className={cn(
        'inline-flex size-14 shrink-0 items-center justify-center dark:bg-[#222]',
        className
      )}
      {...props}
    >
      <img alt="" className="size-full object-contain" src={assetPath('karb.webp')} />
    </span>
  )
}
