import { cn } from '@/lib/utils'
import { useTheme } from '@/themes'

const assetPath = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`

// Brand badge: the rabbit mark in the app-icon squircle, theme-aware —
// light mode shows the black rabbit on the white squircle, dark mode the
// white rabbit on the #0d1117 squircle. The mark PNG carries the rounded
// shape and transparent corners, so no tile/rounding classes here. Sized
// via className (default size-14).
export function BrandMark({ className, ...props }: React.ComponentProps<'span'>) {
  const { renderedMode } = useTheme()
  const dark = renderedMode === 'dark'

  return (
    <span className={cn('inline-flex size-14 shrink-0 items-center justify-center', className)} {...props}>
      <img alt="" className="size-full object-contain" src={assetPath(dark ? 'brand-icon-dark.png' : 'brand-icon.png')} />
    </span>
  )
}
