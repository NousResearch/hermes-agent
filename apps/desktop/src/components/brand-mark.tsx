import { cn } from '@/lib/utils'

const assetPath = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`

// Agent Czesiek brand mark shared by setup and recovery screens.
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
      <img alt="Agent Czesiek" className="size-full object-contain" src={assetPath('apple-touch-icon.png')} />
    </span>
  )
}
