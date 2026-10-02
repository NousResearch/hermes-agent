import { cn } from '@/lib/utils'

const assetPath = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`

// IX Agency mark. The PNG already carries the rounded tile, so size comes from className.
export function BrandMark({ className, ...props }: React.ComponentProps<'span'>) {
  return (
    <span className={cn('inline-flex size-14 shrink-0 items-center justify-center', className)} {...props}>
      <img alt="" className="size-full object-contain" src={assetPath('ix-agency.png')} />
    </span>
  )
}
