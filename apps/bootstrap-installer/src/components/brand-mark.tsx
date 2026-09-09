import { cn } from '../lib/utils'

const assetPath = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`

// Brand badge: the North Forge mark (anvil + flame), white on black —
// rendered on a black tile so it reads identically in light and dark.
export function BrandMark({ className, ...props }: React.ComponentProps<'span'>) {
  return (
    <span
      className={cn('inline-flex size-14 shrink-0 items-center justify-center overflow-hidden rounded-[3px] bg-black', className)}
      {...props}
    >
      <img alt="" className="size-full object-contain" src={assetPath('north-forge-mark.png')} />
    </span>
  )
}
