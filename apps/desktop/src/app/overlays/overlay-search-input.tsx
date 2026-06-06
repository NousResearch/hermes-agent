import type { RefObject } from 'react'

import { SearchField } from '@/components/ui/search-field'
import { cn } from '@/lib/utils'

interface OverlaySearchInputProps {
  containerClassName?: string
  inputRef?: RefObject<HTMLInputElement | null>
  loading?: boolean
  onChange: (value: string) => void
  placeholder: string
  value: string
}

export function OverlaySearchInput({
  containerClassName,
  inputRef,
  loading = false,
  onChange,
  placeholder,
  value
}: OverlaySearchInputProps) {
  return (
    <SearchField
      containerClassName={cn(
        'rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) px-2 shadow-sm focus-within:border-(--ui-stroke-secondary)',
        containerClassName
      )}
      inputClassName="h-8 text-[0.8125rem]"
      inputRef={inputRef}
      loading={loading}
      onChange={onChange}
      placeholder={placeholder}
      value={value}
    />
  )
}
