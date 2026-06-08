import type { RefObject } from 'react'

import { SearchField } from '@/components/ui/search-field'

interface OverlaySearchInputProps {
  containerClassName?: string
  inputRef?: RefObject<HTMLInputElement | null>
  loading?: boolean
  onChange: (value: string) => void
  placeholder: string
  value: string
}

// Borderless underline search — matches the tools/skills page (PageSearchShell).
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
      containerClassName={containerClassName}
      inputRef={inputRef}
      loading={loading}
      onChange={onChange}
      placeholder={placeholder}
      value={value}
    />
  )
}
