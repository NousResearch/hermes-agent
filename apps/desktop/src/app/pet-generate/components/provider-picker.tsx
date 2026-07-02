import { useStore } from '@nanostores/react'

import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Check, ChevronDown } from '@/lib/icons'
import { $petGenProvider, $petGenProviders, setPetGenProvider } from '@/store/pet-generate'

// Image-backend picker for pet generation — the composer's model-pill pattern:
// a quiet trigger + a dropdown of options. No per-option notes: every backend
// resolves to the same faithful OpenAI image model, so there's no tradeoff to
// describe. Hidden unless there are 2+ reference-capable backends (nothing to pick).
export function ProviderPicker() {
  const providers = useStore($petGenProviders)
  const picked = useStore($petGenProvider)

  if (providers.length < 2) {
    return null
  }

  const fallback = providers.find(p => p.default) ?? providers[0]
  const current = providers.find(p => p.name === picked) ?? fallback

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        {/* Plain text affordance (matches "Add a reference"), not a padded pill. */}
        <button
          className="flex h-6 items-center gap-1 text-[0.6875rem] text-(--ui-text-tertiary) transition hover:text-foreground"
          type="button"
        >
          {current?.label}
          <ChevronDown className="size-3" />
        </button>
      </DropdownMenuTrigger>
      {/* The picker lives inside the pet-gen Dialog (z-130) and portals to body,
          so lift its menu above the dialog or it opens behind it. */}
      <DropdownMenuContent align="start" className="z-[140]">
        {providers.map(provider => (
          <DropdownMenuItem
            className="flex items-center gap-1.5"
            key={provider.name}
            // Picking the default clears the override (no need to pin it).
            onSelect={() => setPetGenProvider(provider.default ? '' : provider.name)}
          >
            <span className="min-w-0 flex-1 truncate font-medium text-foreground">{provider.label}</span>
            {provider.name === current?.name && <Check className="size-3.5 text-primary" />}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
