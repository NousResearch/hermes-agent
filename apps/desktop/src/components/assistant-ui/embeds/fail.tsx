import { useI18n } from '@/i18n'

export function EmbedFail({ label }: { label: string }) {
  const copy = useI18n().t.assistant.embeds

  return (
    <span className="grid min-h-32 w-full place-items-center p-4">
      <span className="text-xs font-medium text-(--ui-red)">{copy.failedToLoad(label)}</span>
    </span>
  )
}
