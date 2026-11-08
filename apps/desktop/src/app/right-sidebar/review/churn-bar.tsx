import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import type { HermesReviewFile } from '@/global'
import { $reviewMaxChurn } from '@/store/review'

// Per-row "digital rain" churn bar: a right-anchored, clipped stream of
// Matrix-ish glyphs whose width is the file's churn relative to the biggest
// changed file. Not wired in — drop `<ChurnBar file={file} />` into a review row
// (which must be `relative isolate overflow-hidden`) to revive it.
const GLYPHS = 'ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾅﾆﾇﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾚﾜ0123456789:=*+<>¦'

const MASK = 'linear-gradient(to left, #000 45%, transparent)'

// Deterministic glyph run (FNV-1a seed → xorshift) so a file's rain is stable
// across renders instead of reshuffling every paint.
function rain(seed: string, len: number): string {
  let h = 2166136261

  for (let i = 0; i < seed.length; i++) {
    h = Math.imul(h ^ seed.charCodeAt(i), 16777619)
  }

  let out = ''

  for (let i = 0; i < len; i++) {
    h ^= h << 13
    h ^= h >>> 17
    h ^= h << 5
    out += GLYPHS[Math.abs(h) % GLYPHS.length]
  }

  return out
}

export function ChurnBar({ file }: { file: HermesReviewFile }) {
  const max = useStore($reviewMaxChurn)
  const fill = useMemo(() => rain(file.path, 200), [file.path])
  const width = max > 0 ? ((file.added + file.removed) / max) * 100 : 0

  if (width <= 0) {
    return null
  }

  return (
    <span
      aria-hidden
      className="pointer-events-none absolute inset-y-0 right-0 -z-10 block overflow-hidden text-right font-mono text-[0.7rem] leading-6 tracking-tight whitespace-nowrap opacity-30 dark:opacity-40"
      style={{
        WebkitMaskImage: MASK,
        color: `var(--ui-${file.added >= file.removed ? 'green' : 'red'})`,
        maskImage: MASK,
        width: `${width}%`
      }}
    >
      {fill}
    </span>
  )
}
