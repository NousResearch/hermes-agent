import type { ReactNode } from 'react'

interface HighlighterProps {
  as?: keyof JSX.IntrinsicElements
  children?: ReactNode
  className?: string
}

export function useShikiHighlighter(): ReactNode | null {
  return null
}

export default function ShikiHighlighter({ as: Tag = 'pre', children, className }: HighlighterProps) {
  const content = typeof children === 'string' ? children : String(children ?? '')

  return (
    <Tag className={className}>
      <code className="block whitespace-pre">{content}</code>
    </Tag>
  )
}
