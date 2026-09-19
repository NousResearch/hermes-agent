/**
 * Task text as markdown. Descriptions are written as markdown everywhere else
 * in the product — the decomposer writes numbered plans, operators paste fenced
 * commands — and the drawer used to print the source, so a plan read as one
 * run-on block. Streamdown does the parse (the same renderer core chat uses);
 * the components here keep the typography at drawer scale.
 *
 * Links go through `os.openExternal`: a bare `<a>` would navigate the renderer
 * itself, replacing the app with the page.
 */

import { cn, Streamdown } from '@hermes/plugin-sdk'
import type { ComponentProps } from 'react'

import { pluginOs } from './api'

const MD_TAG_CLASSES = {
  blockquote: 'mb-2 border-l-2 border-(--ui-border) pl-2.5 text-(--ui-text-tertiary) italic last:mb-0',
  h1: 'mt-3 mb-1.5 text-[0.9375rem] font-semibold tracking-tight first:mt-0',
  h2: 'mt-3 mb-1.5 text-[0.875rem] font-semibold tracking-tight first:mt-0',
  h3: 'mt-2.5 mb-1 text-[0.8125rem] font-semibold first:mt-0',
  h4: 'mt-2.5 mb-1 text-[0.8125rem] font-semibold first:mt-0',
  hr: 'my-3 border-(--ui-border)',
  li: 'mt-0.5 leading-relaxed',
  ol: 'mb-2 list-decimal pl-5 last:mb-0',
  p: 'mb-2 leading-relaxed last:mb-0',
  pre: 'mb-2 overflow-x-auto rounded bg-(--ui-bg-quaternary) p-2 font-mono text-[0.6875rem] leading-relaxed last:mb-0 [&_pre]:m-0 [&_pre]:bg-transparent! [&_pre]:p-0',
  table: 'mb-2 w-full border-collapse text-[0.75rem] last:mb-0',
  td: 'border-t border-(--ui-border) px-1.5 py-1 align-top',
  th: 'px-1.5 py-1 text-left font-semibold',
  ul: 'mb-2 list-disc pl-5 last:mb-0'
} as const

// `node` is react-markdown's AST handle; it is not a DOM attribute and React
// warns (and serializes "[object Object]") if it reaches the element.
type MdProps<T extends keyof typeof MD_TAG_CLASSES> = ComponentProps<T> & { node?: unknown }

function tagged<T extends keyof typeof MD_TAG_CLASSES>(Tag: T) {
  const base = MD_TAG_CLASSES[Tag]

  const Component = (({ className, node: _node, ...rest }: MdProps<T>) => {
    const Element = Tag as React.ElementType

    return <Element className={cn(base, className)} {...rest} />
  }) as React.FC<ComponentProps<T>>

  Component.displayName = `Md.${Tag}`

  return Component
}

/** Inline code keeps the pill; a fenced block is already inside a styled
 *  `<pre>`, so it only needs the monospace run. */
function MarkdownCode({ children, className, node: _node, ...rest }: ComponentProps<'code'> & { node?: unknown }) {
  const fenced = /language-/.test(className ?? '')

  return (
    <code
      className={cn(
        fenced ? 'font-mono' : 'rounded bg-(--ui-bg-quaternary) px-1 py-0.5 font-mono text-[0.9em]',
        className
      )}
      {...rest}
    >
      {children}
    </code>
  )
}

function MarkdownLink({ children, className, href, node: _node, ...rest }: ComponentProps<'a'> & { node?: unknown }) {
  return (
    <a
      className={cn('underline decoration-(--ui-text-quaternary) underline-offset-2 hover:text-foreground', className)}
      href={href}
      onClick={event => {
        event.preventDefault()
        event.stopPropagation()

        if (href) {
          void pluginOs()?.openExternal(href)
        }
      }}
      {...rest}
    >
      {children}
    </a>
  )
}

const MARKDOWN_COMPONENTS = {
  a: MarkdownLink,
  blockquote: tagged('blockquote'),
  code: MarkdownCode,
  h1: tagged('h1'),
  h2: tagged('h2'),
  h3: tagged('h3'),
  h4: tagged('h4'),
  hr: tagged('hr'),
  li: tagged('li'),
  ol: tagged('ol'),
  p: tagged('p'),
  pre: tagged('pre'),
  table: tagged('table'),
  td: tagged('td'),
  th: tagged('th'),
  ul: tagged('ul')
}

/** Markdown at drawer scale. `text` is trusted no more than any other card
 *  field — Streamdown renders to React elements, never to raw HTML. */
export function Markdown({ className, text }: { className?: string; text: string }) {
  return (
    <div
      className={cn('text-[0.8125rem] text-(--ui-text-secondary)', className)}
      data-markdown="true"
      data-selectable-text="true"
    >
      <Streamdown components={MARKDOWN_COMPONENTS} controls={false} mode="static" parseIncompleteMarkdown={false}>
        {text}
      </Streamdown>
    </div>
  )
}
