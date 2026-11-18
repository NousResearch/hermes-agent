import { inlineCode, type MarkdownBlock } from '@/lib/markdown-section'

function Inline({ text }: { text: string }) {
  return (
    <>
      {inlineCode(text).map((span, index) =>
        span.code ? <code key={index}>{span.text}</code> : <span key={index}>{span.text}</span>
      )}
    </>
  )
}

function Block({ block }: { block: MarkdownBlock }) {
  if (block.kind === 'paragraph') {
    return (
      <p className="ds-about">
        <Inline text={block.text} />
      </p>
    )
  }

  return (
    <div className="ds-table-scroll">
      <table className="ds-table">
        <thead>
          <tr>
            {block.table.head.map(cell => (
              <th key={cell} scope="col">
                <Inline text={cell} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {block.table.rows.map(row => (
            <tr key={row[0]}>
              {row.map((cell, index) => (
                <td key={index}>
                  <Inline text={cell} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** DESIGN.md's "Still to unify", as the document has it today. */
export function UnifySection({ blocks }: { blocks: readonly MarkdownBlock[] }) {
  return (
    <>
      {blocks.map((block, index) => (
        <Block block={block} key={index} />
      ))}
    </>
  )
}
