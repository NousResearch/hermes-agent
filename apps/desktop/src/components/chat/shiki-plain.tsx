/** Matches highlighted markup so loading a grammar changes color, not layout. */
export function PlainShiki({ code }: { code: string }) {
  return (
    <div className="rs-root not-prose">
      <pre className="shiki" style={{ backgroundColor: 'transparent', margin: 0 }}>
        <code>{code}</code>
      </pre>
    </div>
  )
}
