interface SessionTokensStatusProps {
  input: string
  output: string
}

export function SessionTokensStatus({ input, output }: SessionTokensStatusProps) {
  return (
    <span aria-label={`Session tokens: ${input} in, ${output} out`} className="inline-flex items-center gap-1.5 tabular-nums">
      <span className="inline-flex items-baseline gap-0.5">
        <span className="text-[0.6875rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary)">In</span>
        <span>{input}</span>
      </span>
      <span aria-hidden="true" className="text-(--ui-text-tertiary)">
        ·
      </span>
      <span className="inline-flex items-baseline gap-0.5">
        <span className="text-[0.6875rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary)">Out</span>
        <span>{output}</span>
      </span>
    </span>
  )
}
