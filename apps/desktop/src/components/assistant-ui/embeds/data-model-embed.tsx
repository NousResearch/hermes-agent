import { asRecords, asText, InvalidPlanBlock, MetaBadge, parseJsonRecord, PlanBlockShell } from './plan-block-utils'
import type { RichFenceProps } from './types'

export default function DataModelRenderer({ code }: RichFenceProps) {
  const model = parseJsonRecord(code)

  if (!model) {
    return <InvalidPlanBlock code={code} kind="data-model" />
  }

  const title = asText(model.title, 'Data model')
  const entities = asRecords(model.entities ?? model.models ?? model.tables)

  if (entities.length === 0) {
    return <InvalidPlanBlock code={code} kind="data-model" />
  }

  return (
    <PlanBlockShell title={title}>
      <div className="grid gap-3 sm:grid-cols-2">
        {entities.map((entity, index) => {
          const name = asText(entity.name ?? entity.title, `Entity ${index + 1}`)
          const fields = asRecords(entity.fields ?? entity.columns ?? entity.properties)

          return (
            <section className="rounded-lg border border-(--ui-stroke-tertiary) bg-muted/15" key={`${name}-${index}`}>
              <div className="border-b border-(--ui-stroke-tertiary) px-3 py-2 text-sm font-semibold">{name}</div>
              <div className="divide-y divide-(--ui-stroke-tertiary)">
                {fields.length > 0 ? (
                  fields.map((field, fieldIndex) => {
                    const fieldName = asText(field.name ?? field.key, `field_${fieldIndex + 1}`)
                    const type = asText(field.type ?? field.kind, 'unknown')
                    const note = asText(field.note ?? field.description)

                    return (
                      <div className="grid gap-1 px-3 py-2 text-xs" key={`${fieldName}-${fieldIndex}`}>
                        <div className="flex min-w-0 items-center justify-between gap-2">
                          <span className="truncate font-mono font-semibold">{fieldName}</span>
                          <MetaBadge>{type}</MetaBadge>
                        </div>
                        {note ? <div className="text-muted-foreground">{note}</div> : null}
                      </div>
                    )
                  })
                ) : (
                  <div className="px-3 py-2 text-xs text-muted-foreground">No fields</div>
                )}
              </div>
            </section>
          )
        })}
      </div>
    </PlanBlockShell>
  )
}
