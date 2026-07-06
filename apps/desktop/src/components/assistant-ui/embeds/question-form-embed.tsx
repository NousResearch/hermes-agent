import { asRecords, asStrings, asText, InvalidPlanBlock, parseJsonRecord, PlanBlockShell } from './plan-block-utils'
import type { RichFenceProps } from './types'

function fieldValue(field: Record<string, unknown>): string {
  const value = field.default ?? field.value

  if (Array.isArray(value)) {
    return value.map(item => asText(item)).filter(Boolean).join(', ')
  }

  return asText(value)
}

export default function QuestionFormRenderer({ code }: RichFenceProps) {
  const form = parseJsonRecord(code)

  if (!form) {
    return <InvalidPlanBlock code={code} kind="question-form" />
  }

  const title = asText(form.title, 'Questions')
  const fields = asRecords(form.fields ?? form.questions)

  if (fields.length === 0) {
    return <InvalidPlanBlock code={code} kind="question-form" />
  }

  return (
    <PlanBlockShell title={title}>
      <div className="grid gap-3">
        {fields.map((field, index) => {
          const id = asText(field.id ?? field.name, `question_${index + 1}`)
          const label = asText(field.label ?? field.question ?? field.title, id)
          const type = asText(field.type, 'text')
          const value = fieldValue(field)
          const options = asStrings(field.options)

          return (
            <section className="grid gap-2 rounded-lg border border-(--ui-stroke-tertiary) bg-muted/15 p-3" key={id}>
              <label className="text-sm font-semibold" htmlFor={`question-form-${id}`}>
                {label}
              </label>
              {type === 'checkbox' ? (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <input checked={value === 'true'} disabled id={`question-form-${id}`} readOnly type="checkbox" />
                  <span>{value || 'Unchecked'}</span>
                </div>
              ) : options.length > 0 ? (
                <select className="rounded-md border border-input bg-background px-2 py-1 text-sm" disabled id={`question-form-${id}`} value={value || options[0]}>
                  {options.map(option => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  className="rounded-md border border-input bg-background px-2 py-1 text-sm"
                  disabled
                  id={`question-form-${id}`}
                  placeholder={value || 'No default answer'}
                  readOnly
                  value={value}
                />
              )}
              {field.required ? <div className="text-xs font-medium text-amber-600 dark:text-amber-300">Required</div> : null}
            </section>
          )
        })}
      </div>
    </PlanBlockShell>
  )
}
