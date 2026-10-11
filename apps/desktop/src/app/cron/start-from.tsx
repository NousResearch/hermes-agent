import { Field, FieldHint } from '@/components/ui/field'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import type { AutomationBlueprint, CronJob, CronJobCarriedFields } from '@/hermes'
import type { Translations } from '@/i18n'

import { jobIsScriptOnly } from './cron-job-model'
import { jobTitle } from './job-state'

// "Start from" values: the blank editor, `job:<id>` for a copy of an existing
// job (listed first: the user's own, usually short list), or a blueprint key. Blueprint keys never start with the job prefix.
export const BLANK_START = 'custom'
const JOB_START_PREFIX = 'job:'

export const jobStartValue = (jobId: string): string => `${JOB_START_PREFIX}${jobId}`

export function startJob(value: string, jobs: readonly CronJob[]): CronJob | null {
  return value.startsWith(JOB_START_PREFIX) ? (jobs.find(job => jobStartValue(job.id) === value) ?? null) : null
}

// A script-only job has no prompt for the editor to copy.
export const copyableJobs = (jobs: readonly CronJob[]): CronJob[] => jobs.filter(job => !jobIsScriptOnly(job))

const CARRIED_LISTS = ['context_from', 'enabled_toolsets', 'skills'] as const
const CARRIED_TEXT = ['base_url', 'workdir'] as const

// The settings a job carries that the editor doesn't show, so a copy runs the
// same way as its source. Empty values are left out rather than sent as blanks.
export function carriedFields(source: Partial<Record<keyof CronJobCarriedFields, unknown>>): CronJobCarriedFields {
  const carried: CronJobCarriedFields = {}

  for (const key of CARRIED_LISTS) {
    const value = source[key]

    if (Array.isArray(value) && value.length > 0) {
      carried[key] = value.map(String)
    }
  }

  for (const key of CARRIED_TEXT) {
    const value = source[key]

    if (typeof value === 'string' && value.trim()) {
      carried[key] = value
    }
  }

  return carried
}

export function StartFromField({
  blueprints,
  c,
  customizedFrom,
  jobs,
  onChange,
  value
}: {
  blueprints: readonly AutomationBlueprint[]
  c: Translations['cron']
  customizedFrom: null | string
  jobs: readonly CronJob[]
  onChange: (next: string) => void
  value: string
}) {
  if (blueprints.length === 0 && jobs.length === 0) {
    return null
  }

  const blueprint = blueprints.find(item => item.key === value)
  const hint = customizedFrom ? c.blueprints.customizedFrom(customizedFrom) : blueprint?.description

  return (
    <Field htmlFor="cron-template" label={c.blueprints.startFrom}>
      <Select onValueChange={onChange} value={value}>
        <SelectTrigger className="h-9 rounded-md" id="cron-template">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={BLANK_START}>{c.blueprints.custom}</SelectItem>
          {jobs.length > 0 && (
            <SelectGroup>
              <SelectLabel>{c.blueprints.copyGroup}</SelectLabel>
              {jobs.map(job => (
                <SelectItem key={job.id} value={jobStartValue(job.id)}>
                  {jobTitle(job)}
                </SelectItem>
              ))}
            </SelectGroup>
          )}
          {blueprints.length > 0 && (
            <SelectGroup>
              <SelectLabel>{c.blueprints.recipesGroup}</SelectLabel>
              {blueprints.map(item => (
                <SelectItem key={item.key} value={item.key}>
                  {item.title}
                  {item.plugin && <span className="ml-1.5 text-muted-foreground">· {item.plugin}</span>}
                </SelectItem>
              ))}
            </SelectGroup>
          )}
        </SelectContent>
      </Select>
      {hint && <FieldHint>{hint}</FieldHint>}
    </Field>
  )
}
