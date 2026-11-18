import { useState } from 'react'

import { AccentSwatch } from '@/components/ui/accent-swatch'
import { Checkbox, CheckboxMark } from '@/components/ui/checkbox'
import { Chip } from '@/components/ui/chip'
import { ColorSwatches } from '@/components/ui/color-swatches'
import type { ControlVariantProps } from '@/components/ui/control'
import { Field, FieldHint } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { SanitizedInput } from '@/components/ui/sanitized-input'
import { SearchField } from '@/components/ui/search-field'
import { SegmentedControl } from '@/components/ui/segmented-control'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { SetupFieldList } from '@/components/ui/setup-field-list'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { PROFILE_SWATCHES } from '@/lib/profile-color'

import { everyOf, type GalleryEntry } from './types'

const CONTROL_SIZES = everyOf<NonNullable<ControlVariantProps['size']>>()('xs', 'sm', 'default', 'lg')

function SelectDemo() {
  const [value, setValue] = useState('daily')

  return (
    <Select onValueChange={setValue} value={value}>
      <SelectTrigger aria-label="How often">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectGroup>
          <SelectLabel>How often</SelectLabel>
          <SelectItem value="hourly">Every hour</SelectItem>
          <SelectItem value="daily">Every day</SelectItem>
          <SelectItem value="weekly">Every week</SelectItem>
        </SelectGroup>
      </SelectContent>
    </Select>
  )
}

function SegmentedDemo() {
  const [value, setValue] = useState<'all' | 'images' | 'files'>('all')

  return (
    <SegmentedControl
      onChange={setValue}
      options={[
        { id: 'all', label: 'All' },
        { id: 'images', label: 'Images' },
        { id: 'files', label: 'Files' }
      ]}
      value={value}
    />
  )
}

function SearchDemo({ variant }: { variant: 'box' | 'underline' }) {
  const [value, setValue] = useState('')

  return <SearchField onChange={setValue} placeholder="Search library" value={value} variant={variant} />
}

function ChipDemo({ variant }: { variant: 'card' | 'pill' }) {
  const [on, setOn] = useState(variant === 'pill')

  return (
    <Chip
      label="Research"
      on={on}
      onToggle={() => setOn(!on)}
      sub={variant === 'card' ? 'Reads the web' : undefined}
      variant={variant}
    />
  )
}

function SwatchesDemo() {
  const [value, setValue] = useState<null | string>(PROFILE_SWATCHES[0] ?? null)

  return <ColorSwatches clearLabel="No colour" onChange={setValue} swatches={PROFILE_SWATCHES} value={value} />
}

function SetupFieldsDemo() {
  const [draft, setDraft] = useState<Record<string, string>>({})

  return (
    <SetupFieldList
      draft={draft}
      fields={[
        { name: 'workspace', prompt: 'Workspace', required: true, secret: false, default: '' },
        { name: 'token', prompt: 'Access token', required: false, secret: true, default: '' }
      ]}
      onChange={(name, value) => setDraft(current => ({ ...current, [name]: value }))}
    />
  )
}

function SanitizedDemo() {
  const [value, setValue] = useState('')

  return (
    <SanitizedInput
      aria-label="Agent name"
      onValueChange={setValue}
      placeholder="lowercase-only"
      sanitize={raw => raw.toLowerCase().replace(/[^a-z0-9-]/g, '-')}
      value={value}
    />
  )
}

export const INPUTS: readonly GalleryEntry[] = [
  {
    title: 'Input',
    source: 'components/ui/input.tsx',
    use: 'One line of text; the control chrome every field shares (components/ui/control.ts).',
    group: 'inputs',
    covers: ['components/ui/control.ts'],
    examples: [
      ...CONTROL_SIZES.map(size => ({
        label: `size ${size}`,
        render: () => <Input aria-label={`Input ${size}`} placeholder="Agent name" size={size} />
      })),
      {
        label: 'prefix, suffix',
        render: () => <Input aria-label="Budget" defaultValue="20" prefix="$" suffix="/ month" />
      },
      { label: 'invalid', render: () => <Input aria-invalid aria-label="Invalid" defaultValue="not a url" /> },
      { label: 'disabled', render: () => <Input aria-label="Disabled" defaultValue="Read only" disabled /> }
    ]
  },
  {
    title: 'SanitizedInput',
    source: 'components/ui/sanitized-input.tsx',
    use: 'An Input that rewrites what is typed as it is typed (names, slugs).',
    group: 'inputs',
    examples: [{ label: 'default', render: () => <SanitizedDemo /> }]
  },
  {
    title: 'Textarea',
    source: 'components/ui/textarea.tsx',
    use: 'Several lines of text, with the Input chrome.',
    group: 'inputs',
    examples: [
      {
        label: 'default',
        render: () => <Textarea aria-label="Instructions" placeholder="What should this agent do?" />
      }
    ]
  },
  {
    title: 'Field',
    source: 'components/ui/field.tsx',
    use: 'A label over a control, with an optional hint or error under it.',
    group: 'inputs',
    examples: [
      {
        label: 'with hint',
        render: () => (
          <Field htmlFor="ds-field-name" label="Name">
            <Input defaultValue="research" id="ds-field-name" />
            <FieldHint>Shown in the sidebar.</FieldHint>
          </Field>
        )
      },
      {
        label: 'optional, error',
        render: () => (
          <Field htmlFor="ds-field-url" label="Website" optional optionalLabel="Optional">
            <Input aria-invalid defaultValue="example" id="ds-field-url" />
            <FieldHint error>Enter a full address.</FieldHint>
          </Field>
        )
      }
    ]
  },
  {
    title: 'SetupFieldList',
    source: 'components/ui/setup-field-list.tsx',
    use: 'The fields a connector or channel asks for, from its own schema.',
    group: 'inputs',
    examples: [{ label: 'required and secret', render: () => <SetupFieldsDemo /> }]
  },
  {
    title: 'Select',
    source: 'components/ui/select.tsx',
    use: 'One choice from a list too long for a segmented control.',
    group: 'inputs',
    examples: [{ label: 'grouped', render: () => <SelectDemo /> }]
  },
  {
    title: 'SegmentedControl',
    source: 'components/ui/segmented-control.tsx',
    use: 'Two to four mutually exclusive choices that switch a view or a mode.',
    group: 'inputs',
    examples: [
      { label: 'default', render: () => <SegmentedDemo /> },
      {
        label: 'disabled',
        render: () => (
          <SegmentedControl
            disabled
            onChange={() => undefined}
            options={[
              { id: 'on', label: 'On' },
              { id: 'off', label: 'Off' }
            ]}
            value="on"
          />
        )
      }
    ]
  },
  {
    title: 'Checkbox',
    source: 'components/ui/checkbox.tsx',
    use: 'An independent yes / no in a list; CheckboxMark when the whole row is the toggle.',
    group: 'inputs',
    examples: [
      { label: 'unchecked', render: () => <Checkbox aria-label="Unchecked" /> },
      { label: 'checked', render: () => <Checkbox aria-label="Checked" defaultChecked /> },
      { label: 'indeterminate', render: () => <Checkbox aria-label="Indeterminate" checked="indeterminate" /> },
      { label: 'disabled', render: () => <Checkbox aria-label="Disabled" disabled /> },
      { label: 'CheckboxMark', render: () => <CheckboxMark checked /> }
    ]
  },
  {
    title: 'Switch',
    source: 'components/ui/switch.tsx',
    use: 'A setting that takes effect at once.',
    group: 'inputs',
    examples: [
      { label: 'off', render: () => <Switch aria-label="Off" /> },
      { label: 'on', render: () => <Switch aria-label="On" defaultChecked /> },
      { label: 'size xs', render: () => <Switch aria-label="Small" defaultChecked size="xs" /> },
      { label: 'disabled', render: () => <Switch aria-label="Disabled" disabled /> }
    ]
  },
  {
    title: 'Slider',
    source: 'components/ui/slider.tsx',
    use: 'A value on a continuous range (intensity, size).',
    group: 'inputs',
    examples: [{ label: 'default', render: () => <Slider aria-label="Intensity" defaultValue={40} /> }]
  },
  {
    title: 'SearchField',
    source: 'components/ui/search-field.tsx',
    use: 'The one search box: filter a list or a page, Escape clears it.',
    group: 'inputs',
    examples: [
      { label: 'box', render: () => <SearchDemo variant="box" /> },
      { label: 'underline', render: () => <SearchDemo variant="underline" /> }
    ]
  },
  {
    title: 'Chip',
    source: 'components/ui/chip.tsx',
    use: 'A pressable choice among several that can be on together.',
    group: 'inputs',
    examples: [
      { label: 'card', render: () => <ChipDemo variant="card" /> },
      { label: 'pill', render: () => <ChipDemo variant="pill" /> }
    ]
  },
  {
    title: 'ColorSwatches',
    source: 'components/ui/color-swatches.tsx',
    use: 'Pick a colour for an agent or a project, or clear it.',
    group: 'inputs',
    examples: [{ label: 'default', render: () => <SwatchesDemo /> }]
  },
  {
    title: 'AccentSwatch',
    source: 'components/ui/accent-swatch.tsx',
    use: 'One accent colour as a round swatch, or the custom-colour well.',
    group: 'inputs',
    examples: [
      { label: 'active', render: () => <AccentSwatch active hex={PROFILE_SWATCHES[0]} name="First swatch" /> },
      { label: 'rest', render: () => <AccentSwatch active={false} hex={PROFILE_SWATCHES[1]} name="Second swatch" /> }
    ]
  }
]
