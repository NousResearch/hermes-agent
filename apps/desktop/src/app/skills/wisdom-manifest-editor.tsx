import { useId } from 'react'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'

import {
  parseWisdomManifest,
  type WisdomManifestV1,
  wisdomManifestValidationError,
  type WisdomPluginRequirement,
  type WisdomSystemSpecification,
  type WisdomToolRequirement
} from './wisdom-manifest'

interface SpecificationEditorProps {
  value: WisdomSystemSpecification
  disabled?: boolean
  onChange: (value: WisdomSystemSpecification) => void
}

interface ManifestEditorProps {
  value: string
  disabled?: boolean
  onChange: (value: string) => void
}

const PLATFORM_OPTIONS = [
  { value: 'macOS', label: 'macOS' },
  { value: 'Linux', label: 'Linux' },
  { value: 'Windows', label: 'Windows' }
] as const

const ARCHITECTURE_OPTIONS = [
  { value: 'arm64', label: 'ARM64 / Apple silicon' },
  { value: 'x86_64', label: 'x86-64 / AMD64' }
] as const

function FieldCopy({ label, description }: { label: string; description: string }) {
  return (
    <div>
      <div className="text-[0.68rem] font-medium">{label}</div>
      <p className="mt-0.5 text-[0.62rem] leading-4 text-muted-foreground">{description}</p>
    </div>
  )
}

function StringListField({
  label,
  description,
  itemLabel,
  value,
  disabled = false,
  placeholder,
  onChange
}: {
  label: string
  description: string
  itemLabel: string
  value: string[]
  disabled?: boolean
  placeholder?: string
  onChange: (value: string[]) => void
}) {
  const id = useId()

  return (
    <div className="space-y-2">
      <FieldCopy description={description} label={label} />
      {value.map((item, index) => (
        <div className="flex items-center gap-2" key={`${id}-${index}`}>
          <Input
            aria-label={`${itemLabel} ${index + 1}`}
            className="min-w-0 flex-1"
            disabled={disabled}
            maxLength={512}
            onChange={event =>
              onChange(value.map((existing, itemIndex) => (itemIndex === index ? event.target.value : existing)))
            }
            placeholder={placeholder}
            size="sm"
            value={item}
          />
          <Button
            aria-label={`Remove ${itemLabel} ${index + 1}`}
            disabled={disabled}
            onClick={() => onChange(value.filter((_, itemIndex) => itemIndex !== index))}
            size="xs"
            variant="text"
          >
            Remove
          </Button>
        </div>
      ))}
      <Button
        disabled={disabled || value.length >= 64}
        onClick={() => onChange([...value, ''])}
        size="xs"
        variant="outline"
      >
        Add {itemLabel}
      </Button>
    </div>
  )
}

function PresetMultiSelect({
  label,
  description,
  value,
  options,
  disabled,
  onChange
}: {
  label: string
  description: string
  value: string[]
  options: ReadonlyArray<{ value: string; label: string }>
  disabled?: boolean
  onChange: (value: string[]) => void
}) {
  const id = useId()
  const presets = new Set(options.map(option => option.value))
  const custom = value.filter(item => !presets.has(item))

  return (
    <div className="space-y-2">
      <FieldCopy description={description} label={label} />
      <div className="grid gap-2 sm:grid-cols-2">
        {options.map(option => {
          const checkboxId = `${id}-${option.value}`
          const checked = value.includes(option.value)

          return (
            <label className="flex cursor-pointer items-center gap-2 text-[0.68rem]" htmlFor={checkboxId} key={option.value}>
              <Checkbox
                checked={checked}
                disabled={disabled}
                id={checkboxId}
                onCheckedChange={next =>
                  onChange(next === true ? [...value, option.value] : value.filter(item => item !== option.value))
                }
              />
              {option.label}
            </label>
          )
        })}
      </div>
      <StringListField
        description={`Add a ${label.toLowerCase().replace(/s$/, '')} not listed above.`}
        disabled={disabled}
        itemLabel={`other ${label.toLowerCase().replace(/s$/, '')}`}
        label={`Other ${label.toLowerCase()}`}
        onChange={next => onChange([...value.filter(item => presets.has(item)), ...next])}
        value={custom}
      />
    </div>
  )
}

function BooleanField({
  id,
  label,
  description,
  checked,
  disabled,
  onChange
}: {
  id: string
  label: string
  description: string
  checked: boolean
  disabled?: boolean
  onChange: (checked: boolean) => void
}) {
  return (
    <label className="flex cursor-pointer items-start gap-2" htmlFor={id}>
      <Checkbox
        checked={checked}
        disabled={disabled}
        id={id}
        onCheckedChange={next => onChange(next === true)}
      />
      <FieldCopy description={description} label={label} />
    </label>
  )
}

function ToolRequirements({
  value,
  disabled,
  onChange
}: {
  value: WisdomToolRequirement[]
  disabled?: boolean
  onChange: (value: WisdomToolRequirement[]) => void
}) {
  const id = useId()

  return (
    <div className="space-y-3">
      <FieldCopy
        description="Only tools explicitly needed by this skill belong here. Hermes never copies the full enabled inventory."
        label="Tools"
      />
      {value.map((tool, index) => (
        <div className="grid gap-3 border-t border-(--ui-stroke-tertiary) pt-3 sm:grid-cols-2" key={index}>
          <label className="text-[0.65rem]" htmlFor={`${id}-tool-${index}`}>
            Tool name
            <Input
              className="mt-1 w-full"
              disabled={disabled}
              id={`${id}-tool-${index}`}
              maxLength={512}
              onChange={event =>
                onChange(value.map((item, itemIndex) => (itemIndex === index ? { ...item, name: event.target.value } : item)))
              }
              size="sm"
              value={tool.name}
            />
          </label>
          <label className="text-[0.65rem]" htmlFor={`${id}-tool-version-${index}`}>
            Minimum version (optional)
            <Input
              className="mt-1 w-full"
              disabled={disabled}
              id={`${id}-tool-version-${index}`}
              maxLength={512}
              onChange={event =>
                onChange(
                  value.map((item, itemIndex) =>
                    itemIndex === index ? { ...item, minimum_version: event.target.value || null } : item
                  )
                )
              }
              size="sm"
              value={tool.minimum_version ?? ''}
            />
          </label>
          <BooleanField
            checked={tool.requires_admin}
            description="A teammate must ask an administrator to enable this tool."
            disabled={disabled}
            id={`${id}-tool-admin-${index}`}
            label="Administrator action required"
            onChange={requires_admin =>
              onChange(value.map((item, itemIndex) => (itemIndex === index ? { ...item, requires_admin } : item)))
            }
          />
          <div className="flex items-end justify-between gap-3 text-[0.62rem] text-muted-foreground">
            <span>Automatic installation is always off.</span>
            <Button
              disabled={disabled}
              onClick={() => onChange(value.filter((_, itemIndex) => itemIndex !== index))}
              size="xs"
              variant="text"
            >
              Remove
            </Button>
          </div>
        </div>
      ))}
      <Button
        disabled={disabled || value.length >= 64}
        onClick={() =>
          onChange([...value, { name: '', minimum_version: null, auto_install: false, requires_admin: false }])
        }
        size="xs"
        variant="outline"
      >
        Add tool
      </Button>
    </div>
  )
}

function PluginRequirements({
  value,
  disabled,
  onChange
}: {
  value: WisdomPluginRequirement[]
  disabled?: boolean
  onChange: (value: WisdomPluginRequirement[]) => void
}) {
  const id = useId()

  return (
    <div className="space-y-3">
      <FieldCopy
        description="New drafts include only plugins explicitly declared by the skill, not every installed plugin."
        label="Plugins"
      />
      {value.map((plugin, index) => (
        <div className="grid gap-3 border-t border-(--ui-stroke-tertiary) pt-3 sm:grid-cols-2" key={index}>
          <label className="text-[0.65rem]" htmlFor={`${id}-plugin-${index}`}>
            Plugin ID
            <Input
              className="mt-1 w-full"
              disabled={disabled}
              id={`${id}-plugin-${index}`}
              maxLength={512}
              onChange={event =>
                onChange(value.map((item, itemIndex) => (itemIndex === index ? { ...item, id: event.target.value } : item)))
              }
              size="sm"
              value={plugin.id}
            />
          </label>
          <label className="text-[0.65rem]" htmlFor={`${id}-plugin-version-${index}`}>
            Minimum version (optional)
            <Input
              className="mt-1 w-full"
              disabled={disabled}
              id={`${id}-plugin-version-${index}`}
              maxLength={512}
              onChange={event =>
                onChange(
                  value.map((item, itemIndex) =>
                    itemIndex === index ? { ...item, minimum_version: event.target.value || null } : item
                  )
                )
              }
              size="sm"
              value={plugin.minimum_version ?? ''}
            />
          </label>
          <BooleanField
            checked={plugin.required}
            description="Without this plugin, the skill cannot provide its complete behavior."
            disabled={disabled}
            id={`${id}-plugin-required-${index}`}
            label="Required"
            onChange={required =>
              onChange(value.map((item, itemIndex) => (itemIndex === index ? { ...item, required } : item)))
            }
          />
          <div className="flex items-end justify-end">
            <Button
              disabled={disabled}
              onClick={() => onChange(value.filter((_, itemIndex) => itemIndex !== index))}
              size="xs"
              variant="text"
            >
              Remove
            </Button>
          </div>
        </div>
      ))}
      <Button
        disabled={disabled || value.length >= 64}
        onClick={() => onChange([...value, { id: '', minimum_version: null, required: true }])}
        size="xs"
        variant="outline"
      >
        Add plugin
      </Button>
    </div>
  )
}

export function WisdomSystemSpecificationEditor({
  value,
  disabled = false,
  onChange
}: SpecificationEditorProps) {
  const id = useId()

  const update = <K extends keyof WisdomSystemSpecification>(key: K, next: WisdomSystemSpecification[K]) =>
    onChange({ ...value, [key]: next })

  return (
    <div className="space-y-5">
      <p className="border-l-2 border-(--ui-stroke-secondary) pl-3 text-[0.65rem] leading-4 text-muted-foreground">
        Hermes pre-fills new drafts from this device and requirements explicitly recorded on the skill. Selected
        platforms and architectures restrict where teammates can install it; clear every selection only when the skill
        is known to work everywhere.
      </p>

      <fieldset className="grid gap-5 border-t border-(--ui-stroke-tertiary) pt-4 sm:grid-cols-2">
        <legend className="pr-3 text-[0.68rem] font-medium uppercase tracking-wide">Compatibility targets</legend>
        <label className="text-[0.68rem]" htmlFor={`${id}-hermes-version`}>
          Minimum Hermes version
          <p className="mt-0.5 text-[0.62rem] text-muted-foreground">Older Hermes installations will be blocked.</p>
          <Input
            className="mt-2 w-full"
            disabled={disabled}
            id={`${id}-hermes-version`}
            maxLength={512}
            onChange={event => update('hermes', { minimum_version: event.target.value })}
            size="sm"
            value={value.hermes.minimum_version}
          />
        </label>
        <label className="text-[0.68rem]" htmlFor={`${id}-context-window`}>
          Minimum model context window
          <p className="mt-0.5 text-[0.62rem] text-muted-foreground">Leave blank when no minimum is required.</p>
          <Input
            className="mt-2 w-full"
            disabled={disabled}
            id={`${id}-context-window`}
            min={1}
            onChange={event =>
              update('model', {
                ...value.model,
                minimum_context_window: event.target.value === '' ? null : Number(event.target.value)
              })
            }
            size="sm"
            step={1}
            type="number"
            value={value.model.minimum_context_window ?? ''}
          />
        </label>
        <PresetMultiSelect
          description="Checked systems are the allowed install targets."
          disabled={disabled}
          label="Platforms"
          onChange={platforms => update('platforms', platforms)}
          options={PLATFORM_OPTIONS}
          value={value.platforms}
        />
        <PresetMultiSelect
          description="Checked processors are the allowed install targets."
          disabled={disabled}
          label="Architectures"
          onChange={architectures => update('architectures', architectures)}
          options={ARCHITECTURE_OPTIONS}
          value={value.architectures}
        />
        <StringListField
          description="Capabilities the active model must provide, such as vision."
          disabled={disabled}
          itemLabel="model capability"
          label="Model capabilities"
          onChange={capabilities => update('model', { ...value.model, capabilities })}
          placeholder="vision"
          value={value.model.capabilities}
        />
        <StringListField
          description="Physical hardware requirements, such as a GPU."
          disabled={disabled}
          itemLabel="hardware requirement"
          label="Hardware"
          onChange={hardware => update('hardware', hardware)}
          placeholder="gpu"
          value={value.hardware}
        />
      </fieldset>

      <fieldset className="border-t border-(--ui-stroke-tertiary) pt-4">
        <legend className="pr-3 text-[0.68rem] font-medium uppercase tracking-wide">Runtime access</legend>
        <p className="mb-4 text-[0.62rem] leading-4 text-muted-foreground">
          These requirements are checked before installation; they do not grant permissions by themselves.
        </p>
        <div className="grid gap-4 sm:grid-cols-2">
          <BooleanField
            checked={value.runtime.shell}
            description="The skill needs to run terminal commands."
            disabled={disabled}
            id={`${id}-shell`}
            label="Shell commands"
            onChange={shell => update('runtime', { ...value.runtime, shell })}
          />
          <BooleanField
            checked={value.runtime.browser}
            description="The skill needs an interactive browser."
            disabled={disabled}
            id={`${id}-browser`}
            label="Browser control"
            onChange={browser => update('runtime', { ...value.runtime, browser })}
          />
          <BooleanField
            checked={value.runtime.code}
            description="The skill needs a code-execution environment."
            disabled={disabled}
            id={`${id}-code`}
            label="Code execution"
            onChange={code => update('runtime', { ...value.runtime, code })}
          />
          <BooleanField
            checked={value.runtime.sandbox}
            description="The skill expects an isolated sandbox."
            disabled={disabled}
            id={`${id}-sandbox`}
            label="Sandbox"
            onChange={sandbox => update('runtime', { ...value.runtime, sandbox })}
          />
        </div>
      </fieldset>

      <fieldset className="grid gap-5 border-t border-(--ui-stroke-tertiary) pt-4 lg:grid-cols-2">
        <legend className="pr-3 text-[0.68rem] font-medium uppercase tracking-wide">Tools and plugins</legend>
        <ToolRequirements disabled={disabled} onChange={tools => update('tools', tools)} value={value.tools} />
        <PluginRequirements disabled={disabled} onChange={plugins => update('plugins', plugins)} value={value.plugins} />
      </fieldset>

      <fieldset className="grid gap-5 border-t border-(--ui-stroke-tertiary) pt-4 sm:grid-cols-2">
        <legend className="pr-3 text-[0.68rem] font-medium uppercase tracking-wide">Connections and files</legend>
        <StringListField
          description="Credential names that must already be configured."
          disabled={disabled}
          itemLabel="credential"
          label="Credentials"
          onChange={credentials => update('credentials', credentials)}
          value={value.credentials}
        />
        <StringListField
          description="External connections that must already be available."
          disabled={disabled}
          itemLabel="connection"
          label="Connections"
          onChange={connections => update('connections', connections)}
          value={value.connections}
        />
        <StringListField
          description="Files or directories the skill must read."
          disabled={disabled}
          itemLabel="read path"
          label="Filesystem read access"
          onChange={read => update('filesystem', { ...value.filesystem, read })}
          value={value.filesystem.read}
        />
        <StringListField
          description="Files or directories the skill must write."
          disabled={disabled}
          itemLabel="write path"
          label="Filesystem write access"
          onChange={write => update('filesystem', { ...value.filesystem, write })}
          value={value.filesystem.write}
        />
        <StringListField
          description="Network destinations the skill must contact."
          disabled={disabled}
          itemLabel="network destination"
          label="Network destinations"
          onChange={destinations => update('network', { destinations })}
          value={value.network.destinations}
        />
        <StringListField
          description="Important constraints teammates should understand before installing."
          disabled={disabled}
          itemLabel="known limitation"
          label="Known limitations"
          onChange={known_limitations => update('known_limitations', known_limitations)}
          value={value.known_limitations}
        />
      </fieldset>
    </div>
  )
}

export function WisdomManifestEditor({ value, disabled = false, onChange }: ManifestEditorProps) {
  let manifest: WisdomManifestV1

  try {
    manifest = parseWisdomManifest(value)
  } catch (reason) {
    return (
      <div className="mt-3 text-[0.68rem] text-destructive" role="alert">
        This manifest cannot be edited as a form: {reason instanceof Error ? reason.message : String(reason)}
      </div>
    )
  }

  const update = (next: typeof manifest) => onChange(`${JSON.stringify(next)}\n`)
  const validationError = wisdomManifestValidationError(value)

  return (
    <div className="mt-3 space-y-4">
      <div className="grid gap-3 sm:grid-cols-[1fr_auto]">
        <label className="text-[0.68rem]" htmlFor="desktop-wisdom-manifest-name">
          Skill package name
          <Input
            className="mt-1 w-full"
            disabled={disabled}
            id="desktop-wisdom-manifest-name"
            maxLength={512}
            onChange={event => update({ ...manifest, name: event.target.value })}
            size="sm"
            value={manifest.name}
          />
        </label>
        <div className="self-end pb-1 text-[0.62rem] text-muted-foreground">Schema 1 · fixed by V1</div>
      </div>
      <WisdomSystemSpecificationEditor
        disabled={disabled}
        onChange={requirements => update({ ...manifest, requirements })}
        value={manifest.requirements}
      />
      {validationError && (
        <div className="text-[0.68rem] text-destructive" role="alert">
          {validationError}
        </div>
      )}
    </div>
  )
}
