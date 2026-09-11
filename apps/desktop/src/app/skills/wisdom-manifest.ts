export interface WisdomToolRequirement {
  name: string
  minimum_version: null | string
  auto_install: false
  requires_admin: boolean
}

export interface WisdomPluginRequirement {
  id: string
  minimum_version: null | string
  required: boolean
}

export interface WisdomSystemSpecification extends Record<string, unknown> {
  hermes: { minimum_version: string }
  platforms: string[]
  architectures: string[]
  model: { capabilities: string[]; minimum_context_window: null | number }
  tools: WisdomToolRequirement[]
  plugins: WisdomPluginRequirement[]
  credentials: string[]
  connections: string[]
  filesystem: { read: string[]; write: string[] }
  network: { destinations: string[] }
  runtime: { shell: boolean; browser: boolean; code: boolean; sandbox: boolean }
  hardware: string[]
  known_limitations: string[]
}

export interface WisdomManifestV1 {
  schema_version: 1
  name: string
  requirements: WisdomSystemSpecification
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (!isRecord(value)) {throw new Error(`${label} must be an object`)}

  return value
}

function text(value: unknown, label: string): string {
  if (typeof value !== 'string') {throw new Error(`${label} must be text`)}

  return value
}

function nullableText(value: unknown, label: string): null | string {
  if (value === null) {return null}

  return text(value, label)
}

function boolean(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') {throw new Error(`${label} must be true or false`)}

  return value
}

function textList(value: unknown, label: string): string[] {
  if (!Array.isArray(value) || value.some(item => typeof item !== 'string')) {
    throw new Error(`${label} must be a list of text values`)
  }

  return value
}

function exactKeys(value: Record<string, unknown>, keys: string[], label: string): void {
  const expected = new Set(keys)
  const unexpected = Object.keys(value).filter(key => !expected.has(key))
  const missing = keys.filter(key => !(key in value))

  if (unexpected.length) {throw new Error(`${label} contains unsupported fields: ${unexpected.join(', ')}`)}

  if (missing.length) {throw new Error(`${label} is missing fields: ${missing.join(', ')}`)}
}

export function parseWisdomSystemSpecification(value: unknown): WisdomSystemSpecification {
  const specification = record(value, 'System Specification')
  exactKeys(
    specification,
    [
      'hermes',
      'platforms',
      'architectures',
      'model',
      'tools',
      'plugins',
      'credentials',
      'connections',
      'filesystem',
      'network',
      'runtime',
      'hardware',
      'known_limitations'
    ],
    'System Specification'
  )
  const hermes = record(specification.hermes, 'Hermes requirement')
  exactKeys(hermes, ['minimum_version'], 'Hermes requirement')
  const model = record(specification.model, 'Model requirement')
  exactKeys(model, ['capabilities', 'minimum_context_window'], 'Model requirement')

  if (model.minimum_context_window !== null && typeof model.minimum_context_window !== 'number') {
    throw new Error('Minimum context window must be a number or blank')
  }

  const filesystem = record(specification.filesystem, 'Filesystem requirement')
  exactKeys(filesystem, ['read', 'write'], 'Filesystem requirement')
  const network = record(specification.network, 'Network requirement')
  exactKeys(network, ['destinations'], 'Network requirement')
  const runtime = record(specification.runtime, 'Runtime requirement')
  exactKeys(runtime, ['shell', 'browser', 'code', 'sandbox'], 'Runtime requirement')

  if (!Array.isArray(specification.tools)) {throw new Error('Tools must be a list')}

  const tools = specification.tools.map((raw, index) => {
    const tool = record(raw, `Tool ${index + 1}`)
    exactKeys(tool, ['name', 'minimum_version', 'auto_install', 'requires_admin'], `Tool ${index + 1}`)

    if (tool.auto_install !== false) {throw new Error(`Tool ${index + 1} cannot request automatic installation`)}

    return {
      name: text(tool.name, `Tool ${index + 1} name`),
      minimum_version: nullableText(tool.minimum_version, `Tool ${index + 1} minimum version`),
      auto_install: false as const,
      requires_admin: boolean(tool.requires_admin, `Tool ${index + 1} administrator requirement`)
    }
  })

  if (!Array.isArray(specification.plugins)) {throw new Error('Plugins must be a list')}

  const plugins = specification.plugins.map((raw, index) => {
    const plugin = record(raw, `Plugin ${index + 1}`)
    exactKeys(plugin, ['id', 'minimum_version', 'required'], `Plugin ${index + 1}`)

    return {
      id: text(plugin.id, `Plugin ${index + 1} ID`),
      minimum_version: nullableText(plugin.minimum_version, `Plugin ${index + 1} minimum version`),
      required: boolean(plugin.required, `Plugin ${index + 1} requirement`)
    }
  })

  return {
    hermes: { minimum_version: text(hermes.minimum_version, 'Minimum Hermes version') },
    platforms: textList(specification.platforms, 'Platforms'),
    architectures: textList(specification.architectures, 'Architectures'),
    model: {
      capabilities: textList(model.capabilities, 'Model capabilities'),
      minimum_context_window: model.minimum_context_window === null ? null : Number(model.minimum_context_window)
    },
    tools,
    plugins,
    credentials: textList(specification.credentials, 'Credentials'),
    connections: textList(specification.connections, 'Connections'),
    filesystem: {
      read: textList(filesystem.read, 'Readable paths'),
      write: textList(filesystem.write, 'Writable paths')
    },
    network: { destinations: textList(network.destinations, 'Network destinations') },
    runtime: {
      shell: boolean(runtime.shell, 'Shell requirement'),
      browser: boolean(runtime.browser, 'Browser requirement'),
      code: boolean(runtime.code, 'Code execution requirement'),
      sandbox: boolean(runtime.sandbox, 'Sandbox requirement')
    },
    hardware: textList(specification.hardware, 'Hardware'),
    known_limitations: textList(specification.known_limitations, 'Known limitations')
  }
}

export function parseWisdomManifest(value: string): WisdomManifestV1 {
  const manifest = record(JSON.parse(value) as unknown, 'Manifest')
  exactKeys(manifest, ['schema_version', 'name', 'requirements'], 'Manifest')

  if (manifest.schema_version !== 1) {throw new Error('Only manifest schema version 1 is supported')}

  return {
    schema_version: 1,
    name: text(manifest.name, 'Skill name'),
    requirements: parseWisdomSystemSpecification(manifest.requirements)
  }
}

function boundedTextError(value: string, label: string): null | string {
  if (!value.trim()) {return `${label} is required.`}

  if (new TextEncoder().encode(value).length > 512) {return `${label} must be 512 UTF-8 bytes or fewer.`}

  return null
}

function listError(values: string[], label: string): null | string {
  if (values.length > 64) {return `${label} can contain at most 64 entries.`}

  for (const value of values) {
    const error = boundedTextError(value, `${label} entry`)

    if (error) {return error}
  }

  return null
}

export function wisdomSystemSpecificationValidationError(value: WisdomSystemSpecification): null | string {
  const errors: Array<null | string> = [
    boundedTextError(value.hermes.minimum_version, 'Minimum Hermes version'),
    value.model.minimum_context_window !== null &&
    (!Number.isSafeInteger(value.model.minimum_context_window) || value.model.minimum_context_window < 1)
      ? 'Minimum context window must be blank or a positive whole number.'
      : null,
    listError(value.platforms, 'Platforms'),
    listError(value.architectures, 'Architectures'),
    listError(value.model.capabilities, 'Model capabilities'),
    listError(value.credentials, 'Credentials'),
    listError(value.connections, 'Connections'),
    listError(value.filesystem.read, 'Readable paths'),
    listError(value.filesystem.write, 'Writable paths'),
    listError(value.network.destinations, 'Network destinations'),
    listError(value.hardware, 'Hardware requirements'),
    listError(value.known_limitations, 'Known limitations'),
    value.tools.length > 64 ? 'Tools can contain at most 64 entries.' : null,
    value.plugins.length > 64 ? 'Plugins can contain at most 64 entries.' : null
  ]

  for (const [index, tool] of value.tools.entries()) {
    errors.push(boundedTextError(tool.name, `Tool ${index + 1} name`))

    if (tool.minimum_version !== null) {errors.push(boundedTextError(tool.minimum_version, `Tool ${index + 1} version`))}
  }

  for (const [index, plugin] of value.plugins.entries()) {
    errors.push(boundedTextError(plugin.id, `Plugin ${index + 1} ID`))

    if (plugin.minimum_version !== null) {
      errors.push(boundedTextError(plugin.minimum_version, `Plugin ${index + 1} version`))
    }
  }

  return errors.find((error): error is string => Boolean(error)) ?? null
}

export function wisdomManifestValidationError(value: string): null | string {
  try {
    const manifest = parseWisdomManifest(value)

    return boundedTextError(manifest.name, 'Skill name') ?? wisdomSystemSpecificationValidationError(manifest.requirements)
  } catch (reason) {
    return reason instanceof Error ? reason.message : String(reason)
  }
}
