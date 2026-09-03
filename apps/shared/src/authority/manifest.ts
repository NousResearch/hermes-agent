type JsonRecord = Record<string, unknown>;

const KNOWN_DOMAINS = deepFreeze([
  "github.operation",
  "sandbox.execution",
  "child_process.operation",
  "gateway.mutation",
] as const);

const KNOWN_ACTOR_CLASSES = deepFreeze([
  "human",
  "github_app",
  "workflow",
  "automation",
  "external_bot",
] as const);

const TOP_LEVEL_KEYS = deepFreeze(["schema_version", "policy_version", "domains"] as const);
const DOMAIN_KEYS = deepFreeze(["sinks", "operation_classes"] as const);
const SINK_KEYS = deepFreeze(["broker", "direct_symbols"] as const);
const OPERATION_KEYS = deepFreeze([
  "sink_class",
  "required_capabilities",
  "allowed_actor_classes",
  "allowed_resource_states",
] as const);
const REQUEST_KEYS = deepFreeze([
  "domain",
  "operation_class",
  "actor_class",
  "resource_state",
  "capabilities",
] as const);
const CAPABILITY_GRANT_KEYS = deepFreeze([
  "capability",
  "granted",
  "source",
  "generation",
] as const);

const CANONICAL_MANIFEST_JSON = `{
  "schema_version": 1,
  "policy_version": "2026.08.25-v2",
  "domains": {
    "github.operation": {
      "sinks": {
        "github.issue.metadata.write": {
          "broker": "githubMutationBroker",
          "direct_symbols": [
            "issues.addLabels",
            "issues.removeLabel",
            "issues.update"
          ]
        },
        "github.comment.write": {
          "broker": "githubMutationBroker",
          "direct_symbols": [
            "issues.createComment",
            "pulls.createReview",
            "pulls.createReviewComment"
          ]
        },
        "github.contents.write": {
          "broker": "githubMutationBroker",
          "direct_symbols": [
            "repos.createOrUpdateFileContents",
            "repos.deleteFile"
          ]
        },
        "github.gitdata.write": {
          "broker": "githubMutationBroker",
          "direct_symbols": [
            "git.createBlob",
            "git.createCommit",
            "git.createRef",
            "git.createTree",
            "git.updateRef"
          ]
        },
        "github.pull_request.create": {
          "broker": "githubMutationBroker",
          "direct_symbols": [
            "pulls.create"
          ]
        },
        "github.actions.dispatch": {
          "broker": "githubMutationBroker",
          "direct_symbols": [
            "actions.createWorkflowDispatch"
          ]
        }
      },
      "operation_classes": {
        "github.issue.metadata.write": {
          "sink_class": "github.issue.metadata.write",
          "required_capabilities": [
            "issues:metadata:write"
          ],
          "allowed_actor_classes": [
            "human",
            "github_app",
            "workflow",
            "automation"
          ],
          "allowed_resource_states": [
            "open",
            "closed"
          ]
        },
        "github.comment.write": {
          "sink_class": "github.comment.write",
          "required_capabilities": [
            "comments:write"
          ],
          "allowed_actor_classes": [
            "human",
            "github_app",
            "workflow",
            "automation"
          ],
          "allowed_resource_states": [
            "open",
            "closed"
          ]
        },
        "github.contents.write": {
          "sink_class": "github.contents.write",
          "required_capabilities": [
            "contents:write"
          ],
          "allowed_actor_classes": [
            "human",
            "github_app",
            "workflow",
            "automation"
          ],
          "allowed_resource_states": [
            "current"
          ]
        },
        "github.gitdata.write": {
          "sink_class": "github.gitdata.write",
          "required_capabilities": [
            "git_objects:write",
            "refs:write"
          ],
          "allowed_actor_classes": [
            "human",
            "github_app",
            "workflow",
            "automation"
          ],
          "allowed_resource_states": [
            "current"
          ]
        },
        "github.pull_request.create": {
          "sink_class": "github.pull_request.create",
          "required_capabilities": [
            "pull_requests:create"
          ],
          "allowed_actor_classes": [
            "human",
            "github_app",
            "workflow",
            "automation"
          ],
          "allowed_resource_states": [
            "current"
          ]
        },
        "github.actions.dispatch": {
          "sink_class": "github.actions.dispatch",
          "required_capabilities": [
            "actions:dispatch"
          ],
          "allowed_actor_classes": [
            "human",
            "github_app",
            "workflow",
            "automation"
          ],
          "allowed_resource_states": [
            "current"
          ]
        }
      }
    }
  }
}
`;

const CANONICAL_MANIFEST_SHA256 =
  "d427e3cd580f97a18103cbb73c04dc0baf2565f8509bcd298ee7aff049588d8e";

export class ManifestValidationError extends Error {
  constructor(message: string) {
    super(message);

    this.name = "ManifestValidationError";
  }
}

export class AdmissionRequestValidationError extends Error {
  constructor(message: string) {
    super(message);

    this.name = "AdmissionRequestValidationError";
  }
}

export type CapabilityGrant = Readonly<{
  capability: string;
  granted: boolean;
  source: string;
  generation: string;
}>;

export type CapabilityProof = Readonly<{
  capability: string;
  source: string;
  generation: string;
}>;

export type AdmissionRequest = Readonly<{
  domain: string;
  operation_class: string;
  actor_class: string;
  resource_state: string;
  capabilities: readonly CapabilityGrant[];
}>;

export type AuthorityDecision = Readonly<{
  allowed: boolean;
  operation_class: string;
  reason_code: string;
  matched_rule: string | null;
  capability_proofs: readonly CapabilityProof[];
  policy_version: string;
}>;

export type CompiledSink = Readonly<{
  broker: string;
  direct_symbols: readonly string[];
}>;

export type CompiledOperation = Readonly<{
  sink_class: string;
  required_capabilities: readonly string[];
  allowed_actor_classes: readonly string[];
  allowed_resource_states: readonly string[];
}>;

export type CompiledDomain = Readonly<{
  sinks: Readonly<Record<string, CompiledSink>>;
  operations: Readonly<Record<string, CompiledOperation>>;
}>;

export type CompiledManifest = Readonly<{
  schemaVersion: 1;
  policyVersion: string;
  domains: Readonly<Record<string, CompiledDomain>>;
}>;

export type AuthorityManifestArtifact = Readonly<{
  manifest_bytes: readonly number[];
  manifest_sha256: string;
  manifest: CompiledManifest;
}>;

function deepFreeze<T>(value: T): T {
  if (typeof value !== "object" || value === null || Object.isFrozen(value)) {
    return value;
  }

  for (const child of Object.values(value as JsonRecord)) {
    deepFreeze(child);
  }
  return Object.freeze(value);
}

function hasOwn(value: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function asRecord(value: unknown, where: string): JsonRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new ManifestValidationError(`${where} must be an object`);
  }
  return value as JsonRecord;
}

function requireExactKeys(
  value: JsonRecord,
  expected: readonly string[],
  where: string,
  ErrorType: typeof ManifestValidationError = ManifestValidationError,
): void {
  const actual = Object.keys(value);
  const unknown = actual.filter((key) => !expected.includes(key)).sort();
  const missing = expected.filter((key) => !hasOwn(value, key)).sort();
  if (unknown.length > 0 || missing.length > 0) {
    const details = [
      unknown.length > 0 ? `unknown=${JSON.stringify(unknown)}` : null,
      missing.length > 0 ? `missing=${JSON.stringify(missing)}` : null,
    ]
      .filter(Boolean)
      .join(", ");
    throw new ErrorType(`${where} must be closed (${details})`);
  }
}

function requireNonemptyString(value: unknown, where: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new ManifestValidationError(`${where} must be a non-empty string`);
  }
  return value;
}

function requireUniqueStrings(
  value: unknown,
  where: string,
  options: { nonempty?: boolean } = {},
): string[] {
  const nonempty = options.nonempty ?? true;
  if (!Array.isArray(value)) {
    throw new ManifestValidationError(`${where} must be an array`);
  }
  if (nonempty && value.length === 0) {
    throw new ManifestValidationError(`${where} must not be empty`);
  }
  if (value.some((item) => typeof item !== "string" || item.trim().length === 0)) {
    throw new ManifestValidationError(`${where} must contain only non-empty strings`);
  }
  const strings = value as string[];
  if (new Set(strings).size !== strings.length) {
    throw new ManifestValidationError(`${where} must not contain duplicates`);
  }
  return [...strings];
}

export function compileAuthorityManifest(rawValue: unknown): CompiledManifest {
  const raw = asRecord(rawValue, "manifest");
  requireExactKeys(raw, TOP_LEVEL_KEYS, "manifest");
  if (raw.schema_version !== 1) {
    throw new ManifestValidationError("schema_version must equal 1");
  }
  const policyVersion = requireNonemptyString(raw.policy_version, "policy_version");
  const rawDomains = asRecord(raw.domains, "domains");
  if (Object.keys(rawDomains).length === 0) {
    throw new ManifestValidationError("domains must be a non-empty object");
  }

  const unknownDomains = Object.keys(rawDomains)
    .filter((domain) => !KNOWN_DOMAINS.includes(domain as (typeof KNOWN_DOMAINS)[number]))
    .sort();
  if (unknownDomains.length > 0) {
    throw new ManifestValidationError(`unknown domains: ${JSON.stringify(unknownDomains)}`);
  }

  const domains = Object.create(null) as Record<string, CompiledDomain>;
  for (const domainName of Object.keys(rawDomains).sort()) {
    const rawDomain = asRecord(rawDomains[domainName], `domains.${domainName}`);
    requireExactKeys(rawDomain, DOMAIN_KEYS, `domains.${domainName}`);
    const rawSinks = asRecord(rawDomain.sinks, `domains.${domainName}.sinks`);
    const rawOperations = asRecord(
      rawDomain.operation_classes,
      `domains.${domainName}.operation_classes`,
    );
    if (Object.keys(rawSinks).length === 0) {
      throw new ManifestValidationError(`domains.${domainName}.sinks must be non-empty`);
    }
    if (Object.keys(rawOperations).length === 0) {
      throw new ManifestValidationError(
        `domains.${domainName}.operation_classes must be non-empty`,
      );
    }

    const sinks = Object.create(null) as Record<string, CompiledSink>;
    for (const [sinkName, sinkValue] of Object.entries(rawSinks)) {
      requireNonemptyString(sinkName, `domains.${domainName}.sink name`);
      const sinkSpec = asRecord(sinkValue, `sink ${sinkName}`);
      requireExactKeys(sinkSpec, SINK_KEYS, `sink ${sinkName}`);
      const broker = requireNonemptyString(sinkSpec.broker, `sink ${sinkName}.broker`);
      const directSymbols = requireUniqueStrings(
        sinkSpec.direct_symbols,
        `sink ${sinkName}.direct_symbols`,
        { nonempty: false },
      ).sort();
      sinks[sinkName] = deepFreeze({ broker, direct_symbols: directSymbols });
    }

    const operations = Object.create(null) as Record<string, CompiledOperation>;
    for (const [operationName, operationValue] of Object.entries(rawOperations)) {
      requireNonemptyString(operationName, `domains.${domainName}.operation class name`);
      const operationSpec = asRecord(operationValue, `operation ${operationName}`);
      requireExactKeys(operationSpec, OPERATION_KEYS, `operation ${operationName}`);

      const sinkClass = requireNonemptyString(
        operationSpec.sink_class,
        `operation ${operationName}.sink_class`,
      );
      if (!hasOwn(sinks, sinkClass)) {
        throw new ManifestValidationError(
          `operation ${operationName} references unregistered sink ${sinkClass}`,
        );
      }

      const requiredCapabilities = requireUniqueStrings(
        operationSpec.required_capabilities,
        `operation ${operationName}.required_capabilities`,
      );
      if (requiredCapabilities.some((capability) => capability.includes("*"))) {
        throw new ManifestValidationError(
          `operation ${operationName} contains an ambient wildcard capability`,
        );
      }

      const actorClasses = requireUniqueStrings(
        operationSpec.allowed_actor_classes,
        `operation ${operationName}.allowed_actor_classes`,
      );
      const unknownActors = actorClasses.filter(
        (actor) =>
          !KNOWN_ACTOR_CLASSES.includes(actor as (typeof KNOWN_ACTOR_CLASSES)[number]),
      );
      if (unknownActors.length > 0) {
        throw new ManifestValidationError(
          `operation ${operationName} has unknown actor classes ${JSON.stringify(unknownActors)}`,
        );
      }

      const resourceStates = requireUniqueStrings(
        operationSpec.allowed_resource_states,
        `operation ${operationName}.allowed_resource_states`,
      );
      operations[operationName] = deepFreeze({
        sink_class: sinkClass,
        required_capabilities: requiredCapabilities,
        allowed_actor_classes: actorClasses.sort(),
        allowed_resource_states: resourceStates.sort(),
      });
    }

    domains[domainName] = deepFreeze({ sinks, operations });
  }

  return deepFreeze({ schemaVersion: 1, policyVersion, domains });
}

export function parseAdmissionRequest(rawValue: unknown): AdmissionRequest {
  let raw: JsonRecord;
  try {
    raw = asRecord(rawValue, "admission request");
  } catch (error) {
    throw new AdmissionRequestValidationError((error as Error).message);
  }
  requireExactKeys(
    raw,
    REQUEST_KEYS,
    "admission request",
    AdmissionRequestValidationError,
  );

  const fields = Object.create(null) as Record<string, string>;
  for (const field of ["domain", "operation_class", "actor_class", "resource_state"] as const) {
    const value = raw[field];
    if (typeof value !== "string") {
      throw new AdmissionRequestValidationError(`${field} must be a string`);
    }
    fields[field] = value;
  }

  if (!Array.isArray(raw.capabilities)) {
    throw new AdmissionRequestValidationError("capabilities must be an array");
  }
  const capabilities = raw.capabilities.map((value, index): CapabilityGrant => {
    let grant: JsonRecord;
    try {
      grant = asRecord(value, `capabilities[${index}]`);
    } catch (error) {
      throw new AdmissionRequestValidationError((error as Error).message);
    }
    requireExactKeys(
      grant,
      CAPABILITY_GRANT_KEYS,
      `capabilities[${index}]`,
      AdmissionRequestValidationError,
    );
    if (typeof grant.capability !== "string") {
      throw new AdmissionRequestValidationError(
        `capabilities[${index}].capability must be a string`,
      );
    }
    if (typeof grant.granted !== "boolean") {
      throw new AdmissionRequestValidationError(
        `capabilities[${index}].granted must be a boolean`,
      );
    }
    if (typeof grant.source !== "string") {
      throw new AdmissionRequestValidationError(
        `capabilities[${index}].source must be a string`,
      );
    }
    if (typeof grant.generation !== "string") {
      throw new AdmissionRequestValidationError(
        `capabilities[${index}].generation must be a string`,
      );
    }
    return deepFreeze({
      capability: grant.capability,
      granted: grant.granted,
      source: grant.source,
      generation: grant.generation,
    });
  });

  return deepFreeze({
    domain: fields.domain,
    operation_class: fields.operation_class,
    actor_class: fields.actor_class,
    resource_state: fields.resource_state,
    capabilities,
  });
}

function decision(
  manifest: CompiledManifest,
  request: AdmissionRequest,
  reasonCode: string,
  matchedRule: string | null,
  options: {
    allowed?: boolean;
    capabilityProofs?: readonly CapabilityProof[];
  } = {},
): AuthorityDecision {
  return deepFreeze({
    allowed: options.allowed ?? false,
    operation_class: request.operation_class,
    reason_code: reasonCode,
    matched_rule: matchedRule,
    capability_proofs: [...(options.capabilityProofs ?? [])],
    policy_version: manifest.policyVersion,
  });
}

export function evaluateAuthorityOperation(
  manifest: CompiledManifest,
  requestValue: AdmissionRequest,
): AuthorityDecision {
  const request = parseAdmissionRequest(requestValue);
  const domain = manifest.domains[request.domain];
  const operation = domain?.operations[request.operation_class];
  const matchedRule = operation
    ? `${request.domain}.${request.operation_class}`
    : null;

  if (!operation) {
    return decision(manifest, request, "unsupported_operation", null);
  }

  if (!operation.allowed_actor_classes.includes(request.actor_class)) {
    return decision(manifest, request, "actor_forbidden", matchedRule);
  }

  if (!operation.allowed_resource_states.includes(request.resource_state)) {
    return decision(manifest, request, "resource_state_denied", matchedRule);
  }

  const grants = Object.create(null) as Record<string, CapabilityGrant>;
  for (const grant of request.capabilities) {
    if (
      grant.capability.trim().length === 0 ||
      grant.source.trim().length === 0 ||
      grant.generation.trim().length === 0 ||
      hasOwn(grants, grant.capability)
    ) {
      return decision(manifest, request, "invalid_capability_proof", matchedRule);
    }
    grants[grant.capability] = grant;
  }

  if (
    operation.required_capabilities.some(
      (capability) => !hasOwn(grants, capability) || !grants[capability].granted,
    )
  ) {
    return decision(manifest, request, "missing_capability", matchedRule);
  }

  const capabilityProofs = operation.required_capabilities.map((capability) =>
    deepFreeze({
      capability,
      source: grants[capability].source,
      generation: grants[capability].generation,
    }),
  );
  return decision(manifest, request, "allowed", matchedRule, {
    allowed: true,
    capabilityProofs,
  });
}

function buildCanonicalManifestArtifact(): AuthorityManifestArtifact {
  const manifestBytes = deepFreeze(
    Array.from(new TextEncoder().encode(CANONICAL_MANIFEST_JSON)),
  );
  const raw: unknown = JSON.parse(CANONICAL_MANIFEST_JSON);
  return deepFreeze({
    manifest_bytes: manifestBytes,
    manifest_sha256: CANONICAL_MANIFEST_SHA256,
    manifest: compileAuthorityManifest(raw),
  });
}

export const CANONICAL_AUTHORITY_MANIFEST = buildCanonicalManifestArtifact();
