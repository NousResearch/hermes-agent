#!/usr/bin/env python3
from __future__ import annotations

import filecmp
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

SCHEMA_DRAFT = 'https://json-schema.org/draft/2020-12/schema'
COMMAND_VERSION = 'workflow-command-envelope.v1'
BINDING_VERSION = 'workflow-provider-binding.v1'
DELIVERY_VERSION = 'workflow-delivery-status.v1'
EVENT_VERSION = 'workflow-event-envelope.v1'
REJECTION_VERSION = 'workflow-callback-rejection.v1'
PROJECT_WORK_IDENTITY_VERSION = 'workflow-project-work-identity.v1'
PHASE_TASK_IDENTITY_VERSION = 'workflow-phase-task-identity.v1'
MATERIALIZATION_VERSION = 'workflow-materialization-case.v1'
PHASE_KIND = 'story_implementation'
COMBINED_WORKFLOW_NAME = 'bmad-create-and-dev-story'
PROJECT_WORK_IDENTITY_ALGORITHM = 'uri-component-encoded-identity-v1'
PHASE_TASK_IDENTITY_ALGORITHM = 'phase-prefix-project-work-key-phase-kind-v1'
METADATA_FIELDS = {'schemaVersion', 'intendedProducer', 'intendedConsumer', 'owningSubproject'}
BINDING_FORBIDDEN_KEYS = {'profile', 'agent_name', 'agent', 'agent_provider'}
COMMAND_FORBIDDEN_TEXT_KEYS = {
    'displayText',
    'display_text',
    'humanText',
    'humanReadable',
    'message',
    'prose',
    'stderr',
    'stderrText',
    'stdout',
    'stdoutText',
}
DELIVERY_STATUSES = {
    'healthy',
    'delayed',
    'retrying',
    'failed',
    'duplicated',
    'terminal-failure',
    'reconciliation-pending',
}
DELIVERY_ALIASES = {'waiting-for-reconciliation', 'waiting_for_reconciliation'}
EVENT_TYPES = {
    'workflow.run.started',
    'workflow.run.completed',
    'workflow.run.failed',
    'workflow.approval.requested',
    'workflow.delivery.failed',
    'workflow.artifact.recorded',
}
REJECTION_REASONS = {
    'bad-signature',
    'stale-timestamp',
    'duplicate-event-id',
    'wrong-binding',
    'unknown-project',
    'schema-mismatch',
    'wrong-profile-secret',
    'wrong-codebase',
    'unsupported-provider',
}
REJECTION_DIAGNOSTIC_CATEGORIES = {
    'security_rejection',
    'idempotency_rejection',
    'binding_rejection',
    'project_rejection',
    'provider_contract',
    'unsupported_provider',
}
SENSITIVE_KEY_NAMES = {
    'apikey',
    'authorization',
    'privatekey',
    'rawsecret',
    'rawsignature',
    'secret',
    'secretvalue',
    'sharedsecret',
    'signaturevalue',
    'token',
}
HOST_SPECIFIC_PATH_PREFIXES = ('/Users/', '/home/', '/private/', '/tmp/', '/var/folders/')
WINDOWS_ABSOLUTE_PATH_RE = re.compile(r'^[A-Za-z]:[\\/]')
RFC3339_DATETIME_RE = re.compile(
    r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$'
)
REQUIRED_SCHEMA_FILES = [
    'schemas/workflow-command-envelope.schema.json',
    'schemas/workflow-provider-binding.schema.json',
    'schemas/workflow-delivery-status.schema.json',
    'schemas/workflow-event-envelope.schema.json',
    'schemas/project-work-identity.schema.json',
    'schemas/phase-task-identity.schema.json',
    'schemas/materialization-case.schema.json',
]
REQUIRED_COMMAND_EXAMPLES = [
    'start-success.json',
    'status-success.json',
    'approve-success.json',
    'reject-success.json',
    'resume-success.json',
    'retry-success.json',
    'cancel-success.json',
    'binding-create-success.json',
    'binding-status-success.json',
    'binding-rotate-success.json',
    'binding-disable-success.json',
    'error-malformed-request.json',
    'error-timeout.json',
    'error-schema-mismatch.json',
    'error-unexpected-exit.json',
    'error-unexpected-state.json',
]
REQUIRED_BINDING_EXAMPLES = [
    'create-request.json',
    'update-request.json',
    'create-success.json',
    'update-success.json',
    'rotate-success.json',
    'disable-success.json',
    'remove-success.json',
    'status-valid.json',
    'status-missing.json',
    'status-stale.json',
    'status-disabled.json',
    'status-rotated.json',
    'status-conflicting.json',
    'error-malformed-request.json',
]
REQUIRED_DELIVERY_EXAMPLES = [
    'healthy.json',
    'delayed.json',
    'retrying.json',
    'failed.json',
    'duplicated.json',
    'terminal-failure.json',
    'reconciliation-pending.json',
]
REQUIRED_GENERIC_EVENT_EXAMPLES = [
    'workflow-completed.json',
    'workflow-completed-redelivery.json',
    'workflow-failed.json',
    'approval-requested.json',
    'delivery-failed.json',
    'artifact-event.json',
]
REQUIRED_EVENT_EXAMPLES = [
    'workflow-run-started.json',
    'workflow-completed.json',
    'workflow-completed-redelivery.json',
    'workflow-failed.json',
    'approval-requested.json',
    'delivery-failed.json',
    'artifact-event.json',
]
REQUIRED_REJECTION_EXAMPLES = [
    'bad-signature.json',
    'stale-timestamp.json',
    'duplicate-event-id.json',
    'wrong-binding.json',
    'unknown-project.json',
    'schema-mismatch.json',
    'wrong-profile-secret.json',
    'wrong-codebase.json',
    'unsupported-provider.json',
]
REQUIRED_MATERIALIZATION_EXAMPLES = [
    'new-story.json',
    'unchanged-story.json',
    'changed-story.json',
    'missing-sprint-status.json',
    'malformed-sprint-status.json',
    'duplicate-phase-task-prevention.json',
]
DEFERRED_SIGNATURE_POLICY = {
    'algorithm': '${WORKFLOW_EVENT_SIGNATURE_ALGORITHM}',
    'signatureHeader': '${WORKFLOW_EVENT_SIGNATURE_HEADER}',
    'timestampHeader': '${WORKFLOW_EVENT_TIMESTAMP_HEADER}',
    'canonicalization': '${WORKFLOW_EVENT_SIGNATURE_CANONICALIZATION}',
}
NO_MUTATION_FIELDS = {
    'stateMutation',
    'workflowRunMutation',
    'bindingMutation',
    'eventLedgerMutation',
    'projectWorkItemMutation',
    'phaseTaskMutation',
    'hiltGateMutation',
    'workflowReferenceMutation',
    'commentMutation',
    'storyStatusHistoryMutation',
}
PROJECT_WORK_IDENTITY_DERIVED_FROM = ('boundProjectCwd', 'bmadArtifactPath', 'bmadIdentity')
PROJECT_WORK_IDENTITY_DERIVED_FROM_SET = set(PROJECT_WORK_IDENTITY_DERIVED_FROM)
PHASE_TASK_IDENTITY_DERIVED_FROM = ('projectWorkItemIdentityKey', 'phaseKind')
PHASE_TASK_IDENTITY_DERIVED_FROM_SET = set(PHASE_TASK_IDENTITY_DERIVED_FROM)
PROJECT_WORK_IDENTITY_FORBIDDEN_KEYS = {
    'currentKanbanStatus',
    'gateKind',
    'githubPrNumber',
    'hiltGateState',
    'kanbanStatus',
    'phaseKind',
    'phaseTaskIdentity',
    'phaseTaskIdentityKey',
    'providerRunId',
    'targetBmadStatus',
    'verificationResult',
    'workflowEventId',
    'workflowRunId',
    'workflowRunRef',
}
MATERIALIZATION_SUCCESS_CASES = {
    'new-story',
    'unchanged-story',
    'changed-story',
    'duplicate-phase-task-prevention',
}
MATERIALIZATION_FAIL_CLOSED_CASES = {'missing-sprint-status', 'malformed-sprint-status'}
MATERIALIZATION_MUTATIONS = {'create', 'update', 'reuse', 'no-op', 'none'}
MATERIALIZATION_SOURCE_AVAILABILITY = {
    'new-story': 'present',
    'unchanged-story': 'present',
    'changed-story': 'present',
    'duplicate-phase-task-prevention': 'present',
    'missing-sprint-status': 'missing',
    'malformed-sprint-status': 'malformed',
}
MATERIALIZATION_CASE_MUTATIONS = {
    'new-story': ('create', 'create'),
    'unchanged-story': ('no-op', 'reuse'),
    'changed-story': ('update', 'reuse'),
    'duplicate-phase-task-prevention': ('no-op', 'reuse'),
    'missing-sprint-status': ('none', 'none'),
    'malformed-sprint-status': ('none', 'none'),
}
DOWNSTREAM_READINESS_FLAG_FIELDS = {
    'requiresMigrationExpectation',
    'requiresUniquenessExpectation',
    'requiresIdempotencyExpectation',
    'blocksImplementationWhenRelevantExampleMissing',
}
DOWNSTREAM_READINESS_EXPECTATION_FIELDS = {
    'migrationExpectation',
    'uniquenessExpectation',
    'idempotencyExpectation',
}
PAYLOAD_REQUIRED_FIELDS = {
    'workflow.run.started': {'state', 'phase', 'commandCorrelationId', 'startedAt'},
    'workflow.run.completed': {'state', 'result'},
    'workflow.run.failed': {'state', 'failure'},
    'workflow.approval.requested': {'state', 'approval'},
    'workflow.delivery.failed': {
        'deliveryOnly',
        'mutationIntent',
        'deliveryStatus',
        'failedDeliveryId',
        'nextStatus',
        'diagnosticRef',
    },
    'workflow.artifact.recorded': {'artifact'},
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parent_package() -> Path:
    return Path(__file__).resolve().parent


def local_packages(root: Path) -> list[Path]:
    return [
        root / 'archon/_bmad-output/planning-artifacts/contracts/workflow-commander',
        root / 'hermes-agent/_bmad-output/planning-artifacts/contracts/workflow-commander',
    ]


def has_isolated_handoffs(root: Path) -> bool:
    return (root / 'archon').exists() or (root / 'hermes-agent').exists()


def owned_json_files() -> list[str]:
    files = list(REQUIRED_SCHEMA_FILES)
    files.extend(f'examples/providers/archon/commands/{name}' for name in REQUIRED_COMMAND_EXAMPLES)
    files.extend(f'examples/providers/archon/bindings/{name}' for name in REQUIRED_BINDING_EXAMPLES)
    files.extend(f'examples/providers/archon/delivery/{name}' for name in REQUIRED_DELIVERY_EXAMPLES)
    files.extend(f'examples/workflow-events/{name}' for name in REQUIRED_GENERIC_EVENT_EXAMPLES)
    files.extend(f'examples/providers/archon/events/{name}' for name in REQUIRED_EVENT_EXAMPLES)
    files.extend(f'examples/callback-rejections/{name}' for name in REQUIRED_REJECTION_EXAMPLES)
    files.extend(f'examples/materialization/{name}' for name in REQUIRED_MATERIALIZATION_EXAMPLES)
    return files


def owned_contract_files() -> list[str]:
    return ['README.md', *owned_json_files(), 'validate_contracts.py']


def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key {key!r}')
        result[key] = value
    return result


def reject_non_standard_json_constant(value: str) -> None:
    raise ValueError(f'non-standard JSON constant {value!r}')


def load_json(path: Path, errors: list[str]) -> Any | None:
    if not path.exists():
        errors.append(f'missing required file: {path}')
        return None
    if path.is_symlink():
        errors.append(f'file must not be a symlink: {path}')
        return None
    if not path.is_file():
        errors.append(f'required path must be a regular file: {path}')
        return None
    if path.stat().st_size == 0:
        errors.append(f'file must be non-empty: {path}')
        return None
    try:
        text = path.read_text(encoding='utf-8')
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_json_keys,
            parse_constant=reject_non_standard_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        errors.append(f'invalid JSON in {path}: {exc}')
        return None


def pointer(parts: list[str | int]) -> str:
    if not parts:
        return '$'
    return '$' + ''.join(f'[{part}]' if isinstance(part, int) else f'.{part}' for part in parts)


def resolve_ref(root_schema: dict[str, Any], ref: str) -> Any:
    if not ref.startswith('#/'):
        raise ValueError(f'unsupported schema ref {ref!r}')
    value: Any = root_schema
    for raw_part in ref[2:].split('/'):
        part = raw_part.replace('~1', '/').replace('~0', '~')
        value = value[part]
    return value


def type_matches(value: Any, expected: str) -> bool:
    if expected == 'object':
        return isinstance(value, dict)
    if expected == 'array':
        return isinstance(value, list)
    if expected == 'string':
        return isinstance(value, str)
    if expected == 'boolean':
        return isinstance(value, bool)
    if expected == 'integer':
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == 'number':
        return (isinstance(value, int | float) and not isinstance(value, bool))
    if expected == 'null':
        return value is None
    raise ValueError(f'unsupported schema type {expected!r}')


def is_valid_datetime(value: str) -> bool:
    if not RFC3339_DATETIME_RE.match(value):
        return False
    try:
        datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return False
    return True


def schema_errors(value: Any, schema: dict[str, Any], root_schema: dict[str, Any], path: list[str | int]) -> list[str]:
    errors: list[str] = []

    if '$ref' in schema:
        try:
            return schema_errors(value, resolve_ref(root_schema, schema['$ref']), root_schema, path)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f'{pointer(path)} has invalid schema ref {schema["$ref"]!r}: {exc}')
            return errors

    if 'allOf' in schema:
        for subschema in schema['allOf']:
            errors.extend(schema_errors(value, subschema, root_schema, path))

    if 'anyOf' in schema:
        if not any(not schema_errors(value, subschema, root_schema, path) for subschema in schema['anyOf']):
            errors.append(f'{pointer(path)} must match at least one anyOf branch')

    if 'oneOf' in schema:
        matches = sum(1 for subschema in schema['oneOf'] if not schema_errors(value, subschema, root_schema, path))
        if matches != 1:
            errors.append(f'{pointer(path)} must match exactly one oneOf branch; matched {matches}')

    if 'not' in schema and not schema_errors(value, schema['not'], root_schema, path):
        errors.append(f'{pointer(path)} must not match forbidden schema')

    if 'if' in schema:
        if not schema_errors(value, schema['if'], root_schema, path):
            if 'then' in schema:
                errors.extend(schema_errors(value, schema['then'], root_schema, path))
        elif 'else' in schema:
            errors.extend(schema_errors(value, schema['else'], root_schema, path))

    if 'type' in schema:
        expected_types = schema['type']
        if isinstance(expected_types, str):
            expected_types = [expected_types]
        if not any(type_matches(value, expected) for expected in expected_types):
            errors.append(f'{pointer(path)} must be type {expected_types}, got {type(value).__name__}')
            return errors

    if 'const' in schema and value != schema['const']:
        errors.append(f'{pointer(path)} must equal {schema["const"]!r}')

    if 'enum' in schema and value not in schema['enum']:
        errors.append(f'{pointer(path)} must be one of {schema["enum"]!r}')

    if isinstance(value, str):
        if 'minLength' in schema and len(value) < schema['minLength']:
            errors.append(f'{pointer(path)} must be at least {schema["minLength"]} characters')
        if schema.get('format') == 'date-time' and not is_valid_datetime(value):
            errors.append(f'{pointer(path)} must be a valid date-time')

    if isinstance(value, int | float) and not isinstance(value, bool):
        if 'minimum' in schema and value < schema['minimum']:
            errors.append(f'{pointer(path)} must be >= {schema["minimum"]}')

    if isinstance(value, dict):
        if 'required' in schema:
            missing = [field for field in schema['required'] if field not in value]
            if missing:
                errors.append(f'{pointer(path)} missing required fields {missing!r}')

        properties = schema.get('properties', {})
        if 'propertyNames' in schema:
            for key in value:
                errors.extend(schema_errors(key, schema['propertyNames'], root_schema, path + [key]))

        for key, child_schema in properties.items():
            if key in value:
                errors.extend(schema_errors(value[key], child_schema, root_schema, path + [key]))

        if schema.get('additionalProperties') is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                errors.append(f'{pointer(path)} has unexpected fields {extra!r}')
        elif isinstance(schema.get('additionalProperties'), dict):
            child_schema = schema['additionalProperties']
            for key in sorted(set(value) - set(properties)):
                errors.extend(schema_errors(value[key], child_schema, root_schema, path + [key]))

    if isinstance(value, list):
        if 'minItems' in schema and len(value) < schema['minItems']:
            errors.append(f'{pointer(path)} must contain at least {schema["minItems"]} items')
        if 'maxItems' in schema and len(value) > schema['maxItems']:
            errors.append(f'{pointer(path)} must contain at most {schema["maxItems"]} items')
        if schema.get('uniqueItems') is True:
            seen_items = set()
            for item in value:
                marker = json.dumps(item, sort_keys=True, separators=(',', ':'))
                if marker in seen_items:
                    errors.append(f'{pointer(path)} must contain unique items')
                    break
                seen_items.add(marker)
        if isinstance(schema.get('items'), dict):
            for index, item in enumerate(value):
                errors.extend(schema_errors(item, schema['items'], root_schema, path + [index]))

    return errors


def validate_against_schema(path: Path, data: Any, schema: dict[str, Any], errors: list[str]) -> None:
    for error in schema_errors(data, schema, schema, []):
        errors.append(f'{path} does not conform to {schema.get("title", "schema")}: {error}')


def walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(key)
            keys.extend(walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(walk_keys(child))
    return keys


def check_metadata(path: Path, data: dict[str, Any], errors: list[str]) -> None:
    missing = METADATA_FIELDS - set(data)
    if missing:
        errors.append(f'{path} missing metadata fields: {sorted(missing)}')
    for field in METADATA_FIELDS:
        if field in data and not isinstance(data[field], str):
            errors.append(f'{path} metadata field {field} must be a string')
        elif field in data and not data[field].strip():
            errors.append(f'{path} metadata field {field} must be non-empty')


def check_no_host_specific_paths(path: Path, data: Any, errors: list[str]) -> None:
    if isinstance(data, str):
        if data.startswith(HOST_SPECIFIC_PATH_PREFIXES) or WINDOWS_ABSOLUTE_PATH_RE.match(data):
            errors.append(f'{path} must not contain a host-specific absolute path: {data}')
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                if value.startswith(HOST_SPECIFIC_PATH_PREFIXES) or WINDOWS_ABSOLUTE_PATH_RE.match(value):
                    errors.append(f'{path} {key} must not use a host-specific absolute path: {value}')
            else:
                check_no_host_specific_paths(path, value, errors)
    elif isinstance(data, list):
        for value in data:
            check_no_host_specific_paths(path, value, errors)


def check_no_raw_secrets(path: Path, data: Any, errors: list[str]) -> None:
    if isinstance(data, str):
        if '-----BEGIN ' in data or data.startswith(('sk-', 'ghp_', 'whsec_')):
            errors.append(f'{path} appears to contain raw secret material')
    elif isinstance(data, dict):
        for key, value in data.items():
            normalized_key = key.replace('-', '').replace('_', '').lower()
            if normalized_key in SENSITIVE_KEY_NAMES:
                errors.append(f'{path} must not contain raw secret or signature field {key!r}')
            if isinstance(value, str) and ('-----BEGIN ' in value or value.startswith(('sk-', 'ghp_', 'whsec_'))):
                errors.append(f'{path} appears to contain raw secret material in field {key!r}')
            if not isinstance(value, str):
                check_no_raw_secrets(path, value, errors)
    elif isinstance(data, list):
        for value in data:
            check_no_raw_secrets(path, value, errors)


def validate_workspace_relative_path(path: Path, field_name: str, value: Any, errors: list[str]) -> None:
    if not isinstance(value, str):
        return
    if value.startswith('/'):
        errors.append(f'{path} {field_name} must be workspace-relative')
    if '..' in Path(value).parts:
        errors.append(f'{path} {field_name} must not traverse out of the workspace')


def validate_derived_from(
    path: Path,
    label: str,
    value: Any,
    expected_order: tuple[str, ...],
    expected_set: set[str],
    errors: list[str],
) -> None:
    if not isinstance(value, list):
        errors.append(f'{path} {label}.derivedFrom must be a list')
        return
    if any(not isinstance(item, str) for item in value):
        errors.append(f'{path} {label}.derivedFrom entries must be strings')
        return
    if set(value) != expected_set or len(value) != len(expected_order):
        errors.append(f'{path} {label}.derivedFrom must be {list(expected_order)}')


def validate_string_list_field(path: Path, data: dict[str, Any], label: str, field: str, errors: list[str]) -> None:
    value = data.get(field)
    if not isinstance(value, list) or not value:
        errors.append(f'{path} {label}.{field} must be a non-empty list')
        return
    if any(not isinstance(item, str) or not item.strip() for item in value):
        errors.append(f'{path} {label}.{field} entries must be non-empty strings')
    if len(set(value)) != len(value):
        errors.append(f'{path} {label}.{field} entries must be unique')


def validate_string_field(path: Path, data: dict[str, Any], label: str, field: str, errors: list[str]) -> None:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f'{path} {label}.{field} must be a non-empty string')


def discover_example_json_files(package: Path) -> set[str]:
    examples = package / 'examples'
    if not examples.exists():
        return set()
    return {str(path.relative_to(package)) for path in examples.rglob('*.json')}


def validate_known_json_files(package: Path, errors: list[str]) -> None:
    known_examples = {rel for rel in owned_json_files() if rel.startswith('examples/')}
    for rel in sorted(discover_example_json_files(package) - known_examples):
        errors.append(f'unexpected JSON example outside canonical contract list: {package / rel}')


def validate_schema(path: Path, data: Any, expected_version: str, errors: list[str]) -> None:
    if not isinstance(data, dict):
        errors.append(f'{path} schema root must be an object')
        return
    if data.get('$schema') != SCHEMA_DRAFT:
        errors.append(f'{path} must declare JSON Schema draft 2020-12')
    version = data.get('properties', {}).get('schemaVersion', {}).get('const')
    if version != expected_version:
        errors.append(f'{path} schemaVersion const must be {expected_version}')


def validate_command(path: Path, data: Any, errors: list[str]) -> None:
    if not isinstance(data, dict):
        errors.append(f'{path} command example root must be an object')
        return
    check_metadata(path, data, errors)
    required = {'provider', 'command', 'correlationId', 'issuedAt', 'success'}
    missing = required - set(data)
    if missing:
        errors.append(f'{path} missing command fields: {sorted(missing)}')
    if data.get('schemaVersion') != COMMAND_VERSION:
        errors.append(f'{path} must use schemaVersion {COMMAND_VERSION}')
    command = data.get('command')
    success = data.get('success')
    if not isinstance(success, bool):
        errors.append(f'{path} success must be boolean')
        return
    forbidden = COMMAND_FORBIDDEN_TEXT_KEYS & set(walk_keys(data))
    if forbidden:
        errors.append(f'{path} contains human/prose text keys that Hermes must not parse: {sorted(forbidden)}')
    if success:
        if 'result' not in data or not isinstance(data.get('result'), dict):
            errors.append(f'{path} success example must include object result')
        if 'error' in data:
            errors.append(f'{path} success example must not include error')
        if isinstance(command, str) and command.startswith('workflow.') and 'workflowRunRef' not in data:
            errors.append(f'{path} successful workflow command must include workflowRunRef')
        if isinstance(command, str) and command.startswith('binding.') and 'bindingRef' not in data:
            errors.append(f'{path} successful binding command must include bindingRef')
    else:
        if 'error' not in data or not isinstance(data.get('error'), dict):
            errors.append(f'{path} failed example must include object error')
            return
        if 'result' in data:
            errors.append(f'{path} failed example must not include result')
        error = data['error']
        for field in ('code', 'category', 'retryable', 'details'):
            if field not in error:
                errors.append(f'{path} error missing {field}')
        if not isinstance(error.get('retryable'), bool):
            errors.append(f'{path} error retryable must be boolean')
        if not isinstance(error.get('details'), dict):
            errors.append(f'{path} error details must be object')


def validate_binding(path: Path, data: Any, errors: list[str]) -> None:
    if not isinstance(data, dict):
        errors.append(f'{path} binding example root must be an object')
        return
    check_metadata(path, data, errors)
    if data.get('schemaVersion') != BINDING_VERSION:
        errors.append(f'{path} must use schemaVersion {BINDING_VERSION}')
    for field in ('provider', 'name', 'shape', 'operation', 'correlationId'):
        if field not in data:
            errors.append(f'{path} missing binding field {field}')
    forbidden = BINDING_FORBIDDEN_KEYS & set(walk_keys(data))
    if forbidden:
        errors.append(f'{path} contains forbidden binding keys: {sorted(forbidden)}')
    if data.get('owningSubproject') != 'archon':
        errors.append(f'{path} binding owningSubproject must be archon')
    shape = data.get('shape')
    if shape == 'request':
        if data.get('intendedProducer') != 'Hermes' or data.get('intendedConsumer') != 'Archon':
            errors.append(f'{path} request binding payload must be Hermes -> Archon')
        operation = data.get('operation')
        if operation not in {'create', 'update'}:
            errors.append(f'{path} request binding operation must be create or update')
        elif 'request' not in data:
            errors.append(f'{path} create/update request must include request object')
    elif shape in {'result', 'status', 'error'}:
        if data.get('intendedProducer') != 'Archon' or data.get('intendedConsumer') != 'Hermes':
            errors.append(f'{path} {shape} binding payload must be Archon -> Hermes')
    else:
        errors.append(f'{path} unsupported binding shape {shape!r}')


def validate_delivery(path: Path, data: Any, errors: list[str]) -> None:
    if not isinstance(data, dict):
        errors.append(f'{path} delivery example root must be an object')
        return
    check_metadata(path, data, errors)
    if data.get('schemaVersion') != DELIVERY_VERSION:
        errors.append(f'{path} must use schemaVersion {DELIVERY_VERSION}')
    for field in ('provider', 'correlationId', 'status', 'workflowRunRef', 'checkedAt'):
        if field not in data:
            errors.append(f'{path} missing delivery field {field}')
    status = data.get('status')
    if status in DELIVERY_ALIASES:
        errors.append(f'{path} uses compatibility alias {status}; emit reconciliation-pending instead')
    if status not in DELIVERY_STATUSES:
        errors.append(f'{path} unsupported delivery status {status!r}')
    if data.get('blockingWorkflowExecution') is not False:
        errors.append(f'{path} delivery status must not block workflow execution')
    if status == 'duplicated' and data.get('duplicateSafe') is not True:
        errors.append(f'{path} duplicated delivery must set duplicateSafe true')
    if status == 'reconciliation-pending' and data.get('reconciliationNeeded') is not True:
        errors.append(f'{path} reconciliation-pending delivery must set reconciliationNeeded true')
    if status in {'failed', 'terminal-failure'}:
        if not isinstance(data.get('diagnostic'), dict):
            errors.append(f'{path} {status} delivery must include object diagnostic')
        else:
            require_object_fields(path, data['diagnostic'], 'diagnostic', {'category', 'code', 'actionRequired'}, errors)
            if data['diagnostic'].get('actionRequired') is not True:
                errors.append(f'{path} {status} diagnostic.actionRequired must be true')
        if not isinstance(data.get('recovery'), dict):
            errors.append(f'{path} {status} delivery must include object recovery')
        else:
            require_object_fields(path, data['recovery'], 'recovery', {'kind', 'safeToRetry'}, errors)


def require_object_fields(path: Path, data: dict[str, Any], container: str, fields: set[str], errors: list[str]) -> None:
    missing = fields - set(data)
    if missing:
        errors.append(f'{path} {container} missing fields: {sorted(missing)}')


def package_workspace_root(package: Path) -> Path:
    return package.parents[3]


def iter_string_values(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for child in value.values():
            values.extend(iter_string_values(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(iter_string_values(child))
    elif isinstance(value, str):
        values.append(value)
    return values


def check_file_uris(path: Path, package: Path, data: Any, errors: list[str]) -> None:
    workspace_root = package_workspace_root(package)
    for value in iter_string_values(data):
        if not value.startswith('file:'):
            continue
        uri_path = value.removeprefix('file:')
        if uri_path.startswith('/'):
            errors.append(f'{path} file URI must be workspace-relative, got {value}')
            continue
        if '..' in Path(uri_path).parts:
            errors.append(f'{path} file URI must not traverse out of the workspace: {value}')
            continue
        if not (workspace_root / uri_path).exists():
            errors.append(f'{path} file URI target does not exist from {workspace_root}: {value}')


def validate_project_identity(
    path: Path,
    project_ref: Any,
    binding_ref: Any,
    workflow_run_ref: Any,
    errors: list[str],
) -> None:
    if not isinstance(project_ref, dict):
        return
    project_id = project_ref.get('id')
    if not isinstance(project_id, str) or not project_id.strip():
        return
    expected_project_ref = f'project:{project_id}'
    for container, value in (
        ('bindingRef', binding_ref),
        ('workflowRunRef', workflow_run_ref),
    ):
        if isinstance(value, dict) and value.get('projectRef') != expected_project_ref:
            errors.append(f'{path} {container}.projectRef must match projectRef.id as {expected_project_ref}')


def validate_profile_route_shape(path: Path, provider: Any, profile_route: Any, errors: list[str]) -> None:
    if not isinstance(provider, str) or not isinstance(profile_route, dict):
        return
    profile = profile_route.get('profile')
    if not isinstance(profile, str) or not profile.strip():
        return
    expected_path = f'/p/{profile}/webhooks/workflow-events/{provider}'
    if profile_route.get('ingressPath') != expected_path:
        errors.append(f'{path} profileRoute.ingressPath must be {expected_path}')


def validate_signature_policy(path: Path, signature: Any, errors: list[str]) -> None:
    if not isinstance(signature, dict):
        return
    for field, expected in DEFERRED_SIGNATURE_POLICY.items():
        if signature.get(field) != expected:
            errors.append(f'{path} signature.{field} must use deferred placeholder {expected}')


def validate_event_payload(path: Path, event_type: Any, payload: Any, errors: list[str]) -> None:
    if not isinstance(event_type, str) or not isinstance(payload, dict):
        return
    required = PAYLOAD_REQUIRED_FIELDS.get(event_type)
    if required is None:
        return
    require_object_fields(path, payload, f'payload for {event_type}', required, errors)
    if event_type == 'workflow.run.completed' and isinstance(payload.get('result'), dict):
        require_object_fields(path, payload['result'], 'payload.result', {'outcome'}, errors)
    if event_type == 'workflow.run.failed' and isinstance(payload.get('failure'), dict):
        require_object_fields(path, payload['failure'], 'payload.failure', {'code', 'category', 'retryable', 'details'}, errors)
    if event_type == 'workflow.approval.requested' and isinstance(payload.get('approval'), dict):
        require_object_fields(path, payload['approval'], 'payload.approval', {'requestId', 'requestedAction', 'phase'}, errors)
    if event_type == 'workflow.delivery.failed':
        if payload.get('deliveryOnly') is not True:
            errors.append(f'{path} workflow.delivery.failed payload.deliveryOnly must be true')
        if payload.get('mutationIntent') != 'none':
            errors.append(f'{path} workflow.delivery.failed payload.mutationIntent must be none')
    if event_type == 'workflow.artifact.recorded' and isinstance(payload.get('artifact'), dict):
        require_object_fields(path, payload['artifact'], 'payload.artifact', {'type', 'name', 'uri', 'digest'}, errors)


def is_valid_redelivery(original: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if candidate.get('eventId') != original.get('eventId'):
        return False
    if candidate.get('idempotencyKey') != original.get('idempotencyKey'):
        return False
    for field in ('provider', 'eventType', 'bindingRef', 'workflowRunRef', 'projectRef', 'profileRoute', 'payload'):
        if candidate.get(field) != original.get(field):
            return False
    original_delivery = original.get('delivery')
    candidate_delivery = candidate.get('delivery')
    if not isinstance(original_delivery, dict) or not isinstance(candidate_delivery, dict):
        return False
    original_attempt = original_delivery.get('attempt')
    candidate_attempt = candidate_delivery.get('attempt')
    if not isinstance(original_attempt, int) or not isinstance(candidate_attempt, int):
        return False
    if candidate_attempt <= original_attempt:
        return False
    return candidate_delivery.get('deliveryId') != original_delivery.get('deliveryId')


def validate_event(path: Path, data: Any, errors: list[str]) -> None:
    if not isinstance(data, dict):
        errors.append(f'{path} event example root must be an object')
        return
    check_metadata(path, data, errors)
    if data.get('schemaVersion') != EVENT_VERSION:
        errors.append(f'{path} must use schemaVersion {EVENT_VERSION}')
    if data.get('intendedProducer') != 'Archon' or data.get('intendedConsumer') != 'Hermes':
        errors.append(f'{path} event payload must be Archon -> Hermes')
    if data.get('owningSubproject') != 'archon':
        errors.append(f'{path} event owningSubproject must be archon')
    required = {
        'provider',
        'eventId',
        'eventType',
        'occurredAt',
        'bindingRef',
        'workflowRunRef',
        'projectRef',
        'profileRoute',
        'idempotencyKey',
        'signature',
        'payload',
    }
    require_object_fields(path, data, 'event', required, errors)
    if data.get('provider') != 'archon':
        errors.append(f'{path} provider must be archon')
    if data.get('eventType') not in EVENT_TYPES:
        errors.append(f'{path} unsupported eventType {data.get("eventType")!r}')
    for field in ('eventId', 'idempotencyKey'):
        if not isinstance(data.get(field), str) or not data[field].strip():
            errors.append(f'{path} {field} must be a non-empty string')
    for field in ('occurredAt',):
        if isinstance(data.get(field), str) and not is_valid_datetime(data[field]):
            errors.append(f'{path} {field} must be a valid date-time')

    binding_ref = data.get('bindingRef')
    if isinstance(binding_ref, dict):
        require_object_fields(path, binding_ref, 'bindingRef', {'provider', 'name', 'bindingId', 'projectRef'}, errors)
        if binding_ref.get('provider') != data.get('provider'):
            errors.append(f'{path} bindingRef provider must match event provider')
    elif 'bindingRef' in data:
        errors.append(f'{path} bindingRef must be an object')

    workflow_run_ref = data.get('workflowRunRef')
    if isinstance(workflow_run_ref, dict):
        require_object_fields(path, workflow_run_ref, 'workflowRunRef', {'provider', 'runId', 'workflowName', 'projectRef'}, errors)
        if workflow_run_ref.get('provider') != data.get('provider'):
            errors.append(f'{path} workflowRunRef provider must match event provider')
    elif 'workflowRunRef' in data:
        errors.append(f'{path} workflowRunRef must be an object')

    project_ref = data.get('projectRef')
    if isinstance(project_ref, dict):
        require_object_fields(path, project_ref, 'projectRef', {'id', 'codebaseRef', 'repositoryPath'}, errors)
    elif 'projectRef' in data:
        errors.append(f'{path} projectRef must be an object')
    validate_project_identity(path, project_ref, binding_ref, workflow_run_ref, errors)

    profile_route = data.get('profileRoute')
    if isinstance(profile_route, dict):
        require_object_fields(
            path,
            profile_route,
            'profileRoute',
            {'profile', 'ingressPath', 'bindingName', 'secretRef', 'expectedProvider'},
            errors,
        )
        if isinstance(binding_ref, dict) and profile_route.get('bindingName') != binding_ref.get('name'):
            errors.append(f'{path} profileRoute bindingName must match bindingRef name')
        if profile_route.get('expectedProvider') != data.get('provider'):
            errors.append(f'{path} profileRoute expectedProvider must match event provider')
        validate_profile_route_shape(path, data.get('provider'), profile_route, errors)
    elif 'profileRoute' in data:
        errors.append(f'{path} profileRoute must be an object')

    signature = data.get('signature')
    if isinstance(signature, dict):
        require_object_fields(
            path,
            signature,
            'signature',
            {'algorithm', 'keyId', 'signedAt', 'signatureHeader', 'timestampHeader', 'signatureInput', 'bodyDigest'},
            errors,
        )
        if isinstance(signature.get('signedAt'), str) and not is_valid_datetime(signature['signedAt']):
            errors.append(f'{path} signature.signedAt must be a valid date-time')
        if not isinstance(signature.get('signatureInput'), list) or not signature.get('signatureInput'):
            errors.append(f'{path} signature.signatureInput must be a non-empty list')
        validate_signature_policy(path, signature, errors)
    elif 'signature' in data:
        errors.append(f'{path} signature must be an object')

    if 'payload' in data and not isinstance(data.get('payload'), dict):
        errors.append(f'{path} payload must be an object')
    validate_event_payload(path, data.get('eventType'), data.get('payload'), errors)


def validate_rejection(
    path: Path,
    data: Any,
    event_schema: dict[str, Any] | None,
    errors: list[str],
) -> None:
    if not isinstance(data, dict):
        errors.append(f'{path} rejection example root must be an object')
        return
    check_metadata(path, data, errors)
    if data.get('schemaVersion') != REJECTION_VERSION:
        errors.append(f'{path} must use schemaVersion {REJECTION_VERSION}')
    if data.get('intendedProducer') != 'Hermes' or data.get('intendedConsumer') != 'Archon':
        errors.append(f'{path} rejection fixture must be Hermes -> Archon')
    if data.get('owningSubproject') != 'hermes-agent':
        errors.append(f'{path} rejection owningSubproject must be hermes-agent')

    required = {
        'caseId',
        'rejectionReason',
        'diagnosticCategory',
        'expectedHttpStatus',
        'expectedMutationBehavior',
        'rejectionEvidence',
        'eventEnvelope',
    }
    require_object_fields(path, data, 'rejection', required, errors)
    reason = data.get('rejectionReason')
    if reason not in REJECTION_REASONS:
        errors.append(f'{path} unsupported rejectionReason {reason!r}')
    elif path.stem != reason:
        errors.append(f'{path} filename stem must match rejectionReason {reason!r}')
    if data.get('diagnosticCategory') not in REJECTION_DIAGNOSTIC_CATEGORIES:
        errors.append(f'{path} unsupported diagnosticCategory {data.get("diagnosticCategory")!r}')

    expected_mutation = data.get('expectedMutationBehavior')
    if isinstance(expected_mutation, dict):
        require_object_fields(
            path,
            expected_mutation,
            'expectedMutationBehavior',
            NO_MUTATION_FIELDS | {'diagnosticEmission'},
            errors,
        )
        for field in sorted(NO_MUTATION_FIELDS):
            if expected_mutation.get(field) != 'none':
                errors.append(f'{path} expectedMutationBehavior.{field} must be none')
        if expected_mutation.get('diagnosticEmission') is not True:
            errors.append(f'{path} expectedMutationBehavior.diagnosticEmission must be true')
    elif 'expectedMutationBehavior' in data:
        errors.append(f'{path} expectedMutationBehavior must be an object')

    event_envelope = data.get('eventEnvelope')
    if not isinstance(event_envelope, dict):
        if 'eventEnvelope' in data:
            errors.append(f'{path} eventEnvelope must be an object')
        return

    evidence = data.get('rejectionEvidence')
    if not isinstance(evidence, dict):
        if 'rejectionEvidence' in data:
            errors.append(f'{path} rejectionEvidence must be an object')
        return
    if evidence.get('checkedBeforeMutation') is not True:
        errors.append(f'{path} rejectionEvidence.checkedBeforeMutation must be true')

    validate_project_identity(
        path,
        event_envelope.get('projectRef'),
        event_envelope.get('bindingRef'),
        event_envelope.get('workflowRunRef'),
        errors,
    )
    validate_profile_route_shape(path, event_envelope.get('provider'), event_envelope.get('profileRoute'), errors)
    validate_signature_policy(path, event_envelope.get('signature'), errors)

    if reason != 'schema-mismatch' and event_schema is not None:
        for error in schema_errors(event_envelope, event_schema, event_schema, []):
            errors.append(f'{path} eventEnvelope should be structurally valid before semantic rejection: {error}')
    if reason == 'schema-mismatch' and event_schema is not None:
        if not schema_errors(event_envelope, event_schema, event_schema, []):
            errors.append(f'{path} schema-mismatch fixture must contain an eventEnvelope that fails the event schema')
        if evidence.get('failsSchemaValidation') is not True:
            errors.append(f'{path} schema-mismatch evidence must set failsSchemaValidation true')
    if reason == 'bad-signature' and evidence.get('signatureVerification') != 'failed':
        errors.append(f'{path} bad-signature evidence must set signatureVerification failed')
    if reason == 'stale-timestamp' and evidence.get('replayWindowStatus') != 'stale':
        errors.append(f'{path} stale-timestamp evidence must set replayWindowStatus stale')
    if reason == 'stale-timestamp':
        if 'allowedClockSkewSeconds' in evidence:
            errors.append(f'{path} stale-timestamp evidence must not hard-code allowedClockSkewSeconds')
        if evidence.get('replayWindowDecisionRef') != 'deferred-signature-and-replay-policy':
            errors.append(f'{path} stale-timestamp evidence must reference deferred-signature-and-replay-policy')
    if reason == 'duplicate-event-id' and evidence.get('duplicateOfEventId') != event_envelope.get('eventId'):
        errors.append(f'{path} duplicate-event-id evidence must reference the duplicate eventId')
    if reason == 'wrong-binding' and evidence.get('bindingAccepted') is not False:
        errors.append(f'{path} wrong-binding evidence must set bindingAccepted false')
    if reason == 'unknown-project' and evidence.get('projectLookupStatus') != 'missing':
        errors.append(f'{path} unknown-project evidence must set projectLookupStatus missing')
    if reason == 'wrong-profile-secret':
        if evidence.get('signatureVerifiedAgainstPresentedProfile') is not True:
            errors.append(f'{path} wrong-profile-secret evidence must verify against the presented profile')
        if evidence.get('profileRouteAccepted') is not False:
            errors.append(f'{path} wrong-profile-secret evidence must set profileRouteAccepted false')
    if reason == 'wrong-codebase' and evidence.get('codebaseAccepted') is not False:
        errors.append(f'{path} wrong-codebase evidence must set codebaseAccepted false')
    if reason == 'unsupported-provider' and evidence.get('providerSupported') is not False:
        errors.append(f'{path} unsupported-provider evidence must set providerSupported false')


def identity_key_component(value: str) -> str:
    return quote(value, safe='')


def derive_project_work_identity_key(identity_inputs: dict[str, Any]) -> str | None:
    bound_project_cwd = identity_inputs.get('boundProjectCwd')
    artifact_path = identity_inputs.get('bmadArtifactPath')
    bmad_identity = identity_inputs.get('bmadIdentity')
    if not isinstance(bound_project_cwd, str) or not bound_project_cwd.strip():
        return None
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        return None
    if not isinstance(bmad_identity, dict):
        return None
    kind = bmad_identity.get('kind')
    bmad_id = bmad_identity.get('id')
    if kind not in {'epic', 'story'} or not isinstance(bmad_id, str) or not bmad_id.strip():
        return None
    return 'pwi:' + ':'.join(
        identity_key_component(component)
        for component in (bound_project_cwd, artifact_path, kind, bmad_id)
    )


def validate_project_work_identity(path: Path, data: Any, errors: list[str]) -> None:
    if not isinstance(data, dict):
        errors.append(f'{path} projectWorkItemIdentity must be an object')
        return
    if data.get('schemaVersion') != PROJECT_WORK_IDENTITY_VERSION:
        errors.append(f'{path} projectWorkItemIdentity must use schemaVersion {PROJECT_WORK_IDENTITY_VERSION}')
    require_object_fields(
        path,
        data,
        'projectWorkItemIdentity',
        {'identityKey', 'identityInputs', 'derivedFrom'},
        errors,
    )
    forbidden = PROJECT_WORK_IDENTITY_FORBIDDEN_KEYS & set(walk_keys(data))
    if forbidden:
        errors.append(f'{path} projectWorkItemIdentity includes phase or volatile identity fields: {sorted(forbidden)}')
    if isinstance(data.get('identityKey'), str) and PHASE_KIND in data['identityKey']:
        errors.append(f'{path} projectWorkItemIdentity.identityKey must not include phase kind')
    if data.get('identityKeyAlgorithm') != PROJECT_WORK_IDENTITY_ALGORITHM:
        errors.append(f'{path} projectWorkItemIdentity.identityKeyAlgorithm must be {PROJECT_WORK_IDENTITY_ALGORITHM}')

    validate_derived_from(
        path,
        'projectWorkItemIdentity',
        data.get('derivedFrom'),
        PROJECT_WORK_IDENTITY_DERIVED_FROM,
        PROJECT_WORK_IDENTITY_DERIVED_FROM_SET,
        errors,
    )

    inputs = data.get('identityInputs')
    if not isinstance(inputs, dict):
        if 'identityInputs' in data:
            errors.append(f'{path} projectWorkItemIdentity.identityInputs must be an object')
        return
    require_object_fields(
        path,
        inputs,
        'projectWorkItemIdentity.identityInputs',
        {'boundProjectCwd', 'bmadArtifactPath', 'bmadIdentity'},
        errors,
    )
    artifact_path = inputs.get('bmadArtifactPath')
    validate_workspace_relative_path(path, 'bmadArtifactPath', artifact_path, errors)
    bmad_identity = inputs.get('bmadIdentity')
    if not isinstance(bmad_identity, dict):
        if 'bmadIdentity' in inputs:
            errors.append(f'{path} projectWorkItemIdentity.identityInputs.bmadIdentity must be an object')
        return
    require_object_fields(path, bmad_identity, 'bmadIdentity', {'kind', 'id'}, errors)
    kind = bmad_identity.get('kind')
    if kind not in {'epic', 'story'}:
        errors.append(f'{path} bmadIdentity.kind must be epic or story')
    if kind == 'story':
        require_object_fields(path, bmad_identity, 'bmadIdentity story', {'epicId', 'storyId'}, errors)
        if bmad_identity.get('id') != bmad_identity.get('storyId'):
            errors.append(f'{path} bmadIdentity.id must equal storyId for story identities')
    if kind == 'epic':
        require_object_fields(path, bmad_identity, 'bmadIdentity epic', {'epicId'}, errors)
        if bmad_identity.get('id') != bmad_identity.get('epicId'):
            errors.append(f'{path} bmadIdentity.id must equal epicId for epic identities')
    expected_identity_key = derive_project_work_identity_key(inputs)
    if expected_identity_key is not None and data.get('identityKey') != expected_identity_key:
        errors.append(f'{path} projectWorkItemIdentity.identityKey must be derived from identityInputs')


def validate_phase_task_identity(path: Path, data: Any, errors: list[str]) -> None:
    if not isinstance(data, dict):
        errors.append(f'{path} phaseTaskIdentity must be an object')
        return
    if data.get('schemaVersion') != PHASE_TASK_IDENTITY_VERSION:
        errors.append(f'{path} phaseTaskIdentity must use schemaVersion {PHASE_TASK_IDENTITY_VERSION}')
    require_object_fields(
        path,
        data,
        'phaseTaskIdentity',
        {'identityKey', 'projectWorkItemIdentityKey', 'phaseKind', 'derivedFrom'},
        errors,
    )
    if data.get('phaseKind') != PHASE_KIND:
        errors.append(f'{path} phaseTaskIdentity.phaseKind must be {PHASE_KIND}')
    project_work_key = data.get('projectWorkItemIdentityKey')
    if isinstance(project_work_key, str) and data.get('identityKey') != f'phase:{project_work_key}:{PHASE_KIND}':
        errors.append(f'{path} phaseTaskIdentity.identityKey must be derived from Project Work Item identity plus phase kind')
    if 'workflowName' in data and data.get('workflowName') != COMBINED_WORKFLOW_NAME:
        errors.append(f'{path} phaseTaskIdentity.workflowName must be {COMBINED_WORKFLOW_NAME}')
    if data.get('identityKeyAlgorithm') != PHASE_TASK_IDENTITY_ALGORITHM:
        errors.append(f'{path} phaseTaskIdentity.identityKeyAlgorithm must be {PHASE_TASK_IDENTITY_ALGORITHM}')
    validate_derived_from(
        path,
        'phaseTaskIdentity',
        data.get('derivedFrom'),
        PHASE_TASK_IDENTITY_DERIVED_FROM,
        PHASE_TASK_IDENTITY_DERIVED_FROM_SET,
        errors,
    )


def validate_downstream_readiness(path: Path, data: Any, errors: list[str]) -> None:
    readiness = data.get('downstreamReadinessExpectations')
    if not isinstance(readiness, dict):
        errors.append(f'{path} downstreamReadinessExpectations must be an object')
        return
    require_object_fields(
        path,
        readiness,
        'downstreamReadinessExpectations',
        DOWNSTREAM_READINESS_FLAG_FIELDS | DOWNSTREAM_READINESS_EXPECTATION_FIELDS,
        errors,
    )
    for field in sorted(DOWNSTREAM_READINESS_FLAG_FIELDS):
        if readiness.get(field) is not True:
            errors.append(f'{path} downstreamReadinessExpectations.{field} must be true')
    expectation_shapes = {
        'migrationExpectation': ('recordKinds', 'requiredTest'),
        'uniquenessExpectation': ('identityKeys', 'requiredTest'),
        'idempotencyExpectation': ('rerunCases', 'requiredTest'),
    }
    for field, (list_field, string_field) in expectation_shapes.items():
        expectation = readiness.get(field)
        if not isinstance(expectation, dict):
            errors.append(f'{path} downstreamReadinessExpectations.{field} must be an object')
            continue
        require_object_fields(path, expectation, f'downstreamReadinessExpectations.{field}', {list_field, string_field}, errors)
        validate_string_list_field(path, expectation, f'downstreamReadinessExpectations.{field}', list_field, errors)
        validate_string_field(path, expectation, f'downstreamReadinessExpectations.{field}', string_field, errors)


def validate_materialization_mutation(
    path: Path,
    data: dict[str, Any],
    case_kind: Any,
    project_work_key: str | None,
    phase_task_key: str | None,
    errors: list[str],
) -> None:
    mutation = data.get('expectedMutationBehavior')
    if not isinstance(mutation, dict):
        errors.append(f'{path} expectedMutationBehavior must be an object')
        return
    require_object_fields(
        path,
        mutation,
        'expectedMutationBehavior',
        {
            'projectWorkItemMutation',
            'phaseTaskMutation',
            'duplicateProjectWorkItemsCreated',
            'duplicatePhaseTasksCreated',
            'diagnosticEmission',
        },
        errors,
    )
    for field in ('projectWorkItemMutation', 'phaseTaskMutation'):
        if mutation.get(field) not in MATERIALIZATION_MUTATIONS:
            errors.append(f'{path} expectedMutationBehavior.{field} has unsupported value {mutation.get(field)!r}')
    expected_case_mutations = MATERIALIZATION_CASE_MUTATIONS.get(case_kind)
    if expected_case_mutations is not None:
        project_mutation, phase_mutation = expected_case_mutations
        if mutation.get('projectWorkItemMutation') != project_mutation:
            errors.append(f'{path} expectedMutationBehavior.projectWorkItemMutation must be {project_mutation}')
        if mutation.get('phaseTaskMutation') != phase_mutation:
            errors.append(f'{path} expectedMutationBehavior.phaseTaskMutation must be {phase_mutation}')
    for field in ('duplicateProjectWorkItemsCreated', 'duplicatePhaseTasksCreated'):
        if mutation.get(field) != 0:
            errors.append(f'{path} expectedMutationBehavior.{field} must be 0')
    if project_work_key is not None and mutation.get('projectWorkItemIdentityKey') != project_work_key:
        errors.append(f'{path} expectedMutationBehavior.projectWorkItemIdentityKey must match projectWorkItemIdentity')
    if phase_task_key is not None and mutation.get('phaseTaskIdentityKey') != phase_task_key:
        errors.append(f'{path} expectedMutationBehavior.phaseTaskIdentityKey must match phaseTaskIdentity')
    if case_kind in MATERIALIZATION_SUCCESS_CASES:
        for field in ('projectWorkItemIdentityStable', 'phaseTaskIdentityStable'):
            if mutation.get(field) is not True:
                errors.append(f'{path} expectedMutationBehavior.{field} must be true')
        for field in ('persistedProjectWorkItemCount', 'persistedPhaseTaskCount'):
            if mutation.get(field) != 1:
                errors.append(f'{path} expectedMutationBehavior.{field} must be 1')
        if mutation.get('diagnosticEmission') is not False:
            errors.append(f'{path} expectedMutationBehavior.diagnosticEmission must be false for success cases')


def validate_materialization_attempts(
    path: Path,
    data: dict[str, Any],
    project_work_key: str,
    phase_task_key: str,
    errors: list[str],
) -> None:
    attempts = data.get('materializationAttempts')
    if not isinstance(attempts, list) or len(attempts) < 2:
        errors.append(f'{path} duplicate prevention must include at least two materializationAttempts')
        return
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            errors.append(f'{path} materializationAttempts[{index}] must be an object')
            continue
        if attempt.get('projectWorkItemIdentityKey') != project_work_key:
            errors.append(f'{path} materializationAttempts[{index}] must keep the same Project Work Item identity')
        if attempt.get('phaseTaskIdentityKey') != phase_task_key:
            errors.append(f'{path} materializationAttempts[{index}] must keep the same Phase Task identity')
        expected_attempt = index + 1
        if attempt.get('attempt') != expected_attempt:
            errors.append(f'{path} materializationAttempts[{index}].attempt must be {expected_attempt}')
        expected_lookup = 'missing' if index == 0 else 'found'
        expected_mutation = 'create' if index == 0 else 'reuse'
        if attempt.get('phaseTaskLookupResult') != expected_lookup:
            errors.append(
                f'{path} materializationAttempts[{index}].phaseTaskLookupResult must be {expected_lookup}'
            )
        if attempt.get('phaseTaskMutation') != expected_mutation:
            errors.append(f'{path} materializationAttempts[{index}].phaseTaskMutation must be {expected_mutation}')
    mutation = data.get('expectedMutationBehavior')
    if isinstance(mutation, dict):
        if mutation.get('persistedPhaseTaskCount') != 1:
            errors.append(f'{path} duplicate prevention must prove one persisted phase task')
        if mutation.get('phaseTaskMutation') != 'reuse':
            errors.append(f'{path} duplicate prevention final phaseTaskMutation must be reuse')


def validate_materialization_case(
    path: Path,
    data: Any,
    schemas: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    if not isinstance(data, dict):
        errors.append(f'{path} materialization example root must be an object')
        return
    check_metadata(path, data, errors)
    if data.get('schemaVersion') != MATERIALIZATION_VERSION:
        errors.append(f'{path} must use schemaVersion {MATERIALIZATION_VERSION}')
    if data.get('intendedProducer') != 'Hermes' or data.get('intendedConsumer') != 'Hermes':
        errors.append(f'{path} materialization fixture must be Hermes -> Hermes')
    if data.get('owningSubproject') != 'hermes-agent':
        errors.append(f'{path} materialization owningSubproject must be hermes-agent')
    require_object_fields(
        path,
        data,
        'materialization',
        {'caseId', 'caseKind', 'projectBindingRef', 'source', 'expectedMutationBehavior'},
        errors,
    )
    case_kind = data.get('caseKind')
    if not isinstance(case_kind, str):
        errors.append(f'{path} caseKind must be a string')
        case_kind = None
    elif case_kind not in MATERIALIZATION_SUCCESS_CASES | MATERIALIZATION_FAIL_CLOSED_CASES:
        errors.append(f'{path} unsupported materialization caseKind {case_kind!r}')
    if isinstance(case_kind, str) and path.stem != case_kind:
        errors.append(f'{path} filename stem must match caseKind {case_kind!r}')

    schema = schemas.get('materialization-case.schema.json')
    if schema is not None:
        validate_against_schema(path, data, schema, errors)

    binding_ref = data.get('projectBindingRef')
    source = data.get('source')
    if isinstance(binding_ref, dict):
        require_object_fields(path, binding_ref, 'projectBindingRef', {'bindingId', 'projectId', 'boundProjectCwd'}, errors)
    elif 'projectBindingRef' in data:
        errors.append(f'{path} projectBindingRef must be an object')
    if isinstance(source, dict):
        require_object_fields(path, source, 'source', {'artifactKind', 'artifactPath', 'availability'}, errors)
        validate_workspace_relative_path(path, 'source.artifactPath', source.get('artifactPath'), errors)
        expected_availability = MATERIALIZATION_SOURCE_AVAILABILITY.get(case_kind)
        if expected_availability is not None and source.get('availability') != expected_availability:
            errors.append(f'{path} source.availability must be {expected_availability}')
    elif 'source' in data:
        errors.append(f'{path} source must be an object')

    project_work_identity = data.get('projectWorkItemIdentity')
    phase_task_identity = data.get('phaseTaskIdentity')
    project_work_key = None
    phase_task_key = None
    if case_kind in MATERIALIZATION_SUCCESS_CASES:
        if not isinstance(project_work_identity, dict):
            errors.append(f'{path} successful materialization must include projectWorkItemIdentity')
        else:
            project_schema = schemas.get('project-work-identity.schema.json')
            if project_schema is not None:
                validate_against_schema(path, project_work_identity, project_schema, errors)
            validate_project_work_identity(path, project_work_identity, errors)
            project_work_key = project_work_identity.get('identityKey')
            inputs = project_work_identity.get('identityInputs')
            if isinstance(inputs, dict) and isinstance(binding_ref, dict):
                if inputs.get('boundProjectCwd') != binding_ref.get('boundProjectCwd'):
                    errors.append(f'{path} Project Work Item identity bound cwd must match projectBindingRef')
            if isinstance(inputs, dict) and isinstance(source, dict):
                if inputs.get('bmadArtifactPath') != source.get('artifactPath'):
                    errors.append(f'{path} Project Work Item identity artifact path must match source artifactPath')
                bmad_identity = inputs.get('bmadIdentity')
                if isinstance(bmad_identity, dict):
                    require_object_fields(
                        path,
                        source,
                        'successful materialization source',
                        {'bmadEpicId', 'bmadStoryId', 'observedStoryStatus', 'storyTitle', 'contentDigest'},
                        errors,
                    )
                    if source.get('bmadStoryId') != bmad_identity.get('storyId'):
                        errors.append(f'{path} Project Work Item identity storyId must match source bmadStoryId')
                    if source.get('bmadEpicId') != bmad_identity.get('epicId'):
                        errors.append(f'{path} Project Work Item identity epicId must match source bmadEpicId')
        if not isinstance(phase_task_identity, dict):
            errors.append(f'{path} successful materialization must include phaseTaskIdentity')
        else:
            phase_schema = schemas.get('phase-task-identity.schema.json')
            if phase_schema is not None:
                validate_against_schema(path, phase_task_identity, phase_schema, errors)
            validate_phase_task_identity(path, phase_task_identity, errors)
            phase_task_key = phase_task_identity.get('identityKey')
            if project_work_key is not None and phase_task_identity.get('projectWorkItemIdentityKey') != project_work_key:
                errors.append(f'{path} Phase Task identity must link to the Project Work Item identity')
    if case_kind in MATERIALIZATION_FAIL_CLOSED_CASES:
        if project_work_identity is not None:
            errors.append(f'{path} fail-closed materialization must not include projectWorkItemIdentity')
        if phase_task_identity is not None:
            errors.append(f'{path} fail-closed materialization must not include phaseTaskIdentity')
        if isinstance(source, dict) and source.get('availability') not in {'missing', 'malformed'}:
            errors.append(f'{path} fail-closed materialization source availability must be missing or malformed')
        mutation = data.get('expectedMutationBehavior')
        if isinstance(mutation, dict):
            if mutation.get('projectWorkItemMutation') != 'none':
                errors.append(f'{path} fail-closed materialization must not mutate Project Work Items')
            if mutation.get('phaseTaskMutation') != 'none':
                errors.append(f'{path} fail-closed materialization must not mutate Phase Tasks')
            if mutation.get('checkedBeforeMutation') is not True:
                errors.append(f'{path} fail-closed materialization must check before mutation')
            if mutation.get('diagnosticEmission') is not True:
                errors.append(f'{path} fail-closed materialization must emit a diagnostic')
        require_object_fields(path, data, 'fail-closed materialization', {'diagnosticCategory', 'recovery'}, errors)
        recovery = data.get('recovery')
        if isinstance(recovery, dict):
            require_object_fields(path, recovery, 'recovery', {'recommendedAction'}, errors)
            recommended_action = recovery.get('recommendedAction')
            if not isinstance(recommended_action, str) or not recommended_action.strip():
                errors.append(f'{path} recovery.recommendedAction must be a non-empty string')
            has_retry_guidance = any(
                key.startswith('safeToRetry') and value is True
                for key, value in recovery.items()
            )
            if not has_retry_guidance:
                errors.append(f'{path} recovery must include safe-to-retry guidance')
        elif 'recovery' in data:
            errors.append(f'{path} recovery must be an object')

    validate_materialization_mutation(path, data, case_kind, project_work_key, phase_task_key, errors)
    validate_downstream_readiness(path, data, errors)
    if case_kind == 'duplicate-phase-task-prevention' and project_work_key and phase_task_key:
        validate_materialization_attempts(path, data, project_work_key, phase_task_key, errors)


def validate_materialization_cross_case(cases: dict[str, dict[str, Any]], errors: list[str]) -> None:
    new_case = cases.get('new-story')
    if new_case is None:
        return
    new_project_key = nested_identity_key(new_case, 'projectWorkItemIdentity')
    new_phase_key = nested_identity_key(new_case, 'phaseTaskIdentity')
    for case_kind in ('unchanged-story', 'changed-story'):
        candidate = cases.get(case_kind)
        if candidate is None:
            continue
        project_key = nested_identity_key(candidate, 'projectWorkItemIdentity')
        phase_key = nested_identity_key(candidate, 'phaseTaskIdentity')
        if project_key != new_project_key:
            errors.append(f'examples/materialization/{case_kind}.json must keep the same Project Work Item identity')
        if phase_key != new_phase_key:
            errors.append(f'examples/materialization/{case_kind}.json must keep the same Phase Task identity')
    unchanged_case = cases.get('unchanged-story')
    if unchanged_case is not None:
        new_source = new_case.get('source')
        unchanged_source = unchanged_case.get('source')
        if isinstance(new_source, dict) and isinstance(unchanged_source, dict) and unchanged_source != new_source:
            errors.append('examples/materialization/unchanged-story.json source must match new-story.json')
    changed_case = cases.get('changed-story')
    if changed_case is not None:
        new_source = new_case.get('source')
        changed_source = changed_case.get('source')
        if changed_source == new_source:
            errors.append('examples/materialization/changed-story.json must change mutable source data')
        if isinstance(new_source, dict) and isinstance(changed_source, dict):
            if changed_source.get('storyTitle') == new_source.get('storyTitle'):
                errors.append('examples/materialization/changed-story.json must cover renamed-story materialization')


def nested_identity_key(case: dict[str, Any], field: str) -> str | None:
    value = case.get(field)
    if not isinstance(value, dict):
        return None
    key = value.get('identityKey')
    return key if isinstance(key, str) else None


def normalized_readme_for_parity(text: str) -> str:
    replacements = {
        'Run this command from the Archon subproject root:': 'Run this command from the parent workspace root:',
        'Run this command from the Hermes Agent subproject root:': 'Run this command from the parent workspace root:',
        'and local package validation without parent workspace traversal.': (
            'and parent-to-local handoff copy parity for owned schemas, examples, README guidance, and validator code.'
        ),
    }
    for before, after in replacements.items():
        text = text.replace(before, after)
    return text


def contract_files_match(rel: str, parent_file: Path, local_file: Path) -> bool:
    if rel == 'README.md':
        return normalized_readme_for_parity(parent_file.read_text(encoding='utf-8')) == normalized_readme_for_parity(
            local_file.read_text(encoding='utf-8')
        )
    return filecmp.cmp(parent_file, local_file, shallow=False)


def validate_parity(parent: Path, locals_: list[Path], errors: list[str]) -> None:
    for package in locals_:
        if not package.exists():
            errors.append(f'local handoff package missing: {package}')
        elif package.is_symlink():
            errors.append(f'local handoff package must not be symlink: {package}')
        elif not package.is_dir():
            errors.append(f'local handoff package must be a directory: {package}')

    for rel in owned_contract_files():
        parent_file = parent / rel
        if not parent_file.exists():
            errors.append(f'parent copy missing: {parent_file}')
            continue
        if parent_file.is_symlink():
            errors.append(f'parent copy must not be symlink: {parent_file}')
            continue
        if not parent_file.is_file():
            errors.append(f'parent copy must be a regular file: {parent_file}')
            continue
        for package in locals_:
            if not package.is_dir():
                continue
            local_file = package / rel
            if not local_file.exists():
                errors.append(f'local copy missing: {local_file}')
                continue
            if local_file.is_symlink():
                errors.append(f'local copy must not be symlink: {local_file}')
                continue
            if not local_file.is_file():
                errors.append(f'local copy must be a regular file: {local_file}')
                continue
            if not contract_files_match(rel, parent_file, local_file):
                errors.append(f'local copy diverges from parent: {local_file}')


def main() -> int:
    root = repo_root()
    parent = parent_package()
    locals_ = local_packages(root) if has_isolated_handoffs(root) else []
    errors: list[str] = []
    schemas: dict[str, dict[str, Any]] = {}

    validate_known_json_files(parent, errors)
    for package in locals_:
        validate_known_json_files(package, errors)

    for rel in REQUIRED_SCHEMA_FILES:
        data = load_json(parent / rel, errors)
        if data is None:
            continue
        expected = {
            'workflow-command-envelope.schema.json': COMMAND_VERSION,
            'workflow-provider-binding.schema.json': BINDING_VERSION,
            'workflow-delivery-status.schema.json': DELIVERY_VERSION,
            'workflow-event-envelope.schema.json': EVENT_VERSION,
            'project-work-identity.schema.json': PROJECT_WORK_IDENTITY_VERSION,
            'phase-task-identity.schema.json': PHASE_TASK_IDENTITY_VERSION,
            'materialization-case.schema.json': MATERIALIZATION_VERSION,
        }[Path(rel).name]
        validate_schema(parent / rel, data, expected, errors)
        check_no_host_specific_paths(parent / rel, data, errors)
        check_no_raw_secrets(parent / rel, data, errors)
        if isinstance(data, dict):
            schemas[Path(rel).name] = data

    for name in REQUIRED_COMMAND_EXAMPLES:
        path = parent / 'examples/providers/archon/commands' / name
        data = load_json(path, errors)
        if data is not None:
            schema = schemas.get('workflow-command-envelope.schema.json')
            if schema is not None:
                validate_against_schema(path, data, schema, errors)
            check_no_host_specific_paths(path, data, errors)
            check_no_raw_secrets(path, data, errors)
            validate_command(path, data, errors)

    for name in REQUIRED_BINDING_EXAMPLES:
        path = parent / 'examples/providers/archon/bindings' / name
        data = load_json(path, errors)
        if data is not None:
            schema = schemas.get('workflow-provider-binding.schema.json')
            if schema is not None:
                validate_against_schema(path, data, schema, errors)
            check_no_host_specific_paths(path, data, errors)
            check_no_raw_secrets(path, data, errors)
            validate_binding(path, data, errors)

    for name in REQUIRED_DELIVERY_EXAMPLES:
        path = parent / 'examples/providers/archon/delivery' / name
        data = load_json(path, errors)
        if data is not None:
            schema = schemas.get('workflow-delivery-status.schema.json')
            if schema is not None:
                validate_against_schema(path, data, schema, errors)
            check_no_host_specific_paths(path, data, errors)
            check_no_raw_secrets(path, data, errors)
            validate_delivery(path, data, errors)

    seen_events_by_id: dict[str, tuple[Path, dict[str, Any]]] = {}
    seen_events_by_idempotency_key: dict[str, tuple[Path, dict[str, Any]]] = {}
    for directory, names in (
        (parent / 'examples/workflow-events', REQUIRED_GENERIC_EVENT_EXAMPLES),
        (parent / 'examples/providers/archon/events', REQUIRED_EVENT_EXAMPLES),
    ):
        for name in names:
            path = directory / name
            data = load_json(path, errors)
            if data is not None:
                schema = schemas.get('workflow-event-envelope.schema.json')
                if schema is not None:
                    validate_against_schema(path, data, schema, errors)
                check_no_host_specific_paths(path, data, errors)
                check_no_raw_secrets(path, data, errors)
                validate_event(path, data, errors)
                check_file_uris(path, parent, data, errors)
                if isinstance(data, dict):
                    event_id = data.get('eventId')
                    idempotency_key = data.get('idempotencyKey')
                    if isinstance(event_id, str) and isinstance(idempotency_key, str):
                        previous_by_id = seen_events_by_id.get(event_id)
                        previous_by_key = seen_events_by_idempotency_key.get(idempotency_key)
                        if previous_by_id is not None or previous_by_key is not None:
                            if previous_by_id is None or previous_by_key is None or previous_by_id[0] != previous_by_key[0]:
                                errors.append(f'{path} must duplicate eventId and idempotencyKey as the same prior event')
                            else:
                                previous_path, previous_event = previous_by_id
                                if not is_valid_redelivery(previous_event, data):
                                    errors.append(f'{path} is not a valid redelivery of {previous_path}')
                        else:
                            seen_events_by_id[event_id] = (path, data)
                            seen_events_by_idempotency_key[idempotency_key] = (path, data)

    for name in REQUIRED_REJECTION_EXAMPLES:
        path = parent / 'examples/callback-rejections' / name
        data = load_json(path, errors)
        if data is not None:
            check_no_host_specific_paths(path, data, errors)
            check_no_raw_secrets(path, data, errors)
            check_file_uris(path, parent, data, errors)
            validate_rejection(path, data, schemas.get('workflow-event-envelope.schema.json'), errors)

    materialization_cases: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_MATERIALIZATION_EXAMPLES:
        path = parent / 'examples/materialization' / name
        data = load_json(path, errors)
        if data is not None:
            check_no_host_specific_paths(path, data, errors)
            check_no_raw_secrets(path, data, errors)
            validate_materialization_case(path, data, schemas, errors)
            if isinstance(data, dict) and isinstance(data.get('caseKind'), str):
                materialization_cases[data['caseKind']] = data
    validate_materialization_cross_case(materialization_cases, errors)

    if locals_:
        validate_parity(parent, locals_, errors)

    if errors:
        print('Workflow Commander 1.3a/1.3b/1.3c contract validation failed:', file=sys.stderr)
        for error in errors:
            print(f'- {error}', file=sys.stderr)
        return 1

    print('Workflow Commander 1.3a/1.3b/1.3c contract validation passed')
    print(f'Validated {len(REQUIRED_SCHEMA_FILES)} schemas')
    print(f'Validated {len(REQUIRED_COMMAND_EXAMPLES)} command examples')
    print(f'Validated {len(REQUIRED_BINDING_EXAMPLES)} binding examples')
    print(f'Validated {len(REQUIRED_DELIVERY_EXAMPLES)} delivery examples')
    print(f'Validated {len(REQUIRED_GENERIC_EVENT_EXAMPLES)} generic event examples')
    print(f'Validated {len(REQUIRED_EVENT_EXAMPLES)} provider event examples')
    print(f'Validated {len(REQUIRED_REJECTION_EXAMPLES)} callback rejection examples')
    print(f'Validated {len(REQUIRED_MATERIALIZATION_EXAMPLES)} materialization examples')
    if locals_:
        print(f'Validated copy parity for {len(owned_contract_files())} contract files across {len(locals_)} local packages')
    else:
        print('Validated isolated local package without parent workspace traversal')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
