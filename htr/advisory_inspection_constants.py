"""Task 29 — advisory artifact/link inspection constants (R5 v1)."""

from __future__ import annotations

PROTOCOL_VERSION = "htr.advisory_inspection.phase29.v1"
SCHEMA_VERSION = "1"

# R5-01 budget / size caps (B-04 aligned)
MAX_CONTROL_JSON_BYTES = 1048576
MAX_RAW_READ_BYTES = MAX_CONTROL_JSON_BYTES + 2
MAX_CONTROL_RECORD_FILE_BYTES = MAX_RAW_READ_BYTES
MAX_MANIFEST_BYTES = MAX_CONTROL_RECORD_FILE_BYTES

MAX_MANIFESTS_PER_AGGREGATE = 64
MAX_ARTIFACT_REFERENCES_PER_MANIFEST = 64
MAX_ARTIFACT_REFERENCES_PER_AGGREGATE = 256
MAX_BYTES_PER_ARTIFACT = 16777216
MAX_TOTAL_BYTES_HASHED = 67108864
MAX_DIRECT_DIRECTORY_ENTRIES_OBSERVED = 256
MAX_LINK_SOURCE_RECORDS = 6
MAX_LINKS_PER_RECORD = 64
MAX_LINKS_PER_AGGREGATE = 128
MAX_PATH_UTF8_BYTES = 4096
MAX_PATH_COMPONENTS = 32
MAX_COMPONENT_UTF8_BYTES = 255
MAX_URL_UTF8_BYTES = 4096
MAX_CONTROL_JSON_DEPTH = 16
MAX_CONTROL_OBJECT_MEMBERS = 64
MAX_CONTROL_ARRAY_LENGTH = 64
MAX_CONTROL_STRING_BYTES = 4096

# B-11 supplemental findings registry (23 tokens; axis scalars excluded).
SUPPLEMENTAL_FINDING_TOKENS: frozenset[str] = frozenset(
    {
        # manifest (2)
        "manifest_unknown_field_observed",
        "manifest_references_not_processed_budget",
        # reference / path (3)
        "reference_same_path_distinct_kind",
        "path_nfc_not_normalized",
        "path_surrogate_rejected",
        # link URL / fetch (8)
        "link_remote_reference_not_fetched",
        "link_reachability_not_inspected",
        "link_content_identity_not_verified",
        "link_http_cleartext_risk",
        "link_credentials_prohibited",
        "link_backslash_rejected",
        "link_control_character_rejected",
        "link_malformed_percent_escape",
        # link host (6)
        "link_host_unicode_observed",
        "link_host_alabel_observed",
        "link_host_trailing_dot_observed",
        "link_host_ipv6_literal",
        "link_host_ipv4_mapped_ipv6",
        "link_primary_derived_conflict",
        # link structure (4)
        "link_query_observed",
        "link_fragment_observed",
        "link_ambiguous_authority",
        "link_percent_encoded_traversal_observed",
    }
)

# Fixed derived alignment slot order (B-02)
DERIVED_ALIGNMENT_ROLES: tuple[str, ...] = ("1a", "1b", "2a", "2b")

# Path B link discovery — six closed control-record filenames (R5-11)
LINK_SOURCE_RECORD_FILENAMES: frozenset[str] = frozenset(
    {
        "run_execution_request_record.json",
        "run_post_verification_execution_request_record.json",
        "run_execution_result_record.json",
        "run_execution_verification_record.json",
        "run_post_verification_execution_result_record.json",
        "run_post_verification_execution_verification_record.json",
    }
)
