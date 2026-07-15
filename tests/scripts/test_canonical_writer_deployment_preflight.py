from __future__ import annotations

import copy
import json

import pytest

from gateway.canonical_writer_db import (
    CANONICAL_EVENT_LOG_COLUMNS,
    CANONICAL_EVENT_LOG_OWNER,
    CANONICAL_EVENT_LOG_TABLE,
    CANONICAL_PRIVATE_WRITER_TABLES,
    CanonicalPrivateRelationIdentity,
    CanonicalPrivateSchemaIdentity,
    managed_cloudsqladmin_hba_receipt_from_mapping,
)
from gateway.canonical_writer_postgres_backend import (
    CANONICAL_WRITER_MIGRATION_OWNER,
    CANONICAL_WRITER_ROLE,
    EXPECTED_HELPER_ROUTINE_SIGNATURES,
    EXPECTED_ROUTINE_SIGNATURES,
)
from gateway import canonical_writer_deployment_preflight as preflight


_WRITER_REVISION = "1" * 40
_WRITER_ARTIFACT_DIGEST = "2" * 64
_WRITER_ARTIFACT_ROOT = f"/opt/muncho-canonical-writer/releases/{_WRITER_REVISION}"
_WRITER_INTERPRETER = f"{_WRITER_ARTIFACT_ROOT}/venv/bin/python3.12"
_WRITER_MODULE_ORIGIN = (
    f"{_WRITER_ARTIFACT_ROOT}/venv/lib/python3.12/site-packages/"
    "gateway/canonical_writer_bootstrap.py"
)
_WRITER_CONFIG = "/etc/muncho-canonical-writer/writer.json"
_WRITER_RUNTIME = "/run/muncho-canonical-writer"
_WRITER_EXPORT = "/var/lib/muncho-canonical-writer/projection"
_GATEWAY_MODULE_ORIGIN = (
    f"{_WRITER_ARTIFACT_ROOT}/venv/lib/python3.12/site-packages/gateway/"
    "canonical_writer_gateway_bootstrap.py"
)
_WRITER_NATIVE_PATH = "/usr/lib/muncho-reviewed/native-writer.so"
_WRITER_NATIVE_SHA256 = "6" * 64
_GATEWAY_NATIVE_PATH = "/usr/lib/muncho-reviewed/native-gateway.so"
_GATEWAY_NATIVE_SHA256 = "7" * 64


def _canonical_non_owner_acl_grants():
    return sorted([
        (
            "schema:canonical_brain::canonical_brain_migration_owner:"
            "canonical_brain_writer:USAGE:f"
        ),
        *(
            "function:" + signature
            + "::canonical_brain_migration_owner:canonical_brain_writer:EXECUTE:f"
            for signature in EXPECTED_ROUTINE_SIGNATURES
        ),
    ])


def _denied_mutation():
    return {
        "create_child": False,
        "rename": False,
        "replace": False,
        "write": False,
    }


def _immutable_parent_chain():
    return {
        "root_owned": True,
        "symlink_free": True,
        "mount_read_only": True,
        "gateway_mutation_access": _denied_mutation(),
        "gateway_child_mutation_access": _denied_mutation(),
    }


def _protected_path(policy):
    return {
        **policy,
        "exists": True,
        "owner_uid": 0,
        "mode": "0555",
        "symlink": False,
        "immutable": True,
        "mount_read_only": True,
        "writer_access": {"read": True, "write": False, "execute": True},
        "gateway_mutation_access": _denied_mutation(),
        "gateway_child_mutation_access": _denied_mutation(),
        "gateway_open_write_fds": [],
        "gateway_child_open_write_fds": [],
        "gateway_writable_mappings": [],
        "gateway_child_writable_mappings": [],
        "parent_chain": _immutable_parent_chain(),
    }


def _writer_deployment():
    import_paths = [
        {
            "path": _WRITER_ARTIFACT_ROOT,
            "kind": "application",
            "digest_sha256": _WRITER_ARTIFACT_DIGEST,
            "object_type": "directory",
        },
        {
            "path": _WRITER_INTERPRETER,
            "kind": "interpreter",
            "digest_sha256": "3" * 64,
            "object_type": "regular_file",
        },
        {
            "path": f"{_WRITER_ARTIFACT_ROOT}/venv/lib/python3.12",
            "kind": "stdlib",
            "digest_sha256": "4" * 64,
            "object_type": "directory",
        },
        {
            "path": (f"{_WRITER_ARTIFACT_ROOT}/venv/lib/python3.12/site-packages"),
            "kind": "site_packages",
            "digest_sha256": "5" * 64,
            "object_type": "directory",
        },
    ]
    exec_start = [
        _WRITER_INTERPRETER,
        "-B",
        "-I",
        "-m",
        "gateway.canonical_writer_bootstrap",
        "--config",
        _WRITER_CONFIG,
    ]
    read_write_paths = [_WRITER_RUNTIME, _WRITER_EXPORT]
    return {
        "policy": {
            "unit_name": "muncho-canonical-writer.service",
            "artifact_root": _WRITER_ARTIFACT_ROOT,
            "revision": _WRITER_REVISION,
            "artifact_digest_sha256": _WRITER_ARTIFACT_DIGEST,
            "interpreter": _WRITER_INTERPRETER,
            "module": "gateway.canonical_writer_bootstrap",
            "module_origin": _WRITER_MODULE_ORIGIN,
            "config_path": _WRITER_CONFIG,
            "exec_start": exec_start,
            "working_directory": _WRITER_ARTIFACT_ROOT,
            "import_paths": import_paths,
            "runtime_directory": _WRITER_RUNTIME,
            "projection_export_directory": _WRITER_EXPORT,
            "read_write_paths": read_write_paths,
            "bind_paths": [],
            "bind_read_only_paths": [_WRITER_ARTIFACT_ROOT],
            "preapproved_external_native_executable_mappings": [
                {
                    "path": _WRITER_NATIVE_PATH,
                    "sha256": _WRITER_NATIVE_SHA256,
                }
            ],
            "preapproved_kernel_executable_mappings": ["[vdso]", "[vsyscall]"],
        },
        "attestation": {
            "complete": True,
            "unit": {
                "name": "muncho-canonical-writer.service",
                "exec_start": exec_start,
                "interpreter": _WRITER_INTERPRETER,
                "module": "gateway.canonical_writer_bootstrap",
                "config_path": _WRITER_CONFIG,
                "working_directory": _WRITER_ARTIFACT_ROOT,
                "revision": _WRITER_REVISION,
                "artifact_digest_sha256": _WRITER_ARTIFACT_DIGEST,
                "alternate_exec_commands": [],
                "environment_files": [],
                "code_injection_environment_variable_names": [],
                "environment_pythonpath": [],
                "environment_pythonhome": "",
                "user_uid": 1002,
                "group_gid": 2002,
            },
            "artifact": {
                **_protected_path(import_paths[0]),
                "revision": _WRITER_REVISION,
            },
            "import_closure": {
                "complete": True,
                "paths": [_protected_path(item) for item in import_paths],
            },
            "process": {
                "complete": True,
                "observed_at_unix": 1_800_000_000,
                "pid": 4343,
                "systemd_main_pid": 4343,
                "process_start_time_ticks": 654321,
                "systemd_main_pid_start_time_ticks": 654321,
                "unit_name": "muncho-canonical-writer.service",
                "cmdline": exec_start,
                "executable_path": _WRITER_INTERPRETER,
                "executable_digest_sha256": "3" * 64,
                "revision": _WRITER_REVISION,
                "artifact_digest_sha256": _WRITER_ARTIFACT_DIGEST,
                "bootstrap_module_origin": _WRITER_MODULE_ORIGIN,
                "effective_import_paths": sorted([
                    import_paths[2]["path"],
                    import_paths[3]["path"],
                ]),
                "loaded_module_origins_complete": True,
                "loaded_module_origins": [_WRITER_MODULE_ORIGIN],
                "mapped_executable_paths_complete": True,
                "mapped_executable_paths": sorted(
                    [_WRITER_INTERPRETER, _WRITER_NATIVE_PATH]
                ),
                "kernel_executable_mappings": ["[vdso]", "[vsyscall]"],
                "unexpected_import_origins": [],
                "deleted_code_mappings": [],
                "writable_code_mappings": [],
            },
            "mounts": {
                "complete": True,
                "read_write_paths": read_write_paths,
                "bind_paths": [],
                "bind_read_only_paths": [_WRITER_ARTIFACT_ROOT],
            },
        },
    }


def _gateway_deployment(writer_deployment):
    policy = writer_deployment["policy"]
    exec_start = [
        _WRITER_INTERPRETER,
        "-B",
        "-I",
        "-m",
        "gateway.canonical_writer_gateway_bootstrap",
    ]
    read_write_paths = ["/run/hermes-cloud-gateway"]
    return {
        "policy": {
            "unit_name": "hermes-cloud-gateway.service",
            "artifact_root": _WRITER_ARTIFACT_ROOT,
            "revision": _WRITER_REVISION,
            "artifact_digest_sha256": _WRITER_ARTIFACT_DIGEST,
            "interpreter": _WRITER_INTERPRETER,
            "module": "gateway.canonical_writer_gateway_bootstrap",
            "module_origin": _GATEWAY_MODULE_ORIGIN,
            "exec_start": exec_start,
            "working_directory": _WRITER_ARTIFACT_ROOT,
            "import_paths": copy.deepcopy(policy["import_paths"]),
            "dynamic_python_loading_mode": "disabled",
            "dynamic_python_discovery_paths": [],
            "read_write_paths": read_write_paths,
            "bind_paths": [],
            "bind_read_only_paths": [_WRITER_ARTIFACT_ROOT],
            "preapproved_external_native_executable_mappings": [
                {
                    "path": _GATEWAY_NATIVE_PATH,
                    "sha256": _GATEWAY_NATIVE_SHA256,
                }
            ],
            "preapproved_kernel_executable_mappings": ["[vdso]", "[vsyscall]"],
        },
        "attestation": {
            "complete": True,
            "unit": {
                "name": "hermes-cloud-gateway.service",
                "exec_start": exec_start,
                "interpreter": _WRITER_INTERPRETER,
                "module": "gateway.canonical_writer_gateway_bootstrap",
                "module_origin": _GATEWAY_MODULE_ORIGIN,
                "working_directory": _WRITER_ARTIFACT_ROOT,
                "revision": _WRITER_REVISION,
                "artifact_digest_sha256": _WRITER_ARTIFACT_DIGEST,
                "alternate_exec_commands": [],
                "environment_files": [],
                "code_injection_environment_variable_names": [],
                "environment_pythonpath": [],
                "environment_pythonhome": "",
                "user_uid": 1001,
                "group_gid": 2000,
            },
            "process": {
                "complete": True,
                "observed_at_unix": 1_800_000_000,
                "pid": 4242,
                "systemd_main_pid": 4242,
                "process_start_time_ticks": 123456,
                "systemd_main_pid_start_time_ticks": 123456,
                "unit_name": "hermes-cloud-gateway.service",
                "cmdline": exec_start,
                "executable_path": _WRITER_INTERPRETER,
                "executable_digest_sha256": "3" * 64,
                "entry_module_origin": _GATEWAY_MODULE_ORIGIN,
                "revision": _WRITER_REVISION,
                "artifact_digest_sha256": _WRITER_ARTIFACT_DIGEST,
                "effective_import_paths": sorted(
                    [
                        policy["import_paths"][2]["path"],
                        policy["import_paths"][3]["path"],
                    ]
                ),
                "loaded_module_origins_complete": True,
                "loaded_module_origins": [_GATEWAY_MODULE_ORIGIN],
                "mapped_executable_paths_complete": True,
                "mapped_executable_paths": sorted(
                    [_WRITER_INTERPRETER, _GATEWAY_NATIVE_PATH]
                ),
                "kernel_executable_mappings": ["[vdso]", "[vsyscall]"],
                "unexpected_import_origins": [],
                "deleted_code_mappings": [],
                "writable_code_mappings": [],
                "dynamic_python_discovery_complete": True,
                "dynamic_python_loading_mode": "disabled",
                "dynamic_python_discovery_paths": [],
                "dynamic_python_loaded_origins": [],
                "dynamic_python_writable_paths": [],
            },
            "mounts": {
                "complete": True,
                "read_write_paths": read_write_paths,
                "bind_paths": [],
                "bind_read_only_paths": [_WRITER_ARTIFACT_ROOT],
            },
        },
    }


def _denied_local_authority():
    return {
        "can_manage_writer_units": False,
        "can_create_transient_units": False,
        "can_manage_timers": False,
        "can_manage_cron": False,
        "can_switch_to_writer_uid": False,
        "can_invoke_polkit_actions": False,
        "can_write_systemd_unit_paths": False,
        "can_write_cron_paths": False,
        "authorized_systemd_dbus_actions": [],
        "authorized_polkit_actions": [],
        "authorized_sudo_commands": [],
        "authorized_doas_commands": [],
        "effective_capabilities": [],
        "writable_systemd_unit_paths": [],
        "writable_cron_paths": [],
    }


def _writer_authority_surface(writer_deployment):
    policy = writer_deployment["policy"]
    export_path = f"{_WRITER_EXPORT}/canonical-events.json"
    export_limit = 200_000
    export_exec = [
        _WRITER_INTERPRETER,
        "-B",
        "-I",
        "-m",
        "gateway.canonical_writer_bootstrap",
        "--config",
        _WRITER_CONFIG,
        "--export-events",
        export_path,
        "--export-limit",
        str(export_limit),
    ]
    return {
        "complete": True,
        "collected_by_uid": 0,
        "observed_at_unix": 1_800_000_000,
        "identities": {
            "gateway": {
                "pid": 4242,
                "process_start_time_ticks": 123456,
                "effective_uid": 1001,
                "effective_gid": 2000,
                "supplementary_gids": [2000, 2001],
                "no_new_privileges": True,
                "effective_capabilities": [],
                "executable": _WRITER_INTERPRETER,
            },
            "gateway_children": {"complete": True, "processes": []},
            "writer": {
                "pid": 4343,
                "process_start_time_ticks": 654321,
                "effective_uid": 1002,
                "effective_gid": 2002,
                "supplementary_gids": [2002, 2003],
                "no_new_privileges": True,
                "effective_capabilities": [],
                "executable": _WRITER_INTERPRETER,
            },
        },
        "group_policy": {
            "complete": True,
            "evaluated_dangerous_group_names": list(
                preflight._DANGEROUS_LOCAL_GROUP_NAMES
            ),
            "gateway_dangerous_memberships": [],
            "gateway_child_dangerous_memberships": [],
            "writer_dangerous_memberships": [],
            "unknown_privileged_gids": [],
            "gateway_account_gids": [2000, 2001],
            "writer_account_gids": [2002, 2003],
            "protected_group_memberships": {
                "2000": ["muncho-gateway"],
                "2001": ["muncho-gateway"],
                "2002": ["muncho-canonical-writer"],
                "2003": [
                    "muncho-canonical-writer",
                    "muncho-projector",
                ],
            },
            "projector_identity": {
                "uid": 992,
                "gid": 2003,
                "home": "/nonexistent",
                "shell": "/usr/sbin/nologin",
                "gids": [2003],
                "process_pids": [],
            },
        },
        "gateway_authority": _denied_local_authority(),
        "gateway_child_authority": _denied_local_authority(),
        "writer_authority": _denied_local_authority(),
        "privileged_execution_inventory": {
            "complete": True,
            "writer_uid_service_units": ["muncho-canonical-writer.service"],
            "writer_uid_timer_units": [],
            "writer_uid_transient_units": [],
            "writer_uid_cron_entries": [],
            "writer_uid_at_jobs": [],
            "writer_uid_process_executables": [_WRITER_INTERPRETER],
            "writer_uid_unattributed_processes": [],
            "writer_unit_reverse_activation": {
                "unit_name": "muncho-canonical-writer.service",
                "triggered_by": [],
                "wanted_by": [],
                "required_by": [],
                "bound_by": ["hermes-cloud-gateway.service"],
                "upheld_by": [],
                "requisite_of": [],
                "on_success_of": [],
                "on_failure_of": [],
                "reverse_references": ["hermes-cloud-gateway.service"],
                "service_units": ["hermes-cloud-gateway.service"],
                "timer_units": [],
                "socket_units": [],
                "path_units": [],
                "target_units": [],
                "automount_units": [],
                "other_units": [],
                "transient_units": [],
            },
            "gateway_writable_writer_unit_files": [],
            "gateway_child_writable_writer_unit_files": [],
            "gateway_uid_service_units": ["hermes-cloud-gateway.service"],
            "gateway_uid_timer_units": [],
            "gateway_uid_transient_units": [],
            "gateway_uid_cron_entries": [],
            "gateway_uid_at_jobs": [],
            "gateway_uid_process_executables": [_WRITER_INTERPRETER],
            "gateway_uid_unattributed_processes": [],
            "gateway_unit_reverse_activation": {
                "unit_name": "hermes-cloud-gateway.service",
                "triggered_by": [],
                "wanted_by": [],
                "required_by": [],
                "bound_by": [],
                "upheld_by": [],
                "requisite_of": [],
                "on_success_of": [],
                "on_failure_of": [],
                "reverse_references": [],
                "service_units": [],
                "timer_units": [],
                "socket_units": [],
                "path_units": [],
                "target_units": [],
                "automount_units": [],
                "other_units": [],
                "transient_units": [],
            },
        },
        "projection_exporter": {
            "policy": {
                "enabled": False,
                "unit_name": "muncho-canonical-writer-export.service",
                "timer_name": "muncho-canonical-writer-export.timer",
                "artifact_root": _WRITER_ARTIFACT_ROOT,
                "revision": _WRITER_REVISION,
                "artifact_digest_sha256": _WRITER_ARTIFACT_DIGEST,
                "interpreter": _WRITER_INTERPRETER,
                "module": "gateway.canonical_writer_bootstrap",
                "module_origin": _WRITER_MODULE_ORIGIN,
                "config_path": _WRITER_CONFIG,
                "export_path": export_path,
                "export_limit": export_limit,
                "exec_start": export_exec,
                "read_write_paths": [_WRITER_EXPORT],
                "bind_paths": [],
                "bind_read_only_paths": [_WRITER_ARTIFACT_ROOT],
                "timer_schedule": {
                    "OnCalendar": "*-*-* *:*:00",
                    "Persistent": False,
                },
            },
            "attestation": {
                "complete": True,
                "enabled": False,
                "unit": {},
                "timer": {},
            },
        },
        "user_systemd": {
            "complete": True,
            "gateway": {
                "uid": 1001,
                "linger_path": "/var/lib/systemd/linger/muncho-gateway",
                "linger_enabled": False,
                "user_manager_unit": "user@1001.service",
                "load_state": "loaded",
                "active_state": "inactive",
                "sub_state": "dead",
                "main_pid": 0,
                "runtime_directory_exists": False,
                "private_socket_exists": False,
                "home_user_unit_path": "/var/lib/hermes-gateway/.config/systemd/user",
                "home_user_unit_path_exists": False,
                "home_user_unit_path_service_writable": False,
                "home_directory_exists": True,
                "evaluated_home_user_unit_paths": [
                    "/var/lib/hermes-gateway/.config/systemd/user",
                    "/var/lib/hermes-gateway/.local/share/systemd/user",
                    "/var/lib/hermes-gateway/.config/systemd/user-generators",
                    "/var/lib/hermes-gateway/.local/share/systemd/user-generators",
                    "/var/lib/hermes-gateway/.config/systemd/user-environment-generators",
                    "/var/lib/hermes-gateway/.local/share/systemd/user-environment-generators",
                ],
                "existing_home_user_unit_paths": [],
                "service_writable_home_user_unit_paths": [],
                "service_units": [],
                "timer_units": [],
                "activation_units": [],
                "transient_units": [],
                "runtime_service_units": [],
                "runtime_timer_units": [],
                "runtime_activation_units": [],
                "global_service_units": [],
                "global_timer_units": [],
                "global_activation_units": [],
                "global_generators": [],
                "evaluated_global_unit_roots": list(
                    preflight._GLOBAL_USER_SYSTEMD_UNIT_ROOTS
                ),
                "existing_global_unit_roots": [
                    "/etc/systemd/user",
                    "/usr/lib/systemd/user",
                ],
                "evaluated_global_generator_roots": list(
                    preflight._GLOBAL_USER_SYSTEMD_GENERATOR_ROOTS
                ),
                "existing_global_generator_roots": [],
                "global_directories_protected": True,
            },
            "writer": {
                "uid": 1002,
                "linger_path": "/var/lib/systemd/linger/muncho-canonical-writer",
                "linger_enabled": False,
                "user_manager_unit": "user@1002.service",
                "load_state": "loaded",
                "active_state": "inactive",
                "sub_state": "dead",
                "main_pid": 0,
                "runtime_directory_exists": False,
                "private_socket_exists": False,
                "home_user_unit_path": "/nonexistent/.config/systemd/user",
                "home_user_unit_path_exists": False,
                "home_user_unit_path_service_writable": False,
                "home_directory_exists": False,
                "evaluated_home_user_unit_paths": [
                    "/nonexistent/.config/systemd/user",
                    "/nonexistent/.local/share/systemd/user",
                    "/nonexistent/.config/systemd/user-generators",
                    "/nonexistent/.local/share/systemd/user-generators",
                    "/nonexistent/.config/systemd/user-environment-generators",
                    "/nonexistent/.local/share/systemd/user-environment-generators",
                ],
                "existing_home_user_unit_paths": [],
                "service_writable_home_user_unit_paths": [],
                "service_units": [],
                "timer_units": [],
                "activation_units": [],
                "transient_units": [],
                "runtime_service_units": [],
                "runtime_timer_units": [],
                "runtime_activation_units": [],
                "global_service_units": [],
                "global_timer_units": [],
                "global_activation_units": [],
                "global_generators": [],
                "evaluated_global_unit_roots": list(
                    preflight._GLOBAL_USER_SYSTEMD_UNIT_ROOTS
                ),
                "existing_global_unit_roots": [
                    "/etc/systemd/user",
                    "/usr/lib/systemd/user",
                ],
                "evaluated_global_generator_roots": list(
                    preflight._GLOBAL_USER_SYSTEMD_GENERATOR_ROOTS
                ),
                "existing_global_generator_roots": [],
                "global_directories_protected": True,
            },
        },
    }


def _identity(signature, *, helper=False):
    return {
        "signature": signature,
        "owner": CANONICAL_WRITER_MIGRATION_OWNER,
        "security_definer": not helper,
        "language": "sql" if helper else "plpgsql",
        "configuration": ["search_path=pg_catalog, canonical_brain"],
        "definition_sha256": ("b" if helper else "a") * 64,
        "owner_dangerous": False,
    }


def _routine_identities():
    return [_identity(signature) for signature in EXPECTED_ROUTINE_SIGNATURES]


def _helper_identities():
    return [
        _identity(signature, helper=True)
        for signature in EXPECTED_HELPER_ROUTINE_SIGNATURES
    ]


def _private_relation_identity(name):
    return {
        "name": name,
        "owner": CANONICAL_WRITER_MIGRATION_OWNER,
        "owner_dangerous": False,
        "relation_kind": "r",
        "persistence": "p",
        "is_partition": False,
        "access_method": "heap",
        "tablespace_oid": 0,
        "row_security": False,
        "force_row_security": False,
        "replica_identity": "d",
        "relation_options": [],
        "columns": [json.dumps({"name": "identity"}, separators=(",", ":"))],
        "constraints": [json.dumps({"type": "p"}, separators=(",", ":"))],
        "indexes": [json.dumps({"primary": True}, separators=(",", ":"))],
        "index_owners": [f"{CANONICAL_WRITER_MIGRATION_OWNER}:f"],
        "user_triggers": [],
        "rewrite_rules": [],
        "policies": [],
        "inheritance": False,
    }


def _private_schema_identity():
    return {
        "schema": "canonical_brain",
        "owner": CANONICAL_WRITER_MIGRATION_OWNER,
        "owner_dangerous": False,
        "relations": [
            _private_relation_identity(name)
            for name in CANONICAL_PRIVATE_WRITER_TABLES
        ],
    }


def _private_schema_identity_sha256(identity):
    return CanonicalPrivateSchemaIdentity(
        schema=identity["schema"],
        owner=identity["owner"],
        owner_dangerous=identity["owner_dangerous"],
        relations=tuple(
            CanonicalPrivateRelationIdentity(**relation)
            for relation in identity["relations"]
        ),
    ).sha256


def _managed_hba_receipt():
    return {
        "version": "managed-cloudsqladmin-hba-rejection-v2",
        "host": "10.0.0.8",
        "tls_server_name": "db.internal",
        "port": 5432,
        "server_certificate_sha256": "d" * 64,
        "database": "cloudsqladmin",
        "user": "canonical_writer",
        "observed_at_unix": 1_799_999_990,
        "expires_at_unix": 1_800_000_200,
        "sqlstate": "28000",
        "server_message": (
            'no pg_hba.conf entry for host "10.0.0.8", user '
            '"canonical_writer", database "cloudsqladmin", SSL encryption'
        ),
        "result": "pg_hba_rejected",
        "tls_peer_verified": True,
    }


def _good_snapshot():
    writer_deployment = _writer_deployment()
    hardened = {name: "yes" for name in preflight._HARDENED_TRUE_PROPERTIES}
    hardened.update({
        "ProtectSystem": "strict",
        "ProtectHome": "yes",
        "ProtectProc": "invisible",
        "ProcSubset": "pid",
        "UMask": "0077",
        "CapabilityBoundingSet": "",
        "AmbientCapabilities": "",
        "LimitCORE": "0",
        "RestrictAddressFamilies": "AF_UNIX AF_INET AF_INET6",
    })
    private_schema_identity = _private_schema_identity()
    managed_hba_receipt = _managed_hba_receipt()
    managed_hba_digest = managed_cloudsqladmin_hba_receipt_from_mapping(
        managed_hba_receipt
    ).sha256
    return {
        "deployment_mode": "writer_only",
        "collected_at_unix": 1_800_000_000,
        "gateway_uid": 1001,
        "gateway_gid": 2000,
        "writer_uid": 1002,
        "writer_gid": 2002,
        "projector_gid": 2003,
        "gateway_supplementary_gids": [2000, 2001],
        "writer_supplementary_gids": [2002, 2003],
        "gateway_process": {
            "complete": True,
            "platform": "linux",
            "pid": 4242,
            "systemd_main_pid": 4242,
            "process_start_time_ticks": 123456,
            "systemd_main_pid_start_time_ticks": 123456,
            "observed_at_unix": 1_800_000_000,
            "dumpable": False,
            "core_soft_limit": 0,
            "core_hard_limit": 0,
        },
        "runtime_secret_sources": {
            "complete": True,
            "legacy_hermes_env": {
                "path": "/home/muncho/.hermes/.env",
                "exists": False,
                "gateway_file_access": {
                    "read": False,
                    "write": False,
                    "execute": False,
                },
                "gateway_child_file_access": {
                    "read": False,
                    "write": False,
                    "execute": False,
                },
            },
            "gateway_readable_secret_files": [],
            "gateway_child_readable_secret_files": [],
            "pgpass": {
                "path": "/home/muncho/.pgpass",
                "exists": False,
                "gateway_file_access": {
                    "read": False,
                    "write": False,
                    "execute": False,
                },
                "gateway_child_file_access": {
                    "read": False,
                    "write": False,
                    "execute": False,
                },
            },
            "gateway_readable_database_credential_files": [],
            "gateway_child_readable_database_credential_files": [],
            "gateway_readable_systemd_environment_files": [],
            "gateway_child_readable_systemd_environment_files": [],
            "open_database_credential_fds": [],
            "inherited_database_credential_fds": [],
            "cloud_sql_unix_socket": {
                "path": "/cloudsql",
                "gateway_access": {
                    "read": False,
                    "write": False,
                    "execute": False,
                },
                "gateway_child_access": {
                    "read": False,
                    "write": False,
                    "execute": False,
                },
                "open_fds": [],
                "inherited_fds": [],
            },
            "effective_gateway_env": {
                "complete": True,
                "values_included": False,
                "database_password_variable_names": [],
                "database_connection_secret_variable_names": [],
            },
            "sources": [
                {
                    "name": "canonical_writer_database_password",
                    "provisioned_by": "systemd",
                    "gateway_file_access": {
                        "read": False,
                        "write": False,
                        "execute": False,
                    },
                    "gateway_child_file_access": {
                        "read": False,
                        "write": False,
                        "execute": False,
                    },
                }
            ],
        },
        "discord_edge": {
            "complete": True,
            "collected_by_uid": 0,
            "observed_at_unix": 1_800_000_000,
            "gateway_enabled": False,
            "writer_authority_enabled": False,
            "unit_name": "muncho-discord-egress.service",
            "unit_exists": False,
            "unit_enabled": False,
            "unit_active": False,
            "main_pid": 0,
            "config_path": "/etc/muncho/discord-edge.json",
            "config_exists": False,
            "token_path": "/etc/muncho/discord-edge-credentials/bot-token",
            "token_exists": False,
            "socket_path": "/run/muncho-discord-egress/edge.sock",
            "socket_exists": False,
            "process_pids": [],
        },
        "helper": {
            "exists": False,
            "gateway_access": {"read": False, "write": False, "execute": False},
        },
        "socket": {
            "owner_uid": 1002,
            "group_gid": 2001,
            "expected_group_gid": 2001,
            "mode": "0660",
        },
        "credential": {
            "owner_uid": 1002,
            "mode": "0400",
            "gateway_access": {"read": False, "write": False, "execute": False},
        },
        "projection_export": {
            "owner_uid": 1002,
            "group_gid": 2003,
            "mode": "0640",
            "gateway_access": {"read": False, "write": False, "execute": False},
            "projector_access": {"read": True, "write": False, "execute": False},
        },
        "writer_deployment": writer_deployment,
        "gateway_deployment": _gateway_deployment(writer_deployment),
        "writer_authority_surface": _writer_authority_surface(writer_deployment),
        "systemd_properties": hardened,
        "gateway_systemd_properties": copy.deepcopy(hardened),
        "gateway_iam": {"complete": True, "roles": [], "permissions": []},
        "writer_iam": {"complete": True, "roles": [], "permissions": []},
        "database": {
            "expected_user": "canonical_writer",
            "connection": {
                "host": "10.0.0.8",
                "tls_server_name": "db.internal",
                "port": 5432,
                "database": "canonical",
                "user": "canonical_writer",
            },
            "managed_cloudsqladmin_hba_rejection_evidence": {
                "complete": True,
                "collector_uid": 0,
                "source_owner_uid": 0,
                "source_mode": "0400",
                "source_symlink": False,
                "same_host": True,
                "same_tls_server_name": True,
                "same_port": True,
                "same_ca": True,
                "same_user": True,
                "same_credential": True,
                "receipt_sha256": managed_hba_digest,
                "receipt": copy.deepcopy(managed_hba_receipt),
            },
            "policy": {
                "schema": "canonical_brain",
                "table_grants": [],
                "sequence_grants": [],
                "executable_routines": list(EXPECTED_ROUTINE_SIGNATURES),
                "routine_identities": _routine_identities(),
                "helper_routine_identities": _helper_identities(),
                "schema_privileges": ["USAGE"],
                "database_privileges": ["CONNECT"],
                "role_memberships": [CANONICAL_WRITER_ROLE],
                "private_schema_identity_sha256": (
                    _private_schema_identity_sha256(private_schema_identity)
                ),
                "managed_cloudsqladmin_hba_rejection_receipt": (
                    copy.deepcopy(managed_hba_receipt)
                ),
                "managed_cloudsqladmin_hba_rejection_sha256": managed_hba_digest,
            },
            "attestation": {
                "managed_cloudsqladmin_hba_rejection_sha256": managed_hba_digest,
                "role": "canonical_writer",
                "superuser": False,
                "createdb": False,
                "createrole": False,
                "replication": False,
                "bypassrls": False,
                "table_owner": False,
                "routine_owner": False,
                "table_grants": [],
                "sequence_grants": [],
                "executable_routines": list(EXPECTED_ROUTINE_SIGNATURES),
                "routine_identities": _routine_identities(),
                "helper_routine_identities": _helper_identities(),
                "schema_privileges": ["USAGE"],
                "database_privileges": ["CONNECT"],
                "role_memberships": [CANONICAL_WRITER_ROLE],
                "unexpected_privileges": [],
                "public_acl_grants": [],
                "canonical_non_owner_acl_grants": (_canonical_non_owner_acl_grants()),
                "canonical_writer_role_inheritors": ["canonical_writer:1:t:f"],
                "canonical_event_log_identity": {
                    "table": CANONICAL_EVENT_LOG_TABLE,
                    "owner": CANONICAL_EVENT_LOG_OWNER,
                    "owner_dangerous": False,
                    "relation_kind": "r",
                    "persistence": "p",
                    "is_partition": False,
                    "access_method": "heap",
                    "tablespace_oid": 0,
                    "row_security": False,
                    "force_row_security": False,
                    "replica_identity": "d",
                    "relation_options": [],
                    "columns": list(CANONICAL_EVENT_LOG_COLUMNS),
                    "constraints": ["PRIMARY KEY (event_id)"],
                    "user_triggers": [],
                    "rewrite_rules": [],
                    "policies": [],
                    "inheritance": False,
                    "non_owner_acl_grants": [],
                    "index_count": 1,
                    "primary_index_exact": True,
                },
                "canonical_private_schema_identity": private_schema_identity,
            },
        },
    }


def _enable_projection_exporter(snapshot):
    surface = snapshot["writer_authority_surface"]
    exporter = surface["projection_exporter"]
    policy = exporter["policy"]
    policy["enabled"] = True
    exporter["attestation"] = {
        "complete": True,
        "enabled": True,
        "unit": {
            "name": policy["unit_name"],
            "type": "oneshot",
            "exec_start": copy.deepcopy(policy["exec_start"]),
            "user_uid": 1002,
            "group_gid": 2002,
            "artifact_root": policy["artifact_root"],
            "revision": policy["revision"],
            "artifact_digest_sha256": policy["artifact_digest_sha256"],
            "module_origin": policy["module_origin"],
            "read_write_paths": copy.deepcopy(policy["read_write_paths"]),
            "bind_paths": [],
            "bind_read_only_paths": copy.deepcopy(policy["bind_read_only_paths"]),
            "alternate_exec_commands": [],
            "environment_files": [],
            "code_injection_environment_variable_names": [],
            "systemd_properties": copy.deepcopy(snapshot["systemd_properties"]),
        },
        "timer": {
            "name": policy["timer_name"],
            "unit_name": policy["unit_name"],
            "enabled": True,
            "schedule": copy.deepcopy(policy["timer_schedule"]),
        },
    }
    inventory = surface["privileged_execution_inventory"]
    inventory["writer_uid_service_units"] = sorted([
        "muncho-canonical-writer.service",
        policy["unit_name"],
    ])
    inventory["writer_uid_timer_units"] = [policy["timer_name"]]


def _failed_names(snapshot):
    return {
        check.name
        for check in preflight.evaluate_snapshot(snapshot).checks
        if not check.passed
    }


def test_good_snapshot_passes_all_invariants():
    report = preflight.evaluate_snapshot(_good_snapshot())

    assert report.ok
    assert report.to_dict()["ok"] is True
    assert all(check.passed for check in report.checks)


@pytest.mark.parametrize(
    ("deployment_name", "policy_failure", "closure_failure"),
    [
        (
            "writer_deployment",
            "writer_deployment.policy_valid",
            "writer_deployment.process_code_closure",
        ),
        (
            "gateway_deployment",
            "gateway_deployment.shared_immutable_release",
            "gateway_deployment.process_code_closure",
        ),
    ],
)
@pytest.mark.parametrize(
    "case",
    [
        "missing_policy",
        "empty_policy",
        "extra_entry_field",
        "invalid_hash",
        "inside_release",
        "duplicate_path",
        "unsorted_paths",
        "missing_live_mapping",
        "unapproved_live_mapping",
    ],
)
def test_external_native_mapping_policy_and_live_set_are_exact(
    deployment_name,
    policy_failure,
    closure_failure,
    case,
):
    snapshot = _good_snapshot()
    deployment = snapshot[deployment_name]
    policy = deployment["policy"]
    process = deployment["attestation"]["process"]
    mappings = policy["preapproved_external_native_executable_mappings"]

    expected_failure = policy_failure
    if case == "missing_policy":
        policy.pop("preapproved_external_native_executable_mappings")
    elif case == "empty_policy":
        mappings.clear()
    elif case == "extra_entry_field":
        mappings[0]["kind"] = "native_library"
    elif case == "invalid_hash":
        mappings[0]["sha256"] = "not-a-sha256"
    elif case == "inside_release":
        mappings[0]["path"] = f"{_WRITER_ARTIFACT_ROOT}/native.so"
    elif case == "duplicate_path":
        mappings.append(copy.deepcopy(mappings[0]))
    elif case == "unsorted_paths":
        mappings.append({"path": "/aaa/native.so", "sha256": "8" * 64})
    elif case == "missing_live_mapping":
        process["mapped_executable_paths"] = [_WRITER_INTERPRETER]
        expected_failure = closure_failure
    elif case == "unapproved_live_mapping":
        process["mapped_executable_paths"].append("/usr/lib/unapproved.so")
        expected_failure = closure_failure
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(case)

    assert expected_failure in _failed_names(snapshot)


def test_exact_immutable_projection_exporter_unit_and_timer_can_be_enabled():
    snapshot = _good_snapshot()
    _enable_projection_exporter(snapshot)

    report = preflight.evaluate_snapshot(snapshot)

    assert report.ok


@pytest.mark.parametrize(
    "mutate,failed",
    [
        (lambda value: value.update(writer_uid=1001), "identity.distinct_uids"),
        (
            lambda value: value.update(gateway_uid=float("inf")),
            "identity.distinct_uids",
        ),
        (
            lambda value: value.update(projector_gid=2002),
            "identity.projector_gid_isolated",
        ),
        (
            lambda value: value.update(projector_gid=2001),
            "identity.projector_gid_isolated",
        ),
        (
            lambda value: value.update(gateway_supplementary_gids=[2001]),
            "identity.gateway_socket_membership",
        ),
        (
            lambda value: value.update(
                gateway_supplementary_gids=[2000, 2001, 2999]
            ),
            "identity.gateway_socket_membership",
        ),
        (
            lambda value: value.update(writer_supplementary_gids=[2003]),
            "identity.writer_projector_membership",
        ),
        (
            lambda value: value.update(writer_supplementary_gids=[]),
            "identity.writer_projector_membership",
        ),
        (
            lambda value: value.update(writer_supplementary_gids=[2003, 2999]),
            "identity.writer_projector_membership",
        ),
        (
            lambda value: value["helper"].update(exists=True),
            "helper.legacy_absent",
        ),
        (
            lambda value: value["helper"]["gateway_access"].update(read=True),
            "helper.gateway_inaccessible",
        ),
        (lambda value: value["socket"].update(owner_uid=1001), "socket.writer_owned"),
        (lambda value: value["socket"].update(group_gid=99), "socket.dedicated_group"),
        (lambda value: value["socket"].update(mode="0666"), "socket.mode"),
        (
            lambda value: value["credential"].update(owner_uid=1001),
            "credential.writer_owned",
        ),
        (lambda value: value["credential"].update(mode="0640"), "credential.mode"),
        (
            lambda value: value["credential"]["gateway_access"].update(execute=True),
            "credential.gateway_inaccessible",
        ),
        (
            lambda value: value["projection_export"].update(owner_uid=1001),
            "projection_export.writer_owned",
        ),
        (
            lambda value: value["projection_export"].update(group_gid=2001),
            "projection_export.projector_group",
        ),
        (
            lambda value: value["projection_export"].update(mode="0600"),
            "projection_export.mode",
        ),
        (
            lambda value: value["projection_export"]["gateway_access"].update(
                read=True
            ),
            "projection_export.gateway_inaccessible",
        ),
        (
            lambda value: value["projection_export"]["projector_access"].update(
                write=True
            ),
            "projection_export.projector_read_only",
        ),
        (
            lambda value: value["gateway_process"].update(dumpable=True),
            "gateway_process.nondumpable",
        ),
        (
            lambda value: value["gateway_process"].update(core_soft_limit=1),
            "gateway_process.core_disabled",
        ),
        (
            lambda value: value["runtime_secret_sources"].update(complete=False),
            "runtime_secrets.evidence_complete",
        ),
        (
            lambda value: value["runtime_secret_sources"][
                "legacy_hermes_env"
            ].update(exists=True),
            "runtime_secrets.legacy_env_absent",
        ),
        (
            lambda value: value["runtime_secret_sources"].update(
                gateway_readable_secret_files=["/tmp/discord-token"]
            ),
            "runtime_secrets.no_readable_files",
        ),
        (
            lambda value: value["runtime_secret_sources"].update(
                gateway_readable_database_credential_files=[
                    "/run/muncho/database-password"
                ]
            ),
            "runtime_secrets.no_readable_database_credentials",
        ),
        (
            lambda value: value["runtime_secret_sources"].update(
                gateway_readable_systemd_environment_files=[
                    "/etc/muncho/gateway.env"
                ]
            ),
            "runtime_secrets.no_readable_systemd_environment",
        ),
        (
            lambda value: value["runtime_secret_sources"].update(
                open_database_credential_fds=[9]
            ),
            "runtime_secrets.no_database_credential_fds",
        ),
        (
            lambda value: value["runtime_secret_sources"][
                "cloud_sql_unix_socket"
            ].update(open_fds=[11]),
            "runtime_secrets.no_cloud_sql_socket",
        ),
        (
            lambda value: value["runtime_secret_sources"][
                "effective_gateway_env"
            ].update(database_password_variable_names=["PGPASSWORD"]),
            "runtime_secrets.gateway_env_database_clean",
        ),
        (
            lambda value: value["runtime_secret_sources"]["sources"][0].update(
                provisioned_by="gateway"
            ),
            "runtime_secrets.trusted_sources",
        ),
        (
            lambda value: value["gateway_iam"].update(
                roles=["roles/cloudsql.client"]
            ),
            "iam.gateway_roles",
        ),
        (
            lambda value: value["gateway_iam"].update(
                permissions=["secretmanager.versions.access"]
            ),
            "iam.gateway_permissions",
        ),
        (
            lambda value: value["gateway_iam"].update(complete=False),
            "iam.gateway_evidence_complete",
        ),
        (
            lambda value: value["writer_iam"].update(
                roles=["roles/secretmanager.secretAccessor"]
            ),
            "iam.writer_roles",
        ),
        (
            lambda value: value["writer_iam"].update(complete=False),
            "iam.writer_evidence_complete",
        ),
        (
            lambda value: value["database"]["attestation"].update(superuser=True),
            "database.least_privilege",
        ),
        (
            lambda value: value["database"]["attestation"].update(
                executable_routines=[]
            ),
            "database.least_privilege",
        ),
        (
            lambda value: value["database"]["attestation"][
                "canonical_event_log_identity"
            ].update(owner="gateway_controlled_owner"),
            "database.least_privilege",
        ),
        (
            lambda value: value["database"]["attestation"][
                "canonical_private_schema_identity"
            ].update(owner="gateway_controlled_owner"),
            "database.least_privilege",
        ),
    ],
)
def test_each_boundary_violation_fails_closed(mutate, failed):
    snapshot = _good_snapshot()
    mutate(snapshot)

    assert failed in _failed_names(snapshot)


@pytest.mark.parametrize(
    "field,value",
    [
        (
            "table_grants",
            [{"table": "canonical_brain.events", "privileges": ["UPDATE"]}],
        ),
        (
            "sequence_grants",
            [{"sequence": "canonical_brain.events_id_seq", "privileges": ["USAGE"]}],
        ),
        ("schema_privileges", ["USAGE", "CREATE"]),
        ("database_privileges", ["CONNECT", "TEMP"]),
        ("role_memberships", []),
        ("role_memberships", [CANONICAL_WRITER_ROLE, "cloudsqlsuperuser"]),
    ],
)
def test_preflight_hard_pins_production_database_privileges(field, value):
    snapshot = _good_snapshot()
    snapshot["database"]["policy"][field] = value
    snapshot["database"]["attestation"][field] = copy.deepcopy(value)

    assert "database.least_privilege" in _failed_names(snapshot)


def test_preflight_rejects_private_schema_structure_drift():
    snapshot = _good_snapshot()
    snapshot["database"]["attestation"]["canonical_private_schema_identity"][
        "relations"
    ][0]["columns"] = ['{"name":"drifted"}']

    assert "database.least_privilege" in _failed_names(snapshot)


_DATABASE_BOOLEAN_EVIDENCE_PATHS = (
    ("attestation", "superuser"),
    ("attestation", "createdb"),
    ("attestation", "createrole"),
    ("attestation", "replication"),
    ("attestation", "bypassrls"),
    ("attestation", "table_owner"),
    ("attestation", "routine_owner"),
    ("attestation", "routine_identities", 0, "security_definer"),
    ("attestation", "routine_identities", 0, "owner_dangerous"),
    ("attestation", "canonical_event_log_identity", "owner_dangerous"),
    ("attestation", "canonical_event_log_identity", "is_partition"),
    ("attestation", "canonical_event_log_identity", "row_security"),
    ("attestation", "canonical_event_log_identity", "force_row_security"),
    ("attestation", "canonical_event_log_identity", "inheritance"),
    ("attestation", "canonical_event_log_identity", "primary_index_exact"),
    (
        "attestation",
        "canonical_private_schema_identity",
        "owner_dangerous",
    ),
    (
        "attestation",
        "canonical_private_schema_identity",
        "relations",
        0,
        "owner_dangerous",
    ),
    (
        "attestation",
        "canonical_private_schema_identity",
        "relations",
        0,
        "is_partition",
    ),
    (
        "attestation",
        "canonical_private_schema_identity",
        "relations",
        0,
        "row_security",
    ),
    (
        "attestation",
        "canonical_private_schema_identity",
        "relations",
        0,
        "force_row_security",
    ),
    (
        "attestation",
        "canonical_private_schema_identity",
        "relations",
        0,
        "inheritance",
    ),
)


@pytest.mark.parametrize(
    "path",
    _DATABASE_BOOLEAN_EVIDENCE_PATHS,
    ids=lambda path: ".".join(str(part) for part in path),
)
@pytest.mark.parametrize(
    "malformed",
    ["true", 1, None],
    ids=("string", "integer", "null"),
)
def test_preflight_rejects_non_boolean_database_evidence(path, malformed):
    snapshot = _good_snapshot()
    target = snapshot["database"]
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = malformed

    assert "database.least_privilege" in _failed_names(snapshot)


@pytest.mark.parametrize("digest", [None, "f" * 63, "F" * 64])
def test_preflight_requires_exact_private_schema_policy_digest(digest):
    snapshot = _good_snapshot()
    if digest is None:
        snapshot["database"]["policy"].pop(
            "private_schema_identity_sha256"
        )
    else:
        snapshot["database"]["policy"][
            "private_schema_identity_sha256"
        ] = digest

    assert "database.least_privilege" in _failed_names(snapshot)


@pytest.mark.parametrize("digest", ["d" * 63, "D" * 64, "g" * 64])
def test_preflight_rejects_invalid_managed_cloudsqladmin_hba_receipt(digest):
    snapshot = _good_snapshot()
    snapshot["database"]["policy"]["managed_cloudsqladmin_hba_rejection_sha256"] = (
        digest
    )

    assert "database.least_privilege" in _failed_names(snapshot)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda database: database[
            "managed_cloudsqladmin_hba_rejection_evidence"
        ].update(collector_uid=1002),
        lambda database: database[
            "managed_cloudsqladmin_hba_rejection_evidence"
        ].update(source_owner_uid=1002),
        lambda database: database[
            "managed_cloudsqladmin_hba_rejection_evidence"
        ].update(same_credential=False),
        lambda database: database[
            "managed_cloudsqladmin_hba_rejection_evidence"
        ].update(same_tls_server_name=False),
        lambda database: database["managed_cloudsqladmin_hba_rejection_evidence"][
            "receipt"
        ].update(expires_at_unix=1_799_999_999),
        lambda database: database["connection"].update(host="10.0.0.9"),
        lambda database: database["connection"].update(
            tls_server_name="other.internal"
        ),
        lambda database: database["connection"].pop("tls_server_name"),
        lambda database: database[
            "managed_cloudsqladmin_hba_rejection_evidence"
        ].pop("same_tls_server_name"),
        lambda database: database["managed_cloudsqladmin_hba_rejection_evidence"][
            "receipt"
        ].update(tls_server_name="other.internal"),
    ],
)
def test_preflight_requires_fresh_root_collected_bound_hba_evidence(mutate):
    snapshot = _good_snapshot()
    mutate(snapshot["database"])

    assert "database.least_privilege" in _failed_names(snapshot)


def test_preflight_accepts_distinct_fresh_active_hba_receipt_bound_to_baseline():
    snapshot = _good_snapshot()
    active = copy.deepcopy(_managed_hba_receipt())
    active["observed_at_unix"] = 1_799_999_995
    active["expires_at_unix"] = 1_800_000_025
    evidence = snapshot["database"][
        "managed_cloudsqladmin_hba_rejection_evidence"
    ]
    evidence["receipt"] = active
    evidence["receipt_sha256"] = managed_cloudsqladmin_hba_receipt_from_mapping(
        active
    ).sha256

    assert preflight.evaluate_snapshot(snapshot).ok


def test_preflight_rejects_active_hba_evidence_older_than_thirty_seconds():
    snapshot = _good_snapshot()
    stale = copy.deepcopy(_managed_hba_receipt())
    stale["observed_at_unix"] = 1_799_999_969
    stale["expires_at_unix"] = 1_800_000_100
    evidence = snapshot["database"][
        "managed_cloudsqladmin_hba_rejection_evidence"
    ]
    evidence["receipt"] = stale
    evidence["receipt_sha256"] = managed_cloudsqladmin_hba_receipt_from_mapping(
        stale
    ).sha256

    assert "database.least_privilege" in _failed_names(snapshot)


def test_preflight_requires_every_pinned_non_executable_helper_identity():
    snapshot = _good_snapshot()
    snapshot["database"]["policy"]["helper_routine_identities"].pop()
    snapshot["database"]["attestation"]["helper_routine_identities"].pop()

    assert "database.least_privilege" in _failed_names(snapshot)


def test_preflight_rejects_helper_identity_drift_and_accidental_execute():
    snapshot = _good_snapshot()
    snapshot["database"]["attestation"]["helper_routine_identities"][0][
        "definition_sha256"
    ] = "f" * 64
    assert "database.least_privilege" in _failed_names(snapshot)

    snapshot = _good_snapshot()
    helper = snapshot["database"]["policy"]["helper_routine_identities"][0]
    helper["security_definer"] = True
    snapshot["database"]["attestation"]["helper_routine_identities"][0][
        "security_definer"
    ] = True
    assert "database.least_privilege" in _failed_names(snapshot)


@pytest.mark.parametrize(
    "mutate,failed",
    [
        (
            lambda value: value["gateway_process"].update(complete=False),
            "gateway_process.evidence_fresh",
        ),
        (
            lambda value: value["gateway_process"].update(
                observed_at_unix=value["collected_at_unix"] - 31
            ),
            "gateway_process.evidence_fresh",
        ),
        (
            lambda value: value["gateway_process"].update(platform="darwin"),
            "gateway_process.linux",
        ),
        (
            lambda value: value["gateway_process"].update(systemd_main_pid=9999),
            "gateway_process.exact_main_pid",
        ),
        (
            lambda value: value["gateway_process"].update(
                systemd_main_pid_start_time_ticks=999999
            ),
            "gateway_process.exact_main_pid",
        ),
        (
            lambda value: value["gateway_process"].update(core_hard_limit=1),
            "gateway_process.core_disabled",
        ),
        (
            lambda value: value["runtime_secret_sources"][
                "legacy_hermes_env"
            ]["gateway_child_file_access"].update(read=True),
            "runtime_secrets.legacy_env_absent",
        ),
        (
            lambda value: value["runtime_secret_sources"]["pgpass"].update(
                exists=True
            ),
            "runtime_secrets.pgpass_absent",
        ),
        (
            lambda value: value["runtime_secret_sources"].update(
                gateway_child_readable_secret_files=["/run/leaked-token"]
            ),
            "runtime_secrets.no_readable_files",
        ),
        (
            lambda value: value["runtime_secret_sources"].update(
                gateway_child_readable_database_credential_files=[
                    "/run/leaked-database-token"
                ]
            ),
            "runtime_secrets.no_readable_database_credentials",
        ),
        (
            lambda value: value["runtime_secret_sources"].update(
                gateway_child_readable_systemd_environment_files=[
                    "/etc/muncho/gateway.env"
                ]
            ),
            "runtime_secrets.no_readable_systemd_environment",
        ),
        (
            lambda value: value["runtime_secret_sources"].update(
                inherited_database_credential_fds=[7]
            ),
            "runtime_secrets.no_database_credential_fds",
        ),
        (
            lambda value: value["runtime_secret_sources"][
                "cloud_sql_unix_socket"
            ]["gateway_child_access"].update(read=True),
            "runtime_secrets.no_cloud_sql_socket",
        ),
        (
            lambda value: value["runtime_secret_sources"][
                "effective_gateway_env"
            ].update(values_included=True),
            "runtime_secrets.gateway_env_database_clean",
        ),
        (
            lambda value: value["runtime_secret_sources"][
                "effective_gateway_env"
            ].update(values={"PGPASSWORD": "must-never-enter-snapshot"}),
            "runtime_secrets.gateway_env_database_clean",
        ),
        (
            lambda value: value["runtime_secret_sources"][
                "effective_gateway_env"
            ].update(
                database_connection_secret_variable_names=["DATABASE_URL"]
            ),
            "runtime_secrets.gateway_env_database_clean",
        ),
        (
            lambda value: value["runtime_secret_sources"]["sources"][0][
                "gateway_file_access"
            ].update(read=True),
            "runtime_secrets.trusted_sources",
        ),
        (
            lambda value: value["runtime_secret_sources"]["sources"][0][
                "gateway_child_file_access"
            ].update(read=True),
            "runtime_secrets.trusted_sources",
        ),
        (
            lambda value: value["runtime_secret_sources"].update(sources=[]),
            "runtime_secrets.trusted_sources",
        ),
        (
            lambda value: value["runtime_secret_sources"]["sources"][0].update(
                name="discord_bot_token"
            ),
            "runtime_secrets.discord_source_isolated",
        ),
    ],
)
def test_process_and_runtime_secret_evidence_fail_closed(mutate, failed):
    snapshot = _good_snapshot()
    mutate(snapshot)

    assert failed in _failed_names(snapshot)


@pytest.mark.parametrize(
    "mutate,failed",
    [
        (
            lambda value: value.update(deployment_mode="full"),
            "deployment.writer_only_mode",
        ),
        (
            lambda value: value["discord_edge"].update(gateway_enabled=True),
            "discord_edge.writer_only_disabled",
        ),
        (
            lambda value: value["discord_edge"].update(
                writer_authority_enabled=True
            ),
            "discord_edge.writer_only_disabled",
        ),
        (
            lambda value: value["discord_edge"].update(unit_exists=True),
            "discord_edge.writer_only_unit_absent",
        ),
        (
            lambda value: value["discord_edge"].update(config_exists=True),
            "discord_edge.writer_only_config_absent",
        ),
        (
            lambda value: value["discord_edge"].update(token_exists=True),
            "discord_edge.writer_only_token_absent",
        ),
        (
            lambda value: value["discord_edge"].update(socket_exists=True),
            "discord_edge.writer_only_socket_absent",
        ),
        (
            lambda value: value["discord_edge"].update(process_pids=[9001]),
            "discord_edge.writer_only_process_absent",
        ),
        (
            lambda value: value["discord_edge"].update(
                observed_at_unix=1_799_999_969
            ),
            "discord_edge.writer_only_evidence_fresh",
        ),
    ],
)
def test_writer_only_discord_surface_must_be_exactly_absent(mutate, failed):
    snapshot = _good_snapshot()
    mutate(snapshot)

    assert failed in _failed_names(snapshot)


@pytest.mark.parametrize(
    "mutate,failed",
    [
        (
            lambda value: value.pop("writer_deployment"),
            "writer_deployment.evidence_complete",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"].update(
                complete=False
            ),
            "writer_deployment.evidence_complete",
        ),
        (
            lambda value: value["writer_deployment"]["policy"].update(
                unit_name="other.service"
            ),
            "writer_deployment.policy_valid",
        ),
        (
            lambda value: value["writer_deployment"]["policy"].update(
                revision="not-a-revision"
            ),
            "writer_deployment.policy_valid",
        ),
        (
            lambda value: value["writer_deployment"]["policy"].update(
                artifact_digest_sha256="f" * 63
            ),
            "writer_deployment.policy_valid",
        ),
        (
            lambda value: value["writer_deployment"]["policy"].update(
                module="gateway.canonical_writer_service"
            ),
            "writer_deployment.policy_valid",
        ),
        (
            lambda value: value["writer_deployment"]["policy"].update(
                module_origin="/tmp/canonical_writer_bootstrap.py"
            ),
            "writer_deployment.policy_valid",
        ),
        (
            lambda value: value["writer_deployment"]["policy"][
                "exec_start"
            ].remove("-B"),
            "writer_deployment.policy_valid",
        ),
        (
            lambda value: value["writer_deployment"]["policy"][
                "exec_start"
            ].remove("-I"),
            "writer_deployment.policy_valid",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "unit"
            ]["exec_start"].append("--unexpected"),
            "writer_deployment.unit_exact",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "unit"
            ].update(interpreter="/usr/bin/python3"),
            "writer_deployment.unit_exact",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "unit"
            ].update(revision="9" * 40),
            "writer_deployment.unit_exact",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "unit"
            ].update(environment_pythonpath=["/tmp/gateway-controlled"]),
            "writer_deployment.unit_exact",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "unit"
            ].update(environment_pythonhome="/tmp/gateway-python"),
            "writer_deployment.unit_exact",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "unit"
            ].update(alternate_exec_commands=["/tmp/gateway-controlled"]),
            "writer_deployment.unit_exact",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "unit"
            ].update(environment_files=["/tmp/gateway-controlled.env"]),
            "writer_deployment.unit_exact",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "unit"
            ].update(
                code_injection_environment_variable_names=["LD_PRELOAD"]
            ),
            "writer_deployment.unit_exact",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "unit"
            ].update(user_uid=1001),
            "writer_deployment.unit_exact",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "unit"
            ].update(group_gid=2001),
            "writer_deployment.unit_exact",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "artifact"
            ].update(revision="9" * 40),
            "writer_deployment.artifact_exact",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "artifact"
            ].update(digest_sha256="9" * 64),
            "writer_deployment.artifact_exact",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "artifact"
            ].update(owner_uid=1001),
            "writer_deployment.artifact_immutable",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "artifact"
            ].update(mode="0775"),
            "writer_deployment.artifact_immutable",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "artifact"
            ].update(mode="4555"),
            "writer_deployment.artifact_immutable",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "artifact"
            ].update(symlink=True),
            "writer_deployment.artifact_immutable",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "artifact"
            ]["gateway_mutation_access"].update(write=True),
            "writer_deployment.artifact_immutable",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "artifact"
            ].update(gateway_open_write_fds=[7]),
            "writer_deployment.artifact_immutable",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "artifact"
            ]["parent_chain"]["gateway_child_mutation_access"].update(
                replace=True
            ),
            "writer_deployment.artifact_immutable",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "artifact"
            ]["parent_chain"].update(mount_read_only=False),
            "writer_deployment.artifact_immutable",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "import_closure"
            ].update(complete=False),
            "writer_deployment.import_closure_complete",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "import_closure"
            ]["paths"].pop(),
            "writer_deployment.import_closure_exact",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "import_closure"
            ]["paths"][1].update(digest_sha256="9" * 64),
            "writer_deployment.import_closure_immutable",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "import_closure"
            ]["paths"][2].update(owner_uid=1001),
            "writer_deployment.import_closure_immutable",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "import_closure"
            ]["paths"][3]["gateway_child_mutation_access"].update(
                rename=True
            ),
            "writer_deployment.import_closure_immutable",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "import_closure"
            ]["paths"][3].update(gateway_child_writable_mappings=["mapping"]),
            "writer_deployment.import_closure_immutable",
        ),
        (
            lambda value: value["writer_deployment"]["policy"][
                "import_paths"
            ][2].update(path="/tmp/gateway-controlled-stdlib"),
            "writer_deployment.policy_valid",
        ),
        (
            lambda value: value["writer_deployment"]["policy"][
                "import_paths"
            ][3].update(kind="native_library"),
            "writer_deployment.policy_valid",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "process"
            ].update(complete=False),
            "writer_deployment.process_fresh",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "process"
            ].update(observed_at_unix=1_799_999_969),
            "writer_deployment.process_fresh",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "process"
            ].update(systemd_main_pid=9999),
            "writer_deployment.process_exact",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "process"
            ].update(systemd_main_pid_start_time_ticks=999999),
            "writer_deployment.process_exact",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "process"
            ]["cmdline"].append("--unexpected"),
            "writer_deployment.process_exact",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "process"
            ].update(executable_digest_sha256="9" * 64),
            "writer_deployment.process_exact",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "process"
            ].update(revision="9" * 40),
            "writer_deployment.process_exact",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "process"
            ].update(bootstrap_module_origin="/tmp/injected_bootstrap.py"),
            "writer_deployment.process_exact",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "process"
            ]["effective_import_paths"].append("/tmp/gateway-imports"),
            "writer_deployment.process_code_closure",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "process"
            ].update(loaded_module_origins_complete=False),
            "writer_deployment.process_code_closure",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "process"
            ].update(loaded_module_origins=["/tmp/gateway_module.py"]),
            "writer_deployment.process_code_closure",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "process"
            ].update(mapped_executable_paths_complete=False),
            "writer_deployment.process_code_closure",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "process"
            ]["mapped_executable_paths"].append("/usr/lib/libpython.so"),
            "writer_deployment.process_code_closure",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "process"
            ].update(unexpected_import_origins=["/tmp/injected.py"]),
            "writer_deployment.process_code_closure",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "process"
            ].update(deleted_code_mappings=["deleted-interpreter"]),
            "writer_deployment.process_code_closure",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "process"
            ].update(writable_code_mappings=["writable-module"]),
            "writer_deployment.process_code_closure",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "mounts"
            ].update(complete=False),
            "writer_deployment.evidence_complete",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "mounts"
            ]["read_write_paths"].append("/home/muncho"),
            "writer_deployment.mount_carveouts_exact",
        ),
        (
            lambda value: value["writer_deployment"]["policy"][
                "read_write_paths"
            ].append("/home/muncho"),
            "writer_deployment.policy_valid",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "mounts"
            ]["bind_paths"].append("/home/muncho:/writer-home"),
            "writer_deployment.mount_carveouts_exact",
        ),
        (
            lambda value: value["writer_deployment"]["attestation"][
                "mounts"
            ].update(bind_read_only_paths=[]),
            "writer_deployment.mount_carveouts_exact",
        ),
        (
            lambda value: value["writer_deployment"]["policy"].update(
                projection_export_directory="/var"
            ),
            "writer_deployment.policy_valid",
        ),
        (
            lambda value: value["writer_deployment"]["policy"].update(
                runtime_directory="/run"
            ),
            "writer_deployment.policy_valid",
        ),
        (
            lambda value: value["writer_deployment"]["policy"].update(
                config_path="/"
            ),
            "writer_deployment.policy_valid",
        ),
    ],
)
def test_writer_deployment_integrity_fails_closed(mutate, failed):
    snapshot = _good_snapshot()
    mutate(snapshot)

    assert failed in _failed_names(snapshot)


def test_writer_deployment_failure_report_never_echoes_snapshot_values():
    snapshot = _good_snapshot()
    sentinel = "must-never-be-emitted-secret"
    snapshot["writer_deployment"]["attestation"]["unit"].update(
        environment_pythonpath=[sentinel]
    )

    encoded = json.dumps(preflight.evaluate_snapshot(snapshot).to_dict())

    assert sentinel not in encoded


@pytest.mark.parametrize(
    "mutate,failed",
    [
        (
            lambda value: value.pop("gateway_deployment"),
            "gateway_deployment.evidence_complete",
        ),
        (
            lambda value: value["gateway_deployment"]["policy"].update(
                artifact_digest_sha256="9" * 64
            ),
            "gateway_deployment.shared_immutable_release",
        ),
        (
            lambda value: value["gateway_deployment"]["policy"].update(
                module="gateway.untrusted"
            ),
            "gateway_deployment.shared_immutable_release",
        ),
        (
            lambda value: value["gateway_deployment"]["policy"].update(
                module_origin="/tmp/gateway.py"
            ),
            "gateway_deployment.shared_immutable_release",
        ),
        (
            lambda value: value["gateway_deployment"]["policy"][
                "exec_start"
            ].remove("-B"),
            "gateway_deployment.shared_immutable_release",
        ),
        (
            lambda value: value["gateway_deployment"]["policy"][
                "exec_start"
            ].remove("-I"),
            "gateway_deployment.shared_immutable_release",
        ),
        (
            lambda value: value["gateway_deployment"]["policy"][
                "import_paths"
            ].pop(),
            "gateway_deployment.shared_immutable_release",
        ),
        (
            lambda value: value["gateway_deployment"]["policy"].update(
                read_write_paths=["/etc"]
            ),
            "gateway_deployment.shared_immutable_release",
        ),
        (
            lambda value: value["gateway_deployment"]["policy"][
                "bind_paths"
            ].append("/tmp:/opt/muncho-canonical-writer"),
            "gateway_deployment.shared_immutable_release",
        ),
        (
            lambda value: value["gateway_deployment"]["policy"].update(
                dynamic_python_loading_mode="disabled",
                dynamic_python_discovery_paths=["/home/muncho/.hermes/plugins"],
            ),
            "gateway_deployment.shared_immutable_release",
        ),
        (
            lambda value: value["gateway_deployment"]["policy"].update(
                dynamic_python_loading_mode="unrestricted"
            ),
            "gateway_deployment.shared_immutable_release",
        ),
        (
            lambda value: value["gateway_deployment"]["policy"].update(
                dynamic_python_loading_mode="immutable_only",
                dynamic_python_discovery_paths=["/home/muncho/.hermes/plugins"],
            ),
            "gateway_deployment.shared_immutable_release",
        ),
        (
            lambda value: value["gateway_deployment"]["attestation"][
                "unit"
            ].update(alternate_exec_commands=["/tmp/restart-wrapper"]),
            "gateway_deployment.unit_exact",
        ),
        (
            lambda value: value["gateway_deployment"]["attestation"][
                "unit"
            ].update(
                code_injection_environment_variable_names=["LD_PRELOAD"]
            ),
            "gateway_deployment.unit_exact",
        ),
        (
            lambda value: value["gateway_deployment"]["attestation"][
                "unit"
            ].update(user_uid=1002),
            "gateway_deployment.unit_exact",
        ),
        (
            lambda value: value["gateway_deployment"]["attestation"][
                "process"
            ].update(observed_at_unix=1_799_999_969),
            "gateway_deployment.process_fresh",
        ),
        (
            lambda value: value["gateway_deployment"]["attestation"][
                "process"
            ].update(systemd_main_pid=9999),
            "gateway_deployment.process_exact",
        ),
        (
            lambda value: value["gateway_deployment"]["attestation"][
                "process"
            ].update(entry_module_origin="/tmp/injected_gateway.py"),
            "gateway_deployment.process_exact",
        ),
        (
            lambda value: value["gateway_deployment"]["attestation"][
                "process"
            ]["effective_import_paths"].append("/tmp/imports"),
            "gateway_deployment.process_code_closure",
        ),
        (
            lambda value: value["gateway_deployment"]["attestation"][
                "process"
            ].update(loaded_module_origins=["/tmp/injected_gateway.py"]),
            "gateway_deployment.process_code_closure",
        ),
        (
            lambda value: value["gateway_deployment"]["attestation"][
                "process"
            ]["mapped_executable_paths"].append("/usr/lib/libpython.so"),
            "gateway_deployment.process_code_closure",
        ),
        (
            lambda value: value["gateway_deployment"]["attestation"][
                "process"
            ].update(dynamic_python_discovery_complete=False),
            "gateway_deployment.process_code_closure",
        ),
        (
            lambda value: value["gateway_deployment"]["attestation"][
                "process"
            ].update(dynamic_python_loading_mode="immutable_only"),
            "gateway_deployment.process_code_closure",
        ),
        (
            lambda value: value["gateway_deployment"]["attestation"][
                "process"
            ].update(
                dynamic_python_discovery_paths=[
                    "/home/muncho/.hermes/plugins"
                ]
            ),
            "gateway_deployment.process_code_closure",
        ),
        (
            lambda value: value["gateway_deployment"]["attestation"][
                "process"
            ].update(
                dynamic_python_writable_paths=[
                    "/home/muncho/.hermes/plugins"
                ]
            ),
            "gateway_deployment.process_code_closure",
        ),
        (
            lambda value: value["gateway_deployment"]["attestation"][
                "mounts"
            ]["read_write_paths"].append("/home/muncho/.hermes"),
            "gateway_deployment.mounts_exact",
        ),
    ],
)
def test_gateway_immutable_release_and_dynamic_loading_fail_closed(
    mutate, failed
):
    snapshot = _good_snapshot()
    mutate(snapshot)

    assert failed in _failed_names(snapshot)


@pytest.mark.parametrize(
    "mutate,failed",
    [
        (
            lambda value: value.pop("writer_authority_surface"),
            "writer_authority.root_evidence_fresh",
        ),
        (
            lambda value: value["writer_authority_surface"].update(
                complete=False
            ),
            "writer_authority.root_evidence_fresh",
        ),
        (
            lambda value: value["writer_authority_surface"].update(
                collected_by_uid=1001
            ),
            "writer_authority.root_evidence_fresh",
        ),
        (
            lambda value: value["writer_authority_surface"].update(
                observed_at_unix=1_799_999_969
            ),
            "writer_authority.root_evidence_fresh",
        ),
        (
            lambda value: value["writer_authority_surface"]["identities"][
                "gateway"
            ].update(effective_gid=2001),
            "writer_authority.identities_exact",
        ),
        (
            lambda value: value["writer_authority_surface"]["identities"][
                "gateway"
            ].update(supplementary_gids=[]),
            "writer_authority.identities_exact",
        ),
        (
            lambda value: value["writer_authority_surface"]["identities"][
                "gateway_children"
            ]["processes"].append(
                {
                    "pid": 5000,
                    "effective_uid": 0,
                    "effective_gid": 2000,
                    "supplementary_gids": [2000, 2001],
                }
            ),
            "writer_authority.identities_exact",
        ),
        (
            lambda value: value["writer_authority_surface"]["identities"][
                "writer"
            ].update(supplementary_gids=[2003, 2001]),
            "writer_authority.identities_exact",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "group_policy"
            ]["evaluated_dangerous_group_names"].pop(),
            "writer_authority.dangerous_groups_absent",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "group_policy"
            ]["gateway_dangerous_memberships"].append("docker"),
            "writer_authority.dangerous_groups_absent",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "group_policy"
            ]["unknown_privileged_gids"].append(9999),
            "writer_authority.dangerous_groups_absent",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "gateway_authority"
            ].update(can_manage_writer_units=True),
            "writer_authority.gateway_denied",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "gateway_authority"
            ].update(can_create_transient_units=True),
            "writer_authority.gateway_denied",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "gateway_authority"
            ]["authorized_polkit_actions"].append(
                "org.freedesktop.systemd1.manage-units"
            ),
            "writer_authority.gateway_denied",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "gateway_authority"
            ]["effective_capabilities"].append("CAP_SETUID"),
            "writer_authority.gateway_denied",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "gateway_child_authority"
            ].update(can_switch_to_writer_uid=True),
            "writer_authority.children_denied",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "gateway_child_authority"
            ]["authorized_sudo_commands"].append("ALL"),
            "writer_authority.children_denied",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "projection_exporter"
            ]["policy"].update(artifact_digest_sha256="9" * 64),
            "writer_authority.exporter_manifest_exact",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "projection_exporter"
            ]["policy"].update(export_path="/tmp/canonical-events.json"),
            "writer_authority.exporter_manifest_exact",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "projection_exporter"
            ]["policy"].update(export_limit=True),
            "writer_authority.exporter_manifest_exact",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "projection_exporter"
            ]["policy"]["exec_start"].append("--unexpected"),
            "writer_authority.exporter_manifest_exact",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "projection_exporter"
            ]["policy"]["read_write_paths"].append("/home/muncho"),
            "writer_authority.exporter_manifest_exact",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "projection_exporter"
            ]["policy"].update(timer_schedule={"OnCalendar": False}),
            "writer_authority.exporter_manifest_exact",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "projection_exporter"
            ]["attestation"].update(enabled=True),
            "writer_authority.exporter_state_safe",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "projection_exporter"
            ]["attestation"].update(unit={"name": "unexpected"}),
            "writer_authority.exporter_state_safe",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "privileged_execution_inventory"
            ]["writer_uid_service_units"].append("unreviewed.service"),
            "writer_authority.privileged_inventory_exact",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "privileged_execution_inventory"
            ]["writer_uid_timer_units"].append("unreviewed.timer"),
            "writer_authority.privileged_inventory_exact",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "privileged_execution_inventory"
            ]["writer_uid_transient_units"].append("run-u123.service"),
            "writer_authority.privileged_inventory_exact",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "privileged_execution_inventory"
            ]["writer_uid_cron_entries"].append("* * * * * injected"),
            "writer_authority.privileged_inventory_exact",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "privileged_execution_inventory"
            ]["writer_uid_at_jobs"].append("job:42"),
            "writer_authority.privileged_inventory_exact",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "privileged_execution_inventory"
            ]["writer_uid_process_executables"].append("/tmp/injected"),
            "writer_authority.privileged_inventory_exact",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "privileged_execution_inventory"
            ]["writer_uid_unattributed_processes"].append("pid:9999"),
            "writer_authority.privileged_inventory_exact",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "privileged_execution_inventory"
            ]["writer_unit_reverse_activation"]["socket_units"].append(
                "injected.socket"
            ),
            "writer_authority.privileged_inventory_exact",
        ),
        (
            lambda value: value["writer_authority_surface"][
                "privileged_execution_inventory"
            ]["gateway_writable_writer_unit_files"].append(
                "/etc/systemd/system/writer.service"
            ),
            "writer_authority.privileged_inventory_exact",
        ),
    ],
)
def test_root_collected_local_authority_surface_fails_closed(mutate, failed):
    snapshot = _good_snapshot()
    mutate(snapshot)

    assert failed in _failed_names(snapshot)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["writer_authority_surface"][
            "projection_exporter"
        ]["attestation"]["unit"]["exec_start"].append("--unexpected"),
        lambda value: value["writer_authority_surface"][
            "projection_exporter"
        ]["attestation"]["unit"]["systemd_properties"].update(
            ProtectSystem="full"
        ),
        lambda value: value["writer_authority_surface"][
            "projection_exporter"
        ]["attestation"]["timer"].update(
            schedule={"OnCalendar": "hourly"}
        ),
    ],
)
def test_enabled_projection_exporter_drift_fails_closed(mutate):
    snapshot = _good_snapshot()
    _enable_projection_exporter(snapshot)
    mutate(snapshot)

    assert "writer_authority.exporter_state_safe" in _failed_names(snapshot)


@pytest.mark.parametrize("property_name", preflight._HARDENED_TRUE_PROPERTIES)
def test_each_required_systemd_boolean_is_enforced(property_name):
    snapshot = _good_snapshot()
    snapshot["systemd_properties"][property_name] = "no"

    assert f"systemd.{property_name}" in _failed_names(snapshot)


@pytest.mark.parametrize("property_name", preflight._HARDENED_TRUE_PROPERTIES)
def test_each_required_gateway_systemd_boolean_is_enforced(property_name):
    snapshot = _good_snapshot()
    snapshot["gateway_systemd_properties"][property_name] = "no"

    assert f"gateway_systemd.{property_name}" in _failed_names(snapshot)


@pytest.mark.parametrize(
    "property_name,value",
    [
        ("ProtectSystem", "full"),
        ("ProtectHome", "no"),
        ("ProtectProc", "default"),
        ("ProcSubset", "all"),
        ("UMask", "0022"),
        ("CapabilityBoundingSet", "CAP_NET_ADMIN"),
        ("AmbientCapabilities", "CAP_DAC_OVERRIDE"),
        ("RestrictAddressFamilies", "AF_UNIX AF_INET AF_PACKET"),
    ],
)
def test_other_systemd_hardening_is_enforced(property_name, value):
    snapshot = _good_snapshot()
    snapshot["systemd_properties"][property_name] = value

    assert f"systemd.{property_name}" in _failed_names(snapshot)


def test_evaluation_is_deterministic_and_does_not_mutate_snapshot():
    snapshot = _good_snapshot()
    before = copy.deepcopy(snapshot)

    first = preflight.evaluate_snapshot(snapshot).to_dict()
    second = preflight.evaluate_snapshot(snapshot).to_dict()

    assert first == second
    assert snapshot == before


def test_cli_emits_stable_json_and_nonzero_for_failure(tmp_path, capsys):
    snapshot = _good_snapshot()
    snapshot["gateway_uid"] = snapshot["writer_uid"]
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    exit_code = preflight.main(["--input", str(path)])
    output = capsys.readouterr().out

    assert exit_code == 2
    decoded = json.loads(output)
    assert decoded["ok"] is False
    assert output == json.dumps(
        decoded, sort_keys=True, separators=(",", ":")
    ) + "\n"


def test_cli_never_treats_arbitrary_passing_json_as_activation_authority(
    tmp_path,
    capsys,
):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(_good_snapshot()), encoding="utf-8")

    assert preflight.main(["--input", str(path)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert any(
        check["name"] == "snapshot.non_authoritative"
        for check in report["checks"]
    )
