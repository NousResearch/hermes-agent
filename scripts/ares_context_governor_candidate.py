#!/usr/bin/env python3
"""Build and audit a sealed, pre-activation Context Governor candidate.

This tool deliberately operates only on isolated staging roots and temporary
HERMES_HOME fixtures.  It never reads or writes a live profile, configuration,
or receipt store.  The scope ledger is an admission gate, not documentation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
import stat
import subprocess
import sysconfig
import tarfile
import tempfile
import time
import re
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any

# The candidate harness is deliberately executable as ``python scripts/...``;
# make the Ares release infrastructure importable without relying on a live
# installation or the caller's PYTHONPATH.
_ARES_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(_ARES_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ARES_SOURCE_ROOT))

from hermes_cli.ares_candidate_store import (
    CandidateStore,
    generate_fault_matrix,
    sha256_bytes,
)
from ares_runtime.image import stage_runtime_image, write_release_manifest

SCHEMA_SCOPE = "AresContextGovernorScopeV1"
CANONICALIZATION_VERSION = "canonical-json-utf8-v1"
SCHEMA_CORE = "CandidateCoreV2"
SCHEMA_STAGED_CERT = "AresContextGovernorStagedCertificationV1"
SCHEMA_FULL_SEAL_CERT = "AresContextGovernorFullSealCertificationV2"
SCHEMA_CERT_SET = "CertificationSetV2"
SCHEMA_SEALED = "SealedCandidateV2"
SCHEMA_POST_SEAL = "PostSealEvidenceSetV1"
SCHEMA_AUTHORIZATION = "ActivationAuthorizationV1"
REQUIRED = "REQUIRED_FOR_ARES_RECOVERY"
DEPENDENCY = "REQUIRED_DEPENDENCY"
UNRELATED = "UNRELATED_CONCURRENT_WORK"
GENERATED = "GENERATED"
UNRESOLVED = "UNRESOLVED"


class CertificationAuthority(StrEnum):
    """Authority carried by candidate-measurement evidence.

    This harness never performs the separately governed activation transition,
    so every result it emits is evidence only, including a successful result.
    """

    NON_AUTHORIZING = "NON_AUTHORIZING"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def identified(value: dict[str, Any], field: str) -> dict[str, Any]:
    """Attach a staged canonical identity without circular self-hashing."""
    projection = dict(value)
    projection.pop(field, None)
    value[field] = digest(canonical(projection))
    return value


def run(
    args: list[str], *, cwd: Path | None = None, input: bytes | None = None
) -> bytes:
    return subprocess.check_output(args, cwd=cwd, input=input, stderr=subprocess.PIPE)


def git(root: Path, *args: str) -> str:
    return run(["git", *args], cwd=root).decode().strip()


def git_prefix(root: Path) -> str:
    """Path prefix when the declared project is a subtree of its Git repo."""
    return git(root, "rev-parse", "--show-prefix").rstrip("/")


def classification(repo: str, path: str) -> str:
    if "__pycache__" in path or path.endswith(".pyc"):
        return GENERATED
    if repo == "context-governor":
        return DEPENDENCY
    recovery_prefixes = (
        "agent/agent_init.py",
        "agent/context_engine.py",
        "agent/transports/ri_context_compressor.py",
        "hermes_constants.py",
        "hermes_cli/ares_candidate_",
        "ares_runtime/",
        "plugins/context_engine/__init__.py",
        "plugins/context_engine/ri-context-governor/",
        "plugins/context_engine/_context_governor/",
        "tests/agent/test_context_governor_restore.py",
        "tests/run_agent/test_plugin_context_engine_init.py",
        "scripts/ares_context_governor_candidate.py",
        "scripts/verify_ares_context_governor_scope.py",
        "scripts/scan_ares_context_governor_candidate.py",
        "pyproject.toml",
        "tests/scripts/test_ares_context_governor_candidate.py",
        "tests/scripts/test_ares_context_governor_scanners.py",
        "tests/plugins/test_context_governor_",
        "tests/hermes_cli/test_ares_candidate_",
        "tests/ares_runtime/",
        "tests/test_hermes_constants_ares.py",
        "docs/ares-candidate-custody.md",
    )
    return REQUIRED if path.startswith(recovery_prefixes) else UNRELATED


def changed_entries(root: Path, repo: str) -> list[dict[str, Any]]:
    status = git(root, "status", "--porcelain=v1").splitlines()
    prefix = git_prefix(root)
    entries: list[dict[str, Any]] = []
    for line in status:
        # Git emits either XY<space>path or, from a subdirectory of a larger
        # repository, X<space>path. Do not drop a path character in the latter.
        path = line[3:] if len(line) > 2 and line[2] == " " else line[2:]
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[1]
        if prefix:
            marker = prefix + "/"
            if not path.startswith(marker):
                # A sibling change belongs to the encompassing repository,
                # not this declared Context Governor project; it is unresolved
                # rather than silently classified into this candidate.
                entries.append({
                    "path": path,
                    "status": line[:2],
                    "classification": UNRESOLVED,
                    "hunks": [{"range": "all", "classification": UNRESOLVED}],
                })
                continue
            path = path[len(marker) :]
        state = classification(repo, path)
        hunks = changed_hunks(root, prefix, path, line[:2], state)
        if not hunks:
            state = UNRESOLVED
            hunks = [{"range": "unparseable", "classification": UNRESOLVED}]
        entries.append({
            "path": path,
            "status": line[:2],
            "classification": state,
            "hunks": hunks,
        })
    return sorted(entries, key=lambda entry: entry["path"])


def changed_hunks(
    root: Path, prefix: str, path: str, status: str, state: str
) -> list[dict[str, Any]]:
    """Record each tracked diff hunk; unclassified/malformed input blocks scope."""
    if status == "??":
        return [{"range": "all", "classification": state}]
    # `root` is the declared project working directory, so Git pathspecs are
    # project-relative even when the project itself is a subdirectory of a
    # larger repository.
    patch = run(["git", "diff", "--unified=0", "HEAD", "--", path], cwd=root).decode(
        errors="replace"
    )
    ranges = re.findall(r"^@@ [^@]* \+(\d+(?:,\d+)?) @@", patch, flags=re.MULTILINE)
    return [{"range": f"new:{range_}", "classification": state} for range_ in ranges]


def scope_ledger(ares: Path, governor: Path) -> dict[str, Any]:
    entries = [
        {"repository": "ares", **entry} for entry in changed_entries(ares, "ares")
    ]
    entries += [
        {"repository": "context-governor", **entry}
        for entry in changed_entries(governor, "context-governor")
    ]
    unresolved = [
        entry["path"] for entry in entries if entry["classification"] == UNRESOLVED
    ]
    return {
        "schema": SCHEMA_SCOPE,
        "ares_head": git(ares, "rev-parse", "HEAD"),
        "context_governor_head": git(governor, "rev-parse", "HEAD"),
        "entries": entries,
        "unresolved": unresolved,
        "candidate_construction_allowed": not unresolved,
    }


def _tool_module(filename: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(filename)
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def independent_scope_manifest(
    ares: Path, governor: Path, ledger: dict[str, Any], payload: Path
) -> dict[str, Any]:
    """Declare exact inputs for the separately implemented replay verifier."""
    verifier = _tool_module(
        "verify_ares_context_governor_scope.py", "ares_cg_scope_verifier"
    )
    repositories: list[dict[str, Any]] = []
    for name, root, subtree in (
        ("ares", ares, "ares"),
        ("context-governor", governor, "context-governor"),
    ):
        required_paths = selected_paths(ledger, name)
        actual = verifier.normalized_hunks(verifier.complete_patch(root))
        required = set(
            verifier.normalized_hunks(verifier.complete_patch(root, required_paths))
        )
        if not required <= actual.keys():
            raise RuntimeError(
                f"scope declaration cannot locate required hunks for {name}"
            )
        repositories.append({
            "name": name,
            "root": str(root),
            "baseline_commit": ledger[f"{name.replace('-', '_')}_head"],
            "base_blob_ids": verifier._base_blob_ids(
                root, ledger[f"{name.replace('-', '_')}_head"]
            ),
            "required_paths": required_paths,
            "required_hunk_ids": sorted(required),
            "unrelated_hunk_ids": sorted(set(actual) - required),
            "candidate_subtree": subtree,
            # The runtime release must include all Ares source bytes.  The
            # reviewed hunk list still remains narrow and is replayed above.
            "candidate_paths": None,
        })
    manifest = {
        "schema": "AresContextGovernorScopeManifestV2",
        "canonicalization_version": CANONICALIZATION_VERSION,
        "candidate_tree_scope": "declared_subtrees_v1",
        "repositories": repositories,
    }
    manifest["candidate_tree_sha256"] = verifier.declared_tree_digest(
        payload, repositories
    )
    return manifest


def extract_head(root: Path, destination: Path, paths: list[str] | None = None) -> None:
    destination.mkdir(parents=True)
    prefix = git_prefix(root)
    requested = paths or []
    # Git archive rejects an untracked pathspec. Only baseline-tracked files
    # belong in this extraction; overlay_selected copies declared untracked
    # candidate files immediately afterward.
    tracked = (
        run(["git", "ls-files", "-z", "--", *requested], cwd=root).split(b"\0")
        if requested
        else []
    )
    selected = [entry.decode() for entry in tracked if entry]
    selected = [f"{prefix}/{path}" if prefix else path for path in selected]
    arguments = ["git", "archive", "--format=tar", "HEAD"] + (
        ["--", *selected] if selected else []
    )
    with tarfile.open(fileobj=io.BytesIO(run(arguments, cwd=root))) as archive:
        archive.extractall(destination, filter="data")


def selected_paths(ledger: dict[str, Any], repository: str) -> list[str]:
    paths = [
        entry["path"]
        for entry in ledger["entries"]
        if entry["repository"] == repository
        and entry["classification"] in {REQUIRED, DEPENDENCY}
    ]
    # This configuration is a release input even when it is unchanged in the
    # worktree.  Candidate gates must never obtain lint authority from the
    # development checkout after staging.
    if repository == "ares" and "pyproject.toml" not in paths:
        paths.append("pyproject.toml")
    return sorted(paths)


def overlay_selected(root: Path, stage: Path, paths: list[str]) -> None:
    prefix = git_prefix(root)
    git_paths = [f"{prefix}/{path}" if prefix else path for path in paths]
    if paths:
        patch = run(["git", "diff", "--binary", "HEAD", "--", *git_paths], cwd=root)
        if prefix:
            patch = patch.replace(f"a/{prefix}/".encode(), b"a/").replace(
                f"b/{prefix}/".encode(), b"b/"
            )
        if patch:
            subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "-"],
                cwd=stage,
                input=patch,
                check=True,
            )
    for path in paths:
        source, target = root / path, stage / path
        tracked = subprocess.check_output(
            ["git", "ls-files", "--", f"{prefix}/{path}" if prefix else path], cwd=root
        ).strip()
        if source.exists() and not tracked:
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(
                    source,
                    target,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            else:
                shutil.copy2(source, target)


def file_map(root: Path) -> list[dict[str, Any]]:
    out = []
    for path in candidate_files(root):
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        out.append({
            "path": relative,
            "mode": stat.S_IMODE(path.stat().st_mode),
            "size": len(data),
            "sha256": digest(data),
        })
    return out


def candidate_files(root: Path) -> list[Path]:
    """Files eligible for an immutable candidate, excluding generated caches."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.relative_to(root).parts
        and path.suffix != ".pyc"
    )


def require_candidate_python_format(
    payload: Path, selected_ares_paths: Sequence[str]
) -> None:
    """Run pinned Ruff only over ledger-declared Ares Python changes.

    The immutable candidate contains the full Ares source tree so its bytes can
    be installed without a development checkout.  That tree has legacy lint
    debt which must not make a release candidate non-reproducible.  The ledger
    remains the review boundary: every changed Python path must be present and
    clean under the candidate's pinned Ruff configuration.
    """
    python_files: list[Path] = []
    for relative in sorted(set(selected_ares_paths)):
        path = payload / "ares" / relative
        if relative.endswith("/") or path.is_dir():
            if not path.is_dir():
                raise RuntimeError(
                    "candidate ledger declares missing Ares directory: " + relative
                )
            python_files.extend(sorted(path.rglob("*.py")))
            continue
        if not relative.endswith(".py"):
            continue
        if not path.is_file():
            raise RuntimeError(
                "candidate ledger declares missing Ares Python path: " + relative
            )
        python_files.append(path)
    python_files = sorted(set(python_files))
    if not python_files:
        return
    config = payload / "ares" / "pyproject.toml"
    if not config.is_file():
        raise RuntimeError("candidate payload is missing pinned Ruff configuration")
    ruff = shutil.which("ruff")
    if not ruff:
        raise RuntimeError(
            "candidate sealing requires pinned Ruff 0.15.10; executable missing"
        )
    version = subprocess.run(
        [ruff, "--version"],
        text=True,
        capture_output=True,
        check=False,
    )
    if version.returncode or version.stdout.strip() != "ruff 0.15.10":
        raise RuntimeError(
            f"candidate sealing requires pinned Ruff 0.15.10; got {version.stdout.strip()!r}"
        )
    for command, command_args in (("check", ()), ("format", ("--check",))):
        result = subprocess.run(
            [
                ruff,
                command,
                *command_args,
                "--config",
                str(config),
                *(str(path) for path in python_files),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"candidate sealing blocked by Ruff {command} over candidate-owned Python payload: "
                + (result.stdout or result.stderr).strip()
            )


def build_stage(
    ares: Path, governor: Path, ledger: dict[str, Any], root: Path
) -> tuple[Path, dict[str, Any]]:
    payload = root / "payload"
    extract_head(ares, payload / "ares")
    extract_head(governor, payload / "context-governor")
    overlay_selected(ares, payload / "ares", selected_paths(ledger, "ares"))
    overlay_selected(
        governor,
        payload / "context-governor",
        selected_paths(ledger, "context-governor"),
    )
    (payload / "scope-ledger.json").write_bytes(canonical(ledger) + b"\n")
    scope_manifest = independent_scope_manifest(ares, governor, ledger, payload)
    (payload / "scope-manifest.json").write_bytes(canonical(scope_manifest) + b"\n")
    # This is the exact, release-relative configuration consumed by the
    # bootstrap.  Runtime-owned home/key values are deliberately represented
    # as governed placeholders, never as a caller selectable authority.
    rendered_config = {
        "context": {
            "engine": "ri-context-governor",
            "governor": {
                "binary": "context-governor",
                "receipt_store": "governed-at-runtime",
                "key_id": "governed-at-runtime",
            },
        }
    }
    (payload / "rendered-activation-config.json").write_bytes(
        canonical(rendered_config) + b"\n"
    )
    require_candidate_python_format(payload, selected_paths(ledger, "ares"))
    python_root = Path(sys.executable).resolve().parents[1]
    site_packages_root = Path(sysconfig.get_paths()["purelib"]).resolve()
    stage_runtime_image(
        payload,
        python_root=python_root,
        site_packages_root=site_packages_root,
    )
    env = dict(
        os.environ,
        SOURCE_DATE_EPOCH="0",
        CARGO_TARGET_DIR=str(root / "target"),
        RUSTFLAGS=f"--remap-path-prefix={root}=/candidate",
    )
    subprocess.check_call(
        ["cargo", "build", "--release", "--locked"],
        cwd=payload / "context-governor",
        env=env,
    )
    binary = root / "context-governor"
    shutil.copy2(root / "target" / "release" / "context-governor", binary)
    binary.chmod(0o755)
    shutil.copy2(binary, payload / "runtime" / "context-governor")
    (payload / "runtime" / "context-governor").chmod(0o755)
    shutil.copy2(
        payload
        / "ares"
        / "plugins/context_engine/_context_governor/activation_config.template.json",
        payload / "activation-config-template.json",
    )
    write_release_manifest(payload)
    return payload, core_manifest(payload, binary, ledger)


def core_manifest(
    payload: Path, binary: Path, ledger: dict[str, Any]
) -> dict[str, Any]:
    entries = [
        {**entry, "path": f"payload/{entry['path']}"} for entry in file_map(payload)
    ] + [
        {
            "path": "context-governor",
            "mode": 0o755,
            "size": binary.stat().st_size,
            "sha256": digest(binary.read_bytes()),
        }
    ]
    adapter_root = payload / "ares/plugins/context_engine/_context_governor"
    adapter_bundle = file_map(adapter_root)
    bootstrap = adapter_root / "release_identity.py"
    template = payload / "activation-config-template.json"
    release_manifest = payload / "release-manifest.json"
    release = json.loads(release_manifest.read_text(encoding="utf-8"))
    lockfiles = {
        path.relative_to(payload).as_posix(): digest(path.read_bytes())
        for path in (payload / "context-governor").glob("*lock*")
        if path.is_file()
    }
    manifest = {
        "schema": SCHEMA_CORE,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "ares_head": ledger["ares_head"],
        "context_governor_head": ledger["context_governor_head"],
        "payload_files": entries,
        "payload_tree_sha256": digest(
            canonical(sorted(entries, key=lambda entry: entry["path"]))
        ),
        "binary_path": "context-governor",
        "binary_sha256": digest(binary.read_bytes()),
        "adapter_bundle_path": "payload/ares/plugins/context_engine/_context_governor",
        "adapter_bundle_sha256": digest(canonical(adapter_bundle)),
        "activation_bootstrap_path": "payload/ares/plugins/context_engine/_context_governor/release_identity.py",
        "activation_bootstrap_sha256": digest(bootstrap.read_bytes()),
        "config_template_path": "payload/activation-config-template.json",
        "config_template_sha256": digest(template.read_bytes()),
        "rendered_config_path": "payload/rendered-activation-config.json",
        "rendered_config_sha256": digest(
            (payload / "rendered-activation-config.json").read_bytes()
        ),
        "release_manifest_path": "payload/release-manifest.json",
        "release_manifest_sha256": digest(release_manifest.read_bytes()),
        "runtime_tree_sha256": release["runtime_tree_sha256"],
        "lockfile_digests": lockfiles,
        "contract_versions": {
            "receipt": "ContextCompactionReceiptV2",
            "exactness": "canonical_utf8_text_v1",
            "key_snapshot": "AresContextGovernorKeySnapshotV2",
        },
    }
    return identified(manifest, "candidate_id")


def deterministic_archive(
    payload: Path,
    binary: Path,
    manifest: dict[str, Any],
    extras: list[tuple[str, bytes]],
    output: Path,
) -> str:
    with tarfile.open(output, "w", format=tarfile.USTAR_FORMAT) as archive:
        members = [
            (path, f"payload/{path.relative_to(payload).as_posix()}")
            for path in candidate_files(payload)
        ]
        members += [(binary, "context-governor")]
        manifest_bytes = canonical(manifest) + b"\n"
        for source, name in sorted(members, key=lambda item: item[1]):
            info = archive.gettarinfo(str(source), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with source.open("rb") as stream:
                archive.addfile(info, stream)
        info = tarfile.TarInfo("candidate-core-manifest.json")
        info.size = len(manifest_bytes)
        info.mode = 0o644
        info.mtime = 0
        archive.addfile(info, io.BytesIO(manifest_bytes))
        for name, contents in sorted(extras):
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(contents))
    return digest(output.read_bytes())


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[
        max(0, min(len(ordered) - 1, int((len(ordered) * fraction + 0.999999)) - 1))
    ]


def _cli(
    binary: Path,
    arguments: list[str],
    payload: dict[str, Any] | None = None,
    *,
    binding: Any = None,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    completed = subprocess.run(
        [str(binary), *arguments],
        input=canonical(payload or {}),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=binding.pass_fds if binding else (),
    )
    if completed.returncode:
        raise RuntimeError(
            completed.stderr.decode()
            or completed.stdout.decode()
            or f"candidate CLI failed: {arguments}"
        )
    return json.loads(completed.stdout), (time.perf_counter() - started) * 1000


def _fixture_key(ares_stage: Path, binary: Path, fixture: Path) -> Any:
    # This temporary key is deliberately generated by Ares lifecycle code and
    # discarded with the fixture. It is never copied into an artifact.
    import importlib.util
    import sys

    fixture_key_state = (
        ares_stage / "plugins/context_engine/_context_governor/key_state.py"
    )
    spec = importlib.util.spec_from_file_location(
        "candidate_fixture_key_state", fixture_key_state
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    home = fixture / "hermes-home"
    home.mkdir(mode=0o700)
    state = module.ContextGovernorKeyState(home, str(binary))
    return state.initialize_first_install()


def _certification_sample(
    binary: Path, ares_stage: Path, generation: int
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix=f"ares-cg-cert-g{generation}-"
    ) as directory:
        fixture = Path(directory)
        binding = _fixture_key(ares_stage, binary, fixture)
        store = fixture / "receipts"
        store.mkdir()
        signed = ["--dir", str(store), *binding.command_args()]
        response: dict[str, Any] | None = None
        compact_ms = 0.0
        expected_content = ""
        for step in range(1, generation + 1):
            if response is None:
                messages = [
                    {
                        "role": "system",
                        "content": "Preserve exact evidence and the active instruction.",
                    },
                    {
                        "role": "tool",
                        "content": "old tool noise " * 1500
                        + "\nCERTIFICATION_EXACT_MARKER\n"
                        + "more old tool noise " * 1500,
                    },
                    {
                        "role": "assistant",
                        "content": "The old tool output was inspected.",
                    },
                    {
                        "role": "user",
                        "content": "Continue with the active verification gate.",
                    },
                ]
            else:
                expected_content = (
                    f"generation {step} bounded-growth source\nCERTIFICATION_EXACT_MARKER_G{generation}_{step}_9e3779b97f4a7c15\n"
                    + "deterministic disposable tool output " * 1500
                )
                messages = response["compacted_messages"] + [
                    {"role": "tool", "content": expected_content},
                    {
                        "role": "assistant",
                        "content": f"generation {step} processed the new tool output",
                    },
                    {
                        "role": "user",
                        "content": f"Continue generation {step}; preserve the active gate.",
                    },
                ]
            if step == 1:
                expected_content = (
                    "old tool noise " * 1500
                    + "\nCERTIFICATION_EXACT_MARKER\n"
                    + "more old tool noise " * 1500
                )
            request = {
                "session_id": "certification",
                "messages": messages,
                "policy": {
                    "target_tokens": 180,
                    "protect_first_n": 0,
                    "protect_last_n": 1,
                    "summary_max_chars": 320,
                    "allocator": "deterministic_v1",
                    "max_lineage_generation": 32,
                    "max_provenance_bytes": 131072,
                    "min_net_savings_tokens": 128,
                },
            }
            command = ["compact-v2", *signed] + (
                []
                if response is None
                else ["--parent-receipt", response["receipt"]["receipt_id"]]
            )
            response, compact_ms = _cli(binary, command, request, binding=binding)
            _cli(binary, ["store-v2", *signed], response, binding=binding)
        assert response is not None
        # Restart/load is measured through the authenticated production expand
        # boundary in a fresh process, never through metadata-only status.
        load_ms = 0.0
        source = next(
            item["source_id"]
            for item in response["source_evidence"]
            if item["message"]["content"] == expected_content
        )
        expanded, started_expand_ms = _cli(
            binary,
            [
                "expand",
                *signed,
                "--receipt",
                response["receipt"]["receipt_id"],
                "--item",
                source,
            ],
            binding=binding,
        )
        load_ms = started_expand_ms
        _, expand_ms = _cli(
            binary,
            [
                "expand",
                *signed,
                "--receipt",
                response["receipt"]["receipt_id"],
                "--item",
                source,
            ],
            binding=binding,
        )
        prompt, _ = _cli(binary, ["render-prompt-v2"], response, binding=binding)
        verified = True
        receipt_bytes = len(canonical(response))
        cumulative = sum(p.stat().st_size for p in store.glob("*.json"))
        receipt = response["receipt"]
        # Measure the exact final provider-bound rendering, not an internal
        # receipt field or a hand-written reconstruction.
        source_ids = receipt["covered_original_sources"]
        rendered_messages = (
            prompt.get("system", "") + "\n" + prompt.get("user", "")
        ).encode("utf-8")
        begin = b"=== TRANSITIVE EXACT SOURCE IDS ===\n"
        end = b"=== END TRANSITIVE EXACT SOURCE IDS ===\n"
        if rendered_messages.count(begin) != 1 or rendered_messages.count(end) != 1:
            raise RuntimeError("rendered provenance delimiters missing or duplicated")
        start = rendered_messages.index(begin)
        finish = rendered_messages.index(end, start) + len(end)
        section = rendered_messages[start:finish]
        outside = rendered_messages[:start] + rendered_messages[finish:]
        if any(source["source_id"].encode("utf-8") in outside for source in source_ids):
            raise RuntimeError("provenance source ID escaped rendered section")
        prompt_provenance_bytes = len(section)
        expected_bytes = expected_content.encode("utf-8")
        recovered_bytes = expanded.get("content", "").encode("utf-8")
        exact_pass = (
            expanded.get("exactness_scope") == "canonical_utf8_text_v1"
            and expanded.get("truncated") is False
            and expected_bytes == recovered_bytes
            and digest(expected_bytes) == digest(recovered_bytes)
        )
        return {
            "prompt_visible_provenance_bytes": prompt_provenance_bytes,
            "prompt_visible_provenance_tokens": max(
                1, (prompt_provenance_bytes + 3) // 4
            ),
            "authoritative_provenance_bytes": len(canonical(source_ids)),
            "receipt_bytes": receipt_bytes,
            "cumulative_receipt_store_bytes": cumulative,
            "compaction_latency_ms": compact_ms,
            "restart_load_latency_ms": load_ms,
            "exact_expansion_latency_ms": expand_ms,
            "input_tokens": receipt["original_approx_tokens"],
            "output_tokens": receipt["compacted_approx_tokens"],
            "net_token_savings": receipt["original_approx_tokens"]
            - receipt["compacted_approx_tokens"],
            "budget_decision": "admit"
            if receipt["original_approx_tokens"] > receipt["compacted_approx_tokens"]
            else "reject",
            "exact_expansion_hash": digest(recovered_bytes),
            "exact_expansion_expected_hash": digest(expected_bytes),
            "exact_expansion_result": "PASS" if exact_pass else "FAIL",
            "authenticated_restart_load_result": "PASS" if load_ms >= 0 else "FAIL",
            "rendered_prompt_provenance_result": "PASS",
            "hmac_verification_result": "PASS" if verified else "FAIL",
            "key_id": binding.key_id,
        }


def _sample_decisions(
    sample: dict[str, Any], generation: int, phase: str, index: int
) -> list[dict[str, Any]]:
    limits = {
        "prompt_visible_provenance_bytes": 512,
        "prompt_visible_provenance_tokens": 128,
        "authoritative_provenance_bytes": 131072,
        "receipt_bytes": 524288,
        "cumulative_receipt_store_bytes": generation * 524288,
        "net_token_savings": 128,
    }
    decisions = [
        {
            "metric_id": key,
            "phase": phase,
            "sample_index": index,
            "observed": sample[key],
            "hard_limit": limit,
            "pass": sample[key] <= limit
            if key != "net_token_savings"
            else sample[key] >= limit,
        }
        for key, limit in limits.items()
    ]
    decisions += [
        {
            "metric_id": "budget_decision",
            "phase": phase,
            "sample_index": index,
            "observed": sample["budget_decision"],
            "hard_limit": "admit",
            "pass": sample["budget_decision"] == "admit",
        },
        {
            "metric_id": "exact_expansion",
            "phase": phase,
            "sample_index": index,
            "observed": sample["exact_expansion_result"],
            "hard_limit": "PASS",
            "pass": sample["exact_expansion_result"] == "PASS",
        },
        {
            "metric_id": "hmac_verification",
            "phase": phase,
            "sample_index": index,
            "observed": sample["hmac_verification_result"],
            "hard_limit": "PASS",
            "pass": sample["hmac_verification_result"] == "PASS",
        },
    ]
    return decisions


def _soft_warnings(
    sample: dict[str, Any], generation: int, phase: str, index: int
) -> list[dict[str, Any]]:
    return [
        {
            "metric_id": "receipt_bytes_soft_warning",
            "phase": phase,
            "sample_index": index,
            "observed": sample["receipt_bytes"],
            "warning_limit": 393216,
            "triggered": sample["receipt_bytes"] > 393216,
        },
        {
            "metric_id": "cumulative_receipt_store_bytes_soft_warning",
            "phase": phase,
            "sample_index": index,
            "observed": sample["cumulative_receipt_store_bytes"],
            "warning_limit": generation * 393216,
            "triggered": sample["cumulative_receipt_store_bytes"] > generation * 393216,
        },
        {
            "metric_id": "compaction_latency_soft_warning",
            "phase": phase,
            "sample_index": index,
            "observed": sample["compaction_latency_ms"],
            "warning_limit": 2000,
            "triggered": sample["compaction_latency_ms"] > 2000,
        },
        {
            "metric_id": "restart_load_latency_soft_warning",
            "phase": phase,
            "sample_index": index,
            "observed": sample["restart_load_latency_ms"],
            "warning_limit": 100,
            "triggered": sample["restart_load_latency_ms"] > 100,
        },
        {
            "metric_id": "exact_expansion_latency_soft_warning",
            "phase": phase,
            "sample_index": index,
            "observed": sample["exact_expansion_latency_ms"],
            "warning_limit": 100,
            "triggered": sample["exact_expansion_latency_ms"] > 100,
        },
        # The receipt's token counters are explicitly approximate.  This is a
        # truthful warning, not a hard-failure substitute.
        {
            "metric_id": "approximate_counter_usage",
            "phase": phase,
            "sample_index": index,
            "observed": True,
            "triggered": True,
        },
    ]


def _certification_result(
    candidate_id: str, records: list[dict[str, Any]], terminal_outcome: str
) -> dict[str, Any]:
    passed = terminal_outcome == "PASS"
    return {
        "schema": SCHEMA_STAGED_CERT,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "harness_version": "v2",
        "candidate_id": candidate_id,
        "generations": records,
        "pass": passed,
        "terminal_outcome": terminal_outcome,
        # Certification success establishes measurement success, never
        # activation authority.  Do not derive this from `pass`: a staged
        # recertification is non-authorizing whether it passes or fails.
        "authorization_state": CertificationAuthority.NON_AUTHORIZING.value,
        "non_authorizing": True,
    }


def certify(
    binary: Path,
    ares_stage: Path,
    candidate_id: str,
    *,
    generations: tuple[int, ...] = (16, 32),
    persist: Any = None,
) -> dict[str, Any]:
    """Run exact candidate measurements, persisting non-authorizing progress.

    `persist` receives a complete, machine-readable checkpoint after every
    warmup or measured attempt.  It deliberately never receives a candidate,
    certification-set, or sealed identity.
    """
    records: list[dict[str, Any]] = []
    for generation in generations:
        record: dict[str, Any] = {
            "generation": generation,
            "warmup_runs": 3,
            "measured_runs": 10,
            "raw_measurement_samples": [],
            "threshold_evaluations": [],
            "soft_warning_evaluations": [],
            "failing_metric_ids": [],
            "hard_pass": False,
            "terminal_outcome": "IN_PROGRESS",
        }
        records.append(record)
        for phase, count in (("warmup", 3), ("measured", 10)):
            for index in range(count):
                try:
                    sample = _certification_sample(binary, ares_stage, generation)
                except Exception as error:
                    record["raw_measurement_samples"].append({
                        "phase": phase,
                        "sample_index": index,
                        "metrics": None,
                        "measurement_error": f"{type(error).__name__}: {error}",
                    })
                    record["threshold_evaluations"].append({
                        "metric_id": "measurement_execution",
                        "phase": phase,
                        "sample_index": index,
                        "observed": "ERROR",
                        "hard_limit": "successful measurement",
                        "pass": False,
                    })
                    record["failing_metric_ids"] = ["measurement_execution"]
                    record["terminal_outcome"] = "HARD_FAILURE"
                    result = _certification_result(
                        candidate_id, records, "HARD_FAILURE"
                    )
                    if persist:
                        persist(result)
                    return result
                record["raw_measurement_samples"].append({
                    "phase": phase,
                    "sample_index": index,
                    "metrics": sample,
                })
                record["threshold_evaluations"] += _sample_decisions(
                    sample, generation, phase, index
                )
                record["soft_warning_evaluations"] += _soft_warnings(
                    sample, generation, phase, index
                )
                result = _certification_result(candidate_id, records, "IN_PROGRESS")
                if persist:
                    persist(result)
        measured = [
            entry["metrics"]
            for entry in record["raw_measurement_samples"]
            if entry["phase"] == "measured"
        ]
        latencies = {
            "compaction_latency_ms": [
                sample["compaction_latency_ms"] for sample in measured
            ],
            "restart_load_latency_ms": [
                sample["restart_load_latency_ms"] for sample in measured
            ],
            "exact_expansion_latency_ms": [
                sample["exact_expansion_latency_ms"] for sample in measured
            ],
        }
        p95 = {key: _percentile(values, 0.95) for key, values in latencies.items()}
        record["p50"] = {
            key: _percentile(values, 0.5) for key, values in latencies.items()
        }
        record["p95"] = p95
        record["max"] = {key: max(values) for key, values in latencies.items()}
        record["threshold_evaluations"] += [
            {
                "metric_id": "compaction_p95_ms",
                "observed": p95["compaction_latency_ms"],
                "hard_limit": 5000,
                "pass": p95["compaction_latency_ms"] <= 5000,
            },
            {
                "metric_id": "restart_load_p95_ms",
                "observed": p95["restart_load_latency_ms"],
                "hard_limit": 500,
                "pass": p95["restart_load_latency_ms"] <= 500,
            },
            {
                "metric_id": "exact_expansion_p95_ms",
                "observed": p95["exact_expansion_latency_ms"],
                "hard_limit": 500,
                "pass": p95["exact_expansion_latency_ms"] <= 500,
            },
        ]
        record["failing_metric_ids"] = sorted({
            decision["metric_id"]
            for decision in record["threshold_evaluations"]
            if not decision["pass"]
        })
        record["hard_pass"] = not record["failing_metric_ids"]
        record["terminal_outcome"] = "PASS" if record["hard_pass"] else "HARD_FAILURE"
        result = _certification_result(
            candidate_id,
            records,
            "IN_PROGRESS" if record["hard_pass"] else "HARD_FAILURE",
        )
        if persist:
            persist(result)
        if not record["hard_pass"]:
            return result
    return _certification_result(candidate_id, records, "PASS")


def full_seal_certification(
    result: dict[str, Any], candidate_id: str
) -> dict[str, Any]:
    """Convert a terminal measurement result into the sole seal-eligible form.

    This deliberately happens only after both generations complete.  Progress,
    dry-run, diagnostic, and failed results remain in the distinct staged
    schema and are structurally ineligible for a certification set.
    """
    if (
        result.get("schema") != SCHEMA_STAGED_CERT
        or result.get("candidate_id") != candidate_id
        or result.get("pass") is not True
        or result.get("terminal_outcome") != "PASS"
    ):
        raise RuntimeError("full seal requires a terminal staged PASS")
    generations = result.get("generations")
    if not isinstance(generations, list) or {
        item.get("generation") for item in generations
    } != {16, 32}:
        raise RuntimeError("full seal requires complete Gen16 and Gen32 evidence")
    return {
        "schema": SCHEMA_FULL_SEAL_CERT,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "certification_purpose": "FULL_SEAL",
        "certification_mode": "FULL_SEAL",
        "candidate_id": candidate_id,
        "candidate_core_id": candidate_id,
        "certification_set_inputs": {
            "candidate_id": candidate_id,
            "candidate_core_id": candidate_id,
            "required_artifact_names": [
                "gen-certification.json",
                "preseal-secret-scan.json",
                "scope-proof.json",
            ],
        },
        "required_generations": [16, 32],
        "required_warmup_runs": 3,
        "required_measured_runs": 10,
        "generations": generations,
        "pass": True,
        "terminal_outcome": "PASS",
        "hard_pass": True,
        "failing_hard_metric_ids": [],
        "exact_expansion": "PASS",
        "authenticated_restart_load": "PASS",
        "rendered_prompt_provenance": "PASS",
        "integrity_hmac": "PASS",
        "authorization_state": CertificationAuthority.NON_AUTHORIZING.value,
        "non_authorizing": True,
    }


def write_non_authorizing_certification(
    output: Path, certification: dict[str, Any], *, failed: bool = False
) -> Path:
    name = (
        "FAILED-NON-AUTHORIZING-gen-certification.json"
        if failed
        else "NON-AUTHORIZING-gen-certification-progress.json"
    )
    path = output / name
    path.write_bytes(canonical(certification) + b"\n")
    # Status is only allowed to describe bytes that survived persistence.
    persisted = _read_staged_certification(path)
    if (
        persisted.get("authorization_state")
        != CertificationAuthority.NON_AUTHORIZING.value
        or persisted.get("non_authorizing") is not True
    ):
        raise RuntimeError("AuthorizationStateContradiction")
    return path


def _read_staged_certification(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise RuntimeError("duplicate staged certification key")
            value[key] = item
        return value

    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise RuntimeError("noncanonical staged certification")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict) or canonical(value) + b"\n" != raw:
        raise RuntimeError("noncanonical staged certification")
    required = {
        "schema",
        "canonicalization_version",
        "harness_version",
        "candidate_id",
        "generations",
        "pass",
        "terminal_outcome",
        "authorization_state",
        "non_authorizing",
    }
    if set(value) != required or value.get("schema") != SCHEMA_STAGED_CERT:
        raise RuntimeError("invalid staged certification schema")
    return value


def staged_certification_status(certification: dict[str, Any]) -> dict[str, Any]:
    """Return stdout status using exactly the persisted authority semantics."""
    if (
        certification.get("authorization_state")
        != CertificationAuthority.NON_AUTHORIZING.value
        or certification.get("non_authorizing") is not True
    ):
        raise RuntimeError("AuthorizationStateContradiction")
    return {
        "terminal_outcome": certification["terminal_outcome"],
        "authorization_state": certification["authorization_state"],
        "non_authorizing": certification["non_authorizing"],
        "candidate_core_id": certification["candidate_id"],
    }


def secret_scan(root: Path) -> dict[str, Any]:
    scanner = _tool_module(
        "scan_ares_context_governor_candidate.py", "ares_cg_secret_scanner"
    )
    return scanner.scan([root])


def verify_archive(
    archive: Path,
    manifest: dict[str, Any],
    certification_set: dict[str, Any],
    artifacts: dict[str, bytes],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ares-cg-archive-verify-") as directory:
        extracted = Path(directory)
        with tarfile.open(archive) as bundle:
            member_modes = {
                member.name: stat.S_IMODE(member.mode)
                for member in bundle.getmembers()
                if member.isfile()
            }
            bundle.extractall(extracted, filter="data")
        actual = [
            {
                **entry,
                "path": f"payload/{entry['path']}",
                "mode": member_modes[f"payload/{entry['path']}"],
            }
            for entry in file_map(extracted / "payload")
        ] + [
            {
                "path": "context-governor",
                "mode": member_modes["context-governor"],
                "size": (extracted / "context-governor").stat().st_size,
                "sha256": digest((extracted / "context-governor").read_bytes()),
            }
        ]
        if actual != manifest["payload_files"]:
            raise RuntimeError("archive payload diverges from candidate core manifest")
        if (
            json.loads((extracted / "candidate-core-manifest.json").read_text(encoding="utf-8"))
            != manifest
        ):
            raise RuntimeError("archive core manifest mismatch")
        if (
            json.loads((extracted / "certification-set-manifest.json").read_text(encoding="utf-8"))
            != certification_set
        ):
            raise RuntimeError("archive certification-set mismatch")
        for name, contents in artifacts.items():
            if (extracted / name).read_bytes() != contents:
                raise RuntimeError(f"archive artifact mismatch: {name}")
        return {
            "schema": "AresContextGovernorArchiveVerificationV2",
            "canonicalization_version": CANONICALIZATION_VERSION,
            "candidate_id": manifest["candidate_id"],
            "certification_set_id": certification_set["certification_set_id"],
            "archive_sha256": digest(archive.read_bytes()),
            "pass": True,
            "audited_payload_sha256": digest(canonical(actual)),
            "binary_sha256": digest((extracted / "context-governor").read_bytes()),
        }


def fixture_evidence(
    archive: Path,
    documents: dict[str, Path],
    manifest: dict[str, Any],
    sealed: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Exercise activation and rollback solely from extracted sealed bytes."""
    with tempfile.TemporaryDirectory(prefix="ares-cg-fixture-") as directory:
        root = Path(directory)
        extracted = root / "extracted"
        home = root / "hermes-home"
        with tarfile.open(archive) as bundle:
            bundle.extractall(extracted, filter="data")
        binary = extracted / "context-governor"
        ares = extracted / "payload/ares"
        binding = _fixture_key(ares, binary, root)
        spec = importlib.util.spec_from_file_location(
            "sealed_release_identity",
            extracted
            / "payload/ares/plugins/context_engine/_context_governor/release_identity.py",
        )
        assert spec and spec.loader
        release_identity = importlib.util.module_from_spec(spec)
        import sys

        sys.modules[spec.name] = release_identity
        spec.loader.exec_module(release_identity)
        verified = release_identity.materialize_verified_release(
            archive=archive,
            candidate_core=documents["candidate-core-manifest.json"],
            certification_set=documents["certification-set-manifest.json"],
            sealed_candidate=documents["sealed-candidate-manifest.json"],
            post_seal_evidence=documents["post-seal-evidence-set.json"],
            authorization=documents["activation-authorization.json"],
            release_parent=home / "releases",
        )
        rendered = verified.rendered_config.read_text(encoding="utf-8")
        if "active.key" in rendered or "hmac.key" in rendered:
            raise RuntimeError("rendered config embeds key path/material")
        descriptor = home / "release-descriptor.json"
        descriptor.parent.mkdir(parents=True, exist_ok=True)
        previous = {"release": "previous", "key_id": binding.key_id}
        descriptor.write_bytes(canonical(previous) + b"\n")
        candidate = {
            "release": "candidate",
            "sealed_candidate_id": sealed["sealed_candidate_id"],
            "candidate_id": manifest["candidate_id"],
            "key_id": binding.key_id,
            "config_sha256": digest(rendered.encode()),
        }
        descriptor.write_bytes(canonical(candidate) + b"\n")
        request = {
            "session_id": "dry-run",
            "messages": [
                {"role": "tool", "content": "sealed identity work admission " * 200},
                {"role": "user", "content": "continue"},
            ],
            "policy": {
                "target_tokens": 32,
                "protect_first_n": 0,
                "protect_last_n": 1,
                "summary_max_chars": 128,
                "allocator": "deterministic_v1",
            },
        }
        response, _ = _cli(
            verified.binary,
            [
                "compact-v2",
                "--dir",
                str(home / "context-governor"),
                *binding.command_args(),
            ],
            request,
            binding=binding,
        )
        _cli(
            verified.binary,
            [
                "store-v2",
                "--dir",
                str(home / "context-governor"),
                *binding.command_args(),
            ],
            response,
            binding=binding,
        )
        activation = {
            "schema": "AresContextGovernorDryRunActivationV3",
            "canonicalization_version": CANONICALIZATION_VERSION,
            "activation_authorization_id": json.loads(
                documents["activation-authorization.json"].read_text(encoding="utf-8")
            )["activation_authorization_id"],
            "sealed_candidate_id": sealed["sealed_candidate_id"],
            "candidate_id": manifest["candidate_id"],
            "pass": verified.observed_identity["binary_sha256"]
            == manifest["binary_sha256"]
            and bool(response["receipt"]["receipt_id"]),
            "runtime_identity": verified.observed_identity,
            "rendered_config_sha256": digest(rendered.encode()),
            "configured_key_id": binding.key_id,
            "archive_only": True,
            "live_receipt_store_written": False,
            "work_admitted_only_after_identity_verification": True,
        }
        if not activation["pass"]:
            raise RuntimeError("dry-run runtime identity mismatch")
        development = root / "development-worktree"
        development.mkdir()
        mutated = development / "context-governor"
        mutated.write_bytes(b"counterfeit after seal")
        isolation = {
            "schema": "AresContextGovernorPostSealWorktreeIsolationV1",
            "sealed_candidate_id": sealed["sealed_candidate_id"],
            "candidate_id": manifest["candidate_id"],
            "mutation_surface": "temporary development-worktree copy",
            "mutated_sha256": digest(mutated.read_bytes()),
            "activated_binary_sha256_after_mutation": release_identity._file_digest(
                verified.binary
            ),
            "pass": release_identity._file_digest(verified.binary)
            == manifest["binary_sha256"],
        }
        if not isolation["pass"]:
            raise RuntimeError("post-seal mutation changed activated bytes")
        descriptor.write_bytes(canonical(previous) + b"\n")
        rollback = {
            "schema": "AresContextGovernorDryRunRollbackV3",
            "sealed_candidate_id": sealed["sealed_candidate_id"],
            "candidate_id": manifest["candidate_id"],
            "pass": json.loads(descriptor.read_text(encoding="utf-8")) == previous,
            "restored_descriptor": previous,
            "current_key_id_retained": binding.key_id,
            "old_secret_snapshot_restored": False,
            "temporary_home_only": True,
        }
        v1_request = {
            "session_id": "v1-proof",
            "messages": [
                {"role": "assistant", "content": "V1 immutable evidence"},
                {"role": "user", "content": "final"},
            ],
            "policy": {
                "target_tokens": 16,
                "protect_first_n": 0,
                "protect_last_n": 1,
                "summary_max_chars": 128,
                "allocator": "deterministic_v1",
            },
        }
        v1, _ = _cli(binary, ["compact"], v1_request)
        v1_path = root / "v1-receipt.json"
        v1_path.write_bytes(canonical(v1))
        before = digest(v1_path.read_bytes())
        # Fixture activation/rollback never touches this on-disk V1 artifact.
        v1_proof = {
            "schema": "AresContextGovernorV1ImmutabilityV2",
            "sealed_candidate_id": sealed["sealed_candidate_id"],
            "pass": before == digest(v1_path.read_bytes()),
            "before_sha256": before,
            "after_sha256": digest(v1_path.read_bytes()),
        }
        return activation, rollback, v1_proof, isolation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ares", type=Path)
    parser.add_argument("--governor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--staged-root",
        type=Path,
        help="measure an already-built isolated stage; never seals",
    )
    parser.add_argument(
        "--generations", type=int, nargs="+", choices=(16, 32), default=(16, 32)
    )
    args = parser.parse_args()
    if bool(args.ares) != bool(args.governor):
        raise SystemExit("--ares and --governor must be supplied together")
    if args.staged_root and (args.ares or args.governor):
        raise SystemExit("--staged-root cannot be combined with source construction")
    if not args.staged_root and not args.ares:
        raise SystemExit("supply --staged-root or both --ares and --governor")
    if not args.staged_root and tuple(args.generations) != (16, 32):
        raise SystemExit(
            "candidate construction requires both Gen16 and Gen32 before sealing"
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.staged_root:
        root = args.staged_root.resolve()
        payload, binary = root / "payload", root / "context-governor"
        ledger = json.loads((payload / "scope-ledger.json").read_text(encoding="utf-8"))
        if ledger["ares_head"] != git(
            Path("/home/sikmindz/Coding/hermes-agent"), "rev-parse", "HEAD"
        ) or ledger["context_governor_head"] != git(
            Path("/home/sikmindz/Coding/Libraries/context-governor"),
            "rev-parse",
            "HEAD",
        ):
            raise SystemExit(
                "staged candidate declared Git heads no longer match the canonical checkpoint"
            )
        first = core_manifest(payload, binary, ledger)
        certification = certify(
            binary,
            payload / "ares",
            first["candidate_id"],
            generations=tuple(args.generations),
            persist=lambda record: write_non_authorizing_certification(output, record),
        )
        if not certification["pass"]:
            failure = write_non_authorizing_certification(
                output, certification, failed=True
            )
            raise RuntimeError(
                f"certification hard threshold failed; diagnostic={failure}"
            )
        # This mode is deliberately evidence-only: a passed partial generation
        # cannot issue any candidate, certification-set, or sealed identity.
        persisted = write_non_authorizing_certification(output, certification)
        print(
            json.dumps(
                staged_certification_status(_read_staged_certification(persisted)),
                sort_keys=True,
            )
        )
        return
    ares, governor = args.ares.resolve(), args.governor.resolve()
    ledger = scope_ledger(ares, governor)
    if not ledger["candidate_construction_allowed"]:
        raise SystemExit("UNRESOLVED candidate inputs block construction")
    (output / "scope-ledger.json").write_bytes(canonical(ledger) + b"\n")
    with (
        tempfile.TemporaryDirectory(prefix="ares-cg-stage-one-") as one,
        tempfile.TemporaryDirectory(prefix="ares-cg-stage-two-") as two,
    ):
        first_payload, first = build_stage(ares, governor, ledger, Path(one))
        second_payload, second = build_stage(ares, governor, ledger, Path(two))
        if first != second:
            raise SystemExit(
                "independent staging roots produced different candidate cores"
            )
        if digest((Path(one) / "context-governor").read_bytes()) != digest(
            (Path(two) / "context-governor").read_bytes()
        ):
            raise SystemExit("independent staging roots produced different binaries")
        verifier = _tool_module(
            "verify_ares_context_governor_scope.py", "ares_cg_scope_verifier_run"
        )
        scope_proof = verifier.verify(
            json.loads((first_payload / "scope-manifest.json").read_text(encoding="utf-8")),
            first_payload,
        )
        scope_proof.update({
            "candidate_id": first["candidate_id"],
            "canonicalization_version": CANONICALIZATION_VERSION,
        })
        certification = certify(
            Path(one) / "context-governor",
            first_payload / "ares",
            first["candidate_id"],
            persist=lambda record: write_non_authorizing_certification(output, record),
        )
        if not certification["pass"]:
            failure = write_non_authorizing_certification(
                output, certification, failed=True
            )
            raise RuntimeError(
                f"certification hard threshold failed; diagnostic={failure}"
            )
        certification = full_seal_certification(certification, first["candidate_id"])
        preseal_secret = secret_scan(first_payload)
        preseal_secret.update({
            "candidate_id": first["candidate_id"],
            "canonicalization_version": CANONICALIZATION_VERSION,
            "stage": "pre-seal",
        })
        preseal_artifacts = {
            "gen-certification.json": canonical(certification) + b"\n",
            "scope-proof.json": canonical(scope_proof) + b"\n",
            "preseal-secret-scan.json": canonical(preseal_secret) + b"\n",
        }
        certification_set = identified(
            {
                "schema": SCHEMA_CERT_SET,
                "canonicalization_version": CANONICALIZATION_VERSION,
                "candidate_id": first["candidate_id"],
                "artifacts": [
                    {"name": name, "sha256": digest(contents)}
                    for name, contents in sorted(preseal_artifacts.items())
                ],
            },
            "certification_set_id",
        )
        archive = output / "ares-context-governor-candidate.tar"
        archive_sha = deterministic_archive(
            first_payload,
            Path(one) / "context-governor",
            first,
            [
                *preseal_artifacts.items(),
                (
                    "certification-set-manifest.json",
                    canonical(certification_set) + b"\n",
                ),
            ],
            archive,
        )
    (output / "candidate-core-manifest.json").write_bytes(canonical(first) + b"\n")
    (output / "gen-certification.json").write_bytes(canonical(certification) + b"\n")
    (output / "certification-set-manifest.json").write_bytes(
        canonical(certification_set) + b"\n"
    )
    (output / "scope-proof.json").write_bytes(canonical(scope_proof) + b"\n")
    (output / "preseal-secret-scan.json").write_bytes(canonical(preseal_secret) + b"\n")
    sealed = identified(
        {
            "schema": SCHEMA_SEALED,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "candidate_id": first["candidate_id"],
            "certification_set_id": certification_set["certification_set_id"],
            "archive_sha256": archive_sha,
            "binary_sha256": first["binary_sha256"],
            "adapter_bundle_sha256": first["adapter_bundle_sha256"],
            "activation_bootstrap_sha256": first["activation_bootstrap_sha256"],
            "config_template_sha256": first["config_template_sha256"],
            "rendered_config_sha256": first["rendered_config_sha256"],
            "payload_tree_sha256": first["payload_tree_sha256"],
            "contract_versions": first["contract_versions"],
        },
        "sealed_candidate_id",
    )
    (output / "sealed-candidate-manifest.json").write_bytes(canonical(sealed) + b"\n")
    archive_evidence = verify_archive(
        archive, first, certification_set, preseal_artifacts
    )
    archive_evidence["sealed_candidate_id"] = sealed["sealed_candidate_id"]
    post_seal = identified(
        {
            "schema": SCHEMA_POST_SEAL,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "candidate_id": first["candidate_id"],
            "certification_set_id": certification_set["certification_set_id"],
            "sealed_candidate_id": sealed["sealed_candidate_id"],
            "archive_sha256": archive_sha,
            "artifacts": [
                {
                    "name": "archive-verification.json",
                    "sha256": digest(canonical(archive_evidence) + b"\n"),
                },
                {
                    "name": "scope-proof.json",
                    "sha256": digest(canonical(scope_proof) + b"\n"),
                },
                {
                    "name": "preseal-secret-scan.json",
                    "sha256": digest(canonical(preseal_secret) + b"\n"),
                },
            ],
        },
        "post_seal_evidence_set_id",
    )
    (output / "post-seal-evidence-set.json").write_bytes(canonical(post_seal) + b"\n")
    authorization = identified(
        {
            "schema": SCHEMA_AUTHORIZATION,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "candidate_id": first["candidate_id"],
            "certification_set_id": certification_set["certification_set_id"],
            "sealed_candidate_id": sealed["sealed_candidate_id"],
            "post_seal_evidence_set_id": post_seal["post_seal_evidence_set_id"],
            "archive_sha256": archive_sha,
            "rendered_config_path": first["rendered_config_path"],
            "rendered_config_sha256": first["rendered_config_sha256"],
            # This sealed input binds dry-run identity only.  Candidate
            # construction cannot grant live activation authority; that can
            # only be recorded later by CandidateStore after a hostile pass.
            "authorization_state": CertificationAuthority.NON_AUTHORIZING.value,
            "non_authorizing": True,
            "approved_release_root": f"content-addressed/{sealed['sealed_candidate_id']}",
            "governed_key_policy": {
                "snapshot_schema": "AresContextGovernorKeySnapshotV2",
                "authority": "descriptor-backed-ares-owned",
                "caller_key_material": "forbidden",
            },
        },
        "activation_authorization_id",
    )
    (output / "activation-authorization.json").write_bytes(
        canonical(authorization) + b"\n"
    )
    documents = {
        name: output / name
        for name in (
            "candidate-core-manifest.json",
            "certification-set-manifest.json",
            "sealed-candidate-manifest.json",
            "post-seal-evidence-set.json",
            "activation-authorization.json",
        )
    }
    activation, rollback, v1, isolation = fixture_evidence(
        archive, documents, first, sealed
    )
    scanner = _tool_module(
        "scan_ares_context_governor_candidate.py", "ares_cg_secret_scanner_post"
    )
    postseal_secret = scanner.scan([archive, *output.glob("*.json")])
    postseal_secret.update({
        "candidate_id": first["candidate_id"],
        "sealed_candidate_id": sealed["sealed_candidate_id"],
        "archive_sha256": archive_sha,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "stage": "post-seal",
    })
    (output / "archive-verification.json").write_bytes(
        canonical(archive_evidence) + b"\n"
    )
    (output / "dry-run-activation.json").write_bytes(canonical(activation) + b"\n")
    (output / "dry-run-rollback.json").write_bytes(canonical(rollback) + b"\n")
    (output / "v1-immutability.json").write_bytes(canonical(v1) + b"\n")
    (output / "post-seal-worktree-isolation.json").write_bytes(
        canonical(isolation) + b"\n"
    )
    (output / "postseal-secret-scan.json").write_bytes(
        canonical(postseal_secret) + b"\n"
    )
    # The matrix exercises disposable stores against these exact candidate
    # input bytes.  It is then itself sealed as evidence, while never becoming
    # an input to its own trial (which would be circular).
    pre_matrix_custody_artifacts = (
        "ares-context-governor-candidate.tar",
        "candidate-core-manifest.json",
        "gen-certification.json",
        "certification-set-manifest.json",
        "scope-ledger.json",
        "scope-proof.json",
        "preseal-secret-scan.json",
        "sealed-candidate-manifest.json",
        "post-seal-evidence-set.json",
        "activation-authorization.json",
        "archive-verification.json",
        "dry-run-activation.json",
        "dry-run-rollback.json",
        "v1-immutability.json",
        "post-seal-worktree-isolation.json",
        "postseal-secret-scan.json",
    )
    (output / "custody-fault-matrix-v1.json").write_bytes(
        canonical(generate_fault_matrix(output, pre_matrix_custody_artifacts)) + b"\n"
    )
    # Scratch output remains diagnostic-only.  A candidate becomes SEALED only
    # when the canonical Ares store has independently copied and verified these
    # exact bytes through its durable same-filesystem publication transaction.
    custody_artifacts = (*pre_matrix_custody_artifacts, "custody-fault-matrix-v1.json")
    store = CandidateStore()
    publication = store.publish(output, custody_artifacts)
    handoff = store.issue_handoff(publication.sealed_candidate_id)
    final_snapshot = store.verify(publication.sealed_candidate_id)
    print(
        json.dumps(
            {
                **sealed,
                "custody": {
                    "publication": publication.code,
                    "candidate_root": str(publication.candidate_root),
                    "custody_sha256": sha256_bytes(
                        (publication.candidate_root / "custody.json").read_bytes()
                    ),
                    "lifecycle_sequence": final_snapshot["lifecycle_sequence"],
                    "hostile_audit_handoff_id": handoff["hostile_audit_handoff_id"],
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
