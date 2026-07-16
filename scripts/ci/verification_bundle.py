#!/usr/bin/env python3
"""Suggest local verification commands from changed Hermes paths.

This module is deliberately read-only. It maps path classes to recommended
checks; it never executes those checks, reads file contents, or mutates runtime
state. Unknown inputs fail open by recommending broader hygiene.
"""

from __future__ import annotations

import json
import posixpath
import shlex
from typing import Iterable, Literal, NamedTuple

RiskLevel = Literal["low", "medium", "high"]

_CREDENTIAL_TERMS = (
    "api_key",
    "secret",
    "password",
    "token",
    "passwd",
)


class VerificationCommand(NamedTuple):
    id: str
    command: str
    reason: str
    required: bool = True
    evidence: str = "stdout/stderr and exit code"


class VerificationBundle(NamedTuple):
    changed_paths: tuple[str, ...]
    commands: tuple[VerificationCommand, ...]
    notes: tuple[str, ...]
    risk_level: RiskLevel


def _normalize_path(path: str) -> str:
    p = str(path).strip().replace("\\", "/")
    normalized = posixpath.normpath(p)
    return "" if normalized == "." else normalized


def _normalize_paths(paths: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in paths:
        p = _normalize_path(raw)
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return tuple(out)


def _is_python_path(path: str) -> bool:
    return path.endswith(".py")


def _is_docs_only_path(path: str) -> bool:
    return path.endswith((".md", ".mdx")) or path.startswith(("docs/", "website/"))


def _is_known_path(path: str) -> bool:
    prefixes = (
        ".github/",
        "agent/",
        "apps/desktop/",
        "cli.py",
        "cron/",
        "docs/",
        "gateway/",
        "hermes_cli/",
        "optional-skills/",
        "plugins/",
        "scripts/",
        "skills/",
        "tests/",
        "tools/",
        "ui-tui/",
        "website/",
    )
    exact = {
        "AGENTS.md",
        "CONTRIBUTING.md",
        "package.json",
        "package-lock.json",
        "pyproject.toml",
        "README.md",
        "uv.lock",
    }
    return path in exact or path.startswith(prefixes)


def _add(
    commands: dict[str, VerificationCommand],
    command_id: str,
    command: str,
    reason: str,
    *,
    required: bool = True,
    evidence: str = "stdout/stderr and exit code",
) -> None:
    commands.setdefault(
        command_id,
        VerificationCommand(
            id=command_id,
            command=command,
            reason=reason,
            required=required,
            evidence=evidence,
        ),
    )


def _quote_paths(paths: Iterable[str]) -> str:
    return " ".join(shlex.quote(path) for path in paths)


def scan_added_line_security_hits(diff: str, terms: Iterable[str] = _CREDENTIAL_TERMS) -> list[str]:
    """Return redacted findings for added credential-like assignments.

    The scan is intentionally line-oriented and conservative. It only flags
    added assignment lines when the credential-looking term appears on the
    left-hand side of the assignment. That keeps scanner pattern strings such
    as ``pattern = r"api_key|secret|...\\s*="`` from matching themselves while
    still failing closed for real additions whose assignment target is a
    credential-looking name.
    """

    normalized_terms = tuple(str(term).lower() for term in terms)
    hits: list[str] = []
    for line in diff.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        body = line[1:].strip().lower()
        if "=" not in body:
            continue
        left_hand_side = body.split("=", 1)[0]
        if any(term in left_hand_side for term in normalized_terms):
            hits.append("redacted-hit")
    return hits


def _added_line_security_scan_command() -> str:
    return (
        "python - <<'PY'\n"
        "import codecs\n"
        "import os\n"
        "import stat\n"
        "import subprocess\n"
        "from pathlib import Path\n"
        f"terms={_CREDENTIAL_TERMS!r}\n"
        "hits=[]\n"
        "repo_root=Path.cwd().resolve()\n"
        "git_env=os.environ.copy()\n"
        "repo_env_names={'GIT_DIR','GIT_WORK_TREE','GIT_COMMON_DIR','GIT_OBJECT_DIRECTORY','GIT_ALTERNATE_OBJECT_DIRECTORIES','GIT_INDEX_FILE','GIT_NAMESPACE','GIT_PREFIX','GIT_CEILING_DIRECTORIES','GIT_DISCOVERY_ACROSS_FILESYSTEM','GIT_IMPLICIT_WORK_TREE','GIT_SHALLOW_FILE','GIT_GRAFT_FILE','GIT_REPLACE_REF_BASE','GIT_QUARANTINE_PATH','GIT_NO_REPLACE_OBJECTS'}\n"
        "for name in list(git_env):\n"
        "    if name in repo_env_names or name == 'GIT_CONFIG' or name.startswith('GIT_CONFIG_'):\n"
        "        git_env.pop(name, None)\n"
        "git_env['GIT_OPTIONAL_LOCKS']='0'\n"
        "git_env['GIT_NO_REPLACE_OBJECTS']='1'\n"
        "git_prefix=['git','-c','core.fsmonitor=false']\n"
        "def scan_body(body):\n"
        "    body=body.strip().lower()\n"
        "    if '=' in body:\n"
        "        lhs=body.split('=', 1)[0]\n"
        "        if any(term in lhs for term in terms):\n"
        "            hits.append('redacted-hit')\n"
        "def git_output(args, input_data=None):\n"
        "    try:\n"
        "        proc=subprocess.run(args, cwd=repo_root, env=git_env, input=input_data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30)\n"
        "    except (OSError, subprocess.TimeoutExpired):\n"
        "        hits.append('redacted-hit')\n"
        "        return None\n"
        "    if proc.returncode != 0:\n"
        "        hits.append('redacted-hit')\n"
        "        return None\n"
        "    return proc.stdout\n"
        "def scan_diff(diff):\n"
        "    for line in diff.splitlines():\n"
        "        if line == 'GIT binary patch' or (line.startswith('Binary files ') and line.endswith(' differ')):\n"
        "            hits.append('redacted-hit')\n"
        "        elif line.startswith('+') and not line.startswith('+++'):\n"
        "            scan_body(line[1:])\n"
        "raw_top=git_output([*git_prefix,'rev-parse','--show-toplevel'])\n"
        "repo_bound=False\n"
        "if raw_top is not None:\n"
        "    try:\n"
        "        top_text=raw_top.decode('utf-8').strip()\n"
        "        if not top_text or '\\0' in top_text:\n"
        "            raise ValueError('invalid Git top-level')\n"
        "        repo_bound=Path(top_text).resolve(strict=True) == repo_root\n"
        "    except (OSError, UnicodeError, ValueError):\n"
        "        repo_bound=False\n"
        "if not repo_bound:\n"
        "    hits.append('redacted-hit')\n"
        "    print('credential_like_added_assignments=', hits)\n"
        "    raise SystemExit(1)\n"
        "index_records=git_output([*git_prefix,'ls-files','-v','-z'])\n"
        "unsafe_index=index_records is None or any(record and not record.startswith(b'H ') for record in (index_records or b'').split(b'\\0'))\n"
        "if unsafe_index:\n"
        "    hits.append('redacted-hit')\n"
        "    print('credential_like_added_assignments=', hits)\n"
        "    raise SystemExit(1)\n"
        "index_entries=git_output([*git_prefix,'ls-files','-s','-z'])\n"
        "unsafe_gitlink=index_entries is None or any(record.startswith(b'160000 ') for record in (index_entries or b'').split(b'\\0'))\n"
        "if unsafe_gitlink:\n"
        "    hits.append('redacted-hit')\n"
        "    print('credential_like_added_assignments=', hits)\n"
        "    raise SystemExit(1)\n"
        "inventory=git_output([*git_prefix,'ls-files','--cached','--others','--exclude-standard','-z'])\n"
        "attributes=git_output([*git_prefix,'check-attr','-z','--stdin','filter','ident','working-tree-encoding'], inventory) if inventory is not None else None\n"
        "active_transform=attributes is None\n"
        "if attributes is not None:\n"
        "    parts=attributes.split(b'\\0')\n"
        "    if parts and parts[-1] == b'':\n"
        "        parts.pop()\n"
        "    active_transform=len(parts) % 3 != 0 or any(value not in {b'unspecified', b'unset'} for value in parts[2::3])\n"
        "if active_transform:\n"
        "    hits.append('redacted-hit')\n"
        "    print('credential_like_added_assignments=', hits)\n"
        "    raise SystemExit(1)\n"
        "diff_options=['--no-ext-diff','--no-textconv','--no-color','--output-indicator-new=+','--output-indicator-old=-','--output-indicator-context= ']\n"
        "for args in ([*git_prefix,'diff',*diff_options,'--cached'], [*git_prefix,'diff',*diff_options]):\n"
        "    raw_diff=git_output(args)\n"
        "    if raw_diff is None:\n"
        "        continue\n"
        "    try:\n"
        "        diff=raw_diff.decode('utf-8')\n"
        "        if '\\0' in diff:\n"
        "            raise UnicodeError('NUL byte in tracked diff')\n"
        "    except UnicodeError:\n"
        "        hits.append('redacted-hit')\n"
        "        continue\n"
        "    scan_diff(diff)\n"
        "reparse_mask=getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)\n"
        "raw_paths=git_output([*git_prefix,'ls-files','--others','--exclude-standard','-z']) or b''\n"
        "for raw_path in raw_paths.split(b'\\0'):\n"
        "    if not raw_path:\n"
        "        continue\n"
        "    relative_path=Path(os.fsdecode(raw_path))\n"
        "    if relative_path.is_absolute() or '..' in relative_path.parts:\n"
        "        hits.append('redacted-hit')\n"
        "        continue\n"
        "    path=repo_root / relative_path\n"
        "    try:\n"
        "        current=repo_root\n"
        "        linked=False\n"
        "        for part in relative_path.parts:\n"
        "            if part in {'', '.'}:\n"
        "                continue\n"
        "            current=current / part\n"
        "            metadata=os.lstat(current)\n"
        "            if current.is_symlink() or (getattr(metadata, 'st_file_attributes', 0) & reparse_mask):\n"
        "                linked=True\n"
        "                break\n"
        "        if linked:\n"
        "            hits.append('redacted-hit')\n"
        "            continue\n"
        "        resolved=path.resolve(strict=True)\n"
        "        resolved.relative_to(repo_root)\n"
        "        metadata=os.stat(path, follow_symlinks=False)\n"
        "        if metadata.st_nlink > 1:\n"
        "            hits.append('redacted-hit')\n"
        "            continue\n"
        "        if not path.is_file():\n"
        "            hits.append('redacted-hit')\n"
        "            continue\n"
        "        raw=path.read_bytes()\n"
        "    except (OSError, ValueError):\n"
        "        hits.append('redacted-hit')\n"
        "        continue\n"
        "    try:\n"
        "        if raw.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):\n"
        "            text=raw.decode('utf-32')\n"
        "        elif raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):\n"
        "            text=raw.decode('utf-16')\n"
        "        elif raw.startswith(codecs.BOM_UTF8):\n"
        "            text=raw.decode('utf-8-sig')\n"
        "        else:\n"
        "            text=raw.decode('utf-8')\n"
        "            if '\\0' in text:\n"
        "                raise UnicodeError('NUL byte in untracked file')\n"
        "    except UnicodeError:\n"
        "        hits.append('redacted-hit')\n"
        "        continue\n"
        "    for body in text.splitlines():\n"
        "        scan_body(body)\n"
        "print('credential_like_added_assignments=', hits)\n"
        "raise SystemExit(1 if hits else 0)\n"
        "PY"
    )


def _add_global_hygiene(commands: dict[str, VerificationCommand]) -> None:
    _add(
        commands,
        "git-diff-check",
        "git diff --check",
        "Catch whitespace and patch-format issues before review.",
    )
    _add(
        commands,
        "conflict-marker-scan",
        "python - <<'PY'\nfrom pathlib import Path\nbad=[]\nfor p in Path('.').rglob('*'):\n    if p.is_file() and p.suffix in {'.py','.md','.toml','.yaml','.yml','.json','.ts','.tsx','.cjs'}:\n        text=p.read_text(encoding='utf-8', errors='ignore')\n        for line_no, line in enumerate(text.splitlines(), start=1):\n            if line.startswith('<' * 7 + ' ') or line.startswith('>' * 7 + ' '):\n                bad.append(f'{p}:{line_no}')\nprint('conflict_markers=', bad)\nraise SystemExit(1 if bad else 0)\nPY",
        "Ensure no merge-conflict markers remain in text changes.",
    )
    _add(
        commands,
        "added-line-security-scan",
        _added_line_security_scan_command(),
        "Fail closed on newly added secret-looking assignments without echoing candidate values.",
    )


def suggest_bundle(paths: Iterable[str]) -> VerificationBundle:
    """Return recommended checks for changed Hermes repo paths.

    The mapping is intentionally conservative. Unknown, empty, or CI-config
    inputs fail open by recommending broad checks instead of skipping work.
    """

    changed_paths = _normalize_paths(paths)
    commands: dict[str, VerificationCommand] = {}
    notes: list[str] = []
    risk: RiskLevel = "low"

    if not changed_paths:
        risk = "high"
        notes.append("Empty changed-path input: fail-open with broad baseline checks.")
        _add(
            commands,
            "full-python-tests",
            "python -m pytest tests/ -q -n 0 -o addopts=''",
            "No path scope was provided, so run the broad Python test suite.",
        )
        _add(
            commands,
            "desktop-typecheck",
            "npm --workspace apps/desktop run typecheck",
            "No path scope was provided; include Desktop type checking as a broad guard.",
        )
        _add_global_hygiene(commands)
        return VerificationBundle(changed_paths, tuple(commands.values()), tuple(notes), risk)

    py_paths: list[str] = []
    node_check_paths: list[str] = []
    unknown_paths: list[str] = []

    for path in changed_paths:
        if not _is_known_path(path):
            unknown_paths.append(path)
            continue

        if path.startswith(".github/"):
            risk = "high"
            notes.append(".github change: fail-open with broad Python/Desktop/site checks.")
            _add(
                commands,
                "full-python-tests",
                "python -m pytest tests/ -q -n 0 -o addopts=''",
                "CI config can affect any Python lane; run broad tests.",
            )
            _add(
                commands,
                "desktop-typecheck",
                "npm --workspace apps/desktop run typecheck",
                "CI config can affect frontend lanes; run Desktop type checking.",
            )
            _add(
                commands,
                "site-skill-docs-tests",
                "python -m pytest tests/website/test_generate_skill_docs.py tests/website/test_extract_skills.py -q -n 0 -o addopts=''",
                "CI/doc generation changes can affect generated skill docs.",
            )

        if _is_python_path(path):
            py_paths.append(path)

        if path == "gateway/run.py" or path == "tests/gateway/test_restart_drain.py":
            risk = "medium" if risk != "high" else risk
            _add(
                commands,
                "gateway-restart-drain-pytest",
                "python -m pytest tests/gateway/test_restart_drain.py -q -n 0 -o addopts=''",
                "Gateway restart/drain paths need the focused regression suite.",
            )

        if path == "gateway/session.py" or path.startswith("gateway/platforms/"):
            risk = "medium" if risk != "high" else risk
            _add(
                commands,
                "gateway-focused-pytest",
                "python -m pytest tests/gateway -q -n 0 -o addopts=''",
                "Gateway/session/platform paths need focused gateway regression coverage.",
            )

        if path.startswith("apps/desktop/src/store/"):
            risk = "medium" if risk != "high" else risk
            _add(
                commands,
                "desktop-store-vitest",
                "npm --workspace apps/desktop run test:ui -- src/store/layout.test.ts",
                "Desktop store changes should exercise related store tests when present.",
            )
            _add(
                commands,
                "desktop-typecheck",
                "npm --workspace apps/desktop run typecheck",
                "Desktop TypeScript changes require type checking.",
            )
            _add(
                commands,
                "desktop-eslint",
                "npm --workspace apps/desktop run lint",
                "Desktop TypeScript changes should pass lint if the package script is available.",
                required=False,
            )

        if path.startswith("apps/desktop/electron/") and path.endswith(".cjs"):
            risk = "medium" if risk != "high" else risk
            node_check_paths.append(path)
            _add(
                commands,
                "desktop-typecheck",
                "npm --workspace apps/desktop run typecheck",
                "Electron/Desktop changes should keep package type checks green.",
            )

        if path.startswith("hermes_cli/") or path == "cli.py":
            risk = "medium" if risk != "high" else risk
            _add(
                commands,
                "hermes-cli-focused-pytest",
                "python -m pytest tests/hermes_cli -q -n 0 -o addopts=''",
                "CLI changes need focused hermes_cli tests.",
            )

        if path in {
            "scripts/ci/verification_bundle.py",
            "scripts/suggest_verification_bundle.py",
            "tests/ci/test_verification_bundle.py",
        }:
            risk = "medium" if risk != "high" else risk
            _add(
                commands,
                "verification-bundle-pytest",
                "python -m pytest tests/ci/test_verification_bundle.py -q -n 0 -o addopts=''",
                "Verification bundle helper changes need their focused regression tests.",
            )

        if path in {
            "scripts/scaffold_ulw_ledger.py",
            "tests/scripts/test_scaffold_ulw_ledger.py",
        }:
            risk = "medium" if risk != "high" else risk
            _add(
                commands,
                "ulw-ledger-scaffold-pytest",
                "python -m pytest tests/scripts/test_scaffold_ulw_ledger.py -q -n 0 -o addopts=''",
                "ULW ledger scaffold changes need focused scaffold tests.",
            )

        if path.startswith("tools/"):
            risk = "medium" if risk != "high" else risk
            _add(
                commands,
                "tools-focused-pytest",
                "python -m pytest tests/tools -q -n 0 -o addopts=''",
                "Tool changes need focused tool tests.",
            )

        if path in {"pyproject.toml", "uv.lock"}:
            risk = "high"
            _add(
                commands,
                "dependency-config-checks",
                "python -m pytest tests/hermes_cli/test_config.py tests/hermes_cli/test_provider_config_validation.py -q -n 0 -o addopts=''",
                "Dependency/config changes can affect install and provider validation paths.",
            )
            _add(
                commands,
                "python-hygiene-pytest",
                "python -m pytest tests/ci tests/hermes_cli -q -n 0 -o addopts=''",
                "Dependency/config changes should run Python hygiene-adjacent tests.",
            )

        if path.startswith(("skills/", "optional-skills/")):
            risk = "medium" if risk != "high" else risk
            _add(
                commands,
                "site-skill-docs-tests",
                "python -m pytest tests/website/test_generate_skill_docs.py tests/website/test_extract_skills.py -q -n 0 -o addopts=''",
                "Skill files feed generated docs/tests even when they look prose-only.",
            )

    if unknown_paths:
        risk = "high"
        notes.append(
            "Unknown changed path(s): " + ", ".join(f"`{p}`" for p in unknown_paths) + "; fail-open with Python hygiene."
        )
        _add(
            commands,
            "python-hygiene-pytest",
            "python -m pytest tests/ci tests/hermes_cli -q -n 0 -o addopts=''",
            "Unknown paths may affect Python/runtime behavior; run a broad hygiene subset.",
        )

    if py_paths:
        quoted = _quote_paths(py_paths)
        _add(
            commands,
            "py-compile-changed",
            f"python -m py_compile {quoted}",
            "Changed Python files should compile before tests/review.",
        )

    if node_check_paths:
        quoted = _quote_paths(node_check_paths)
        _add(
            commands,
            "node-check-electron-cjs",
            f"node --check -- {quoted}",
            "Changed Electron CommonJS files should parse under Node.",
        )

    if risk == "low" and not all(_is_docs_only_path(path) for path in changed_paths):
        risk = "medium"

    _add_global_hygiene(commands)
    return VerificationBundle(changed_paths, tuple(commands.values()), tuple(notes), risk)


def bundle_to_dict(bundle: VerificationBundle) -> dict[str, object]:
    return {
        "changed_paths": list(bundle.changed_paths),
        "risk_level": bundle.risk_level,
        "notes": list(bundle.notes),
        "commands": [
            {
                "id": cmd.id,
                "command": cmd.command,
                "reason": cmd.reason,
                "required": cmd.required,
                "evidence": cmd.evidence,
            }
            for cmd in bundle.commands
        ],
    }


def format_json(bundle: VerificationBundle) -> str:
    return json.dumps(bundle_to_dict(bundle), indent=2, sort_keys=True) + "\n"


def format_markdown(bundle: VerificationBundle) -> str:
    lines: list[str] = ["# Verification Bundle", ""]
    lines.append(f"Risk level: `{bundle.risk_level}`")
    lines.append("")
    lines.append("## Changed paths")
    if bundle.changed_paths:
        lines.extend(f"- `{path}`" for path in bundle.changed_paths)
    else:
        lines.append("- _none provided_")
    if bundle.notes:
        lines.append("")
        lines.append("## Notes")
        lines.extend(f"- {note}" for note in bundle.notes)
    lines.append("")
    lines.append("## Recommended verification")
    for index, cmd in enumerate(bundle.commands, start=1):
        required = "required" if cmd.required else "optional"
        lines.append(f"{index}. `{cmd.id}` ({required})")
        lines.append("   - Command:")
        lines.append("     ```bash")
        lines.extend(f"     {line}" for line in cmd.command.splitlines())
        lines.append("     ```")
        lines.append(f"   - Why: {cmd.reason}")
        lines.append(f"   - Evidence: {cmd.evidence}")
    lines.append("")
    lines.append("_This helper only suggests checks; it does not execute them._")
    return "\n".join(lines) + "\n"
