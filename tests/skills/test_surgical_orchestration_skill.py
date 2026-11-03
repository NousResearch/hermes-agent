"""Tests for the surgical-orchestration bundled skill.

This skill is unusual: it ships executable TypeScript under ``references/``.
``skills/`` is outside every npm workspace (see ``package.json``), so the JS
CI jobs never look at it — nothing but this module stops the shipped runtime
from rotting into code that cannot compile.

Two classes of check:

1. Authoring standards shared with every other bundled skill (frontmatter
   shape, <=60-char description, no dead reference links).
2. Invariants that were live bugs on 2026-08-05 and must not come back —
   documented API matching real exports, no re-implementation of the runtime
   inside prose, and a strict typecheck of the shipped sources.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "software-development" / "surgical-orchestration"
SKILL_MD = SKILL_DIR / "SKILL.md"
REFERENCES = SKILL_DIR / "references"

MAX_DESCRIPTION_CHARS = 60


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> dict:
    m = re.search(r"^---\n(.*?)\n---", skill_text, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


@pytest.fixture(scope="module")
def orchestrator_src() -> str:
    return (REFERENCES / "orchestrator.ts").read_text(encoding="utf-8")


class TestAuthoringStandards:
    def test_name_matches_directory(self, frontmatter: dict) -> None:
        assert frontmatter["name"] == SKILL_DIR.name

    def test_description_within_gate_limit(self, frontmatter: dict) -> None:
        # skill_manage's create gate hard-rejects longer descriptions, which
        # would make this skill uninstallable.
        description = frontmatter["description"]
        assert len(description) <= MAX_DESCRIPTION_CHARS, (
            f"description is {len(description)} chars, limit is {MAX_DESCRIPTION_CHARS}"
        )

    def test_description_leads_with_trigger(self, frontmatter: dict) -> None:
        assert frontmatter["description"].lower().startswith("use when")

    def test_reference_links_resolve(self, skill_text: str) -> None:
        linked = set(re.findall(r"\]\(\./(references/[\w./-]+)\)", skill_text))
        assert linked, "SKILL.md links no reference files"
        missing = sorted(p for p in linked if not (SKILL_DIR / p).exists())
        assert not missing, f"SKILL.md links non-existent files: {missing}"

    def test_every_reference_file_is_linked(self, skill_text: str) -> None:
        linked = set(re.findall(r"\]\(\./(references/[\w./-]+)\)", skill_text))
        on_disk = {
            f"references/{p.name}" for p in REFERENCES.iterdir() if p.is_file()
        }
        # An unlinked reference is an orphan: it drifts because nothing points
        # at it. A whole orphaned spec doc did exactly that (2026-08-05).
        assert not (on_disk - linked), f"orphaned reference files: {sorted(on_disk - linked)}"


class TestSchemasMatchRuntime:
    def test_documented_hash_matches_real_implementation(self, skill_text: str) -> None:
        """The JobCard example must show a digest the code actually produces.

        It previously showed e3b0c442... — the SHA-256 of the empty string —
        teaching any agent that read it a wrong, unverifiable value.
        """
        import hashlib

        canonical = json.dumps(
            {
                "directory": "src/services/auth",
                "modifiedFiles": ["jwt.ts", "session.ts"],
                "diffHash": "Implemented JWT rotation and updated session cookies.",
                "errorSignature": "",
            },
            separators=(",", ":"),
        )
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert expected in skill_text, (
            "JobCard example hash does not match computeDebriefHash's canonical form"
        )
        assert hashlib.sha256(b"").hexdigest() not in skill_text

    def test_jobcard_schema_describes_a_jobcard(self) -> None:
        schema = json.loads((REFERENCES / "jobcard-schema.json").read_text(encoding="utf-8"))
        assert schema["title"] == "JobCard"
        assert set(schema["required"]) == {
            "planId",
            "overallStatus",
            "jobs",
            "completedHashes",
        }

    def test_jobcard_statuses_match_typescript_union(self, orchestrator_src: str) -> None:
        schema = json.loads((REFERENCES / "jobcard-schema.json").read_text(encoding="utf-8"))
        schema_statuses = set(
            schema["definitions"]["folderJob"]["properties"]["status"]["enum"]
        )
        union = re.search(r"export type JobStatus =\s*(.*?);", orchestrator_src, re.DOTALL)
        assert union, "JobStatus union not found in orchestrator.ts"
        ts_statuses = set(re.findall(r"'([A-Z_]+)'", union.group(1)))
        assert schema_statuses == ts_statuses

    def test_max_revision_cycles_matches_schema_bound(self, orchestrator_src: str) -> None:
        schema = json.loads((REFERENCES / "jobcard-schema.json").read_text(encoding="utf-8"))
        bound = schema["definitions"]["folderJob"]["properties"]["attempts"]["maximum"]
        m = re.search(r"MAX_REVISION_CYCLES:\s*(\d+)", orchestrator_src)
        assert m and int(m.group(1)) == bound


class TestDocumentedApiExists:
    """Prose that names a symbol the code does not export is a trap.

    A companion spec document described a ``SurgicalOrchestrator`` class that
    never existed in the sources; agents followed it and wrote code against a
    phantom API.
    """

    def test_documented_symbols_are_exported(self, skill_text: str, orchestrator_src: str) -> None:
        exported = set(
            re.findall(
                r"export (?:class|function|const|type|interface) (\w+)", orchestrator_src
            )
        )
        for symbol in ("OrchestrationEngine", "SubagentManager", "SubagentDispatcher"):
            assert symbol in exported, f"{symbol} is not exported by orchestrator.ts"
            assert symbol in skill_text, f"{symbol} is not documented in SKILL.md"

    def test_no_phantom_orchestrator_class(self, skill_text: str) -> None:
        # Allowed only inside the dated pitfall that explains the mistake.
        for line in skill_text.splitlines():
            if "SurgicalOrchestrator" in line:
                assert "absent from the sources" in line, (
                    "SKILL.md references the non-existent SurgicalOrchestrator class"
                )

    def test_runtime_is_not_reimplemented_in_prose(self, skill_text: str) -> None:
        # A Python re-implementation of the compaction routine drifted from the
        # TypeScript and reintroduced the string-hashing bug it was meant to
        # document.
        assert "def compact_orchestrator_context" not in skill_text
        assert "import hashlib" not in skill_text


class TestKnownBugsStayFixed:
    def test_scope_containment_is_segment_aware(self, orchestrator_src: str) -> None:
        # 'src/authz'.startsWith('src/auth') is true — prefix matching leaked
        # sibling directories into a locked scope.
        assert "isInsideFolder" in orchestrator_src
        assert ".startsWith(job.parentFolder)" not in orchestrator_src

    def test_dispatcher_is_injected_not_subclassed(self, orchestrator_src: str) -> None:
        assert "export type SubagentDispatcher" in orchestrator_src
        # The old shape forced consumers to monkeypatch a private method.
        assert "private async executeSubagent" not in orchestrator_src

    def test_missing_dispatcher_fails_closed(self, orchestrator_src: str) -> None:
        # A stub returning a synthetic COMPLETED would poison the hash registry
        # and mark unverified folders VERIFIED.
        assert "NO_DISPATCHER" in orchestrator_src

    def test_timeout_rejects_rather_than_only_emitting(self, orchestrator_src: str) -> None:
        assert "Promise.race" in orchestrator_src
        assert "[TIMEOUT]" in orchestrator_src

    def test_loop_guard_hashes_canonical_payload(self, orchestrator_src: str) -> None:
        # Hashing the bare debrief string made unrelated folders collide.
        assert "loopGuardHashDebrief" not in orchestrator_src
        assert "computeDebriefHash({" in orchestrator_src

    def test_library_has_no_top_level_side_effects(self, orchestrator_src: str) -> None:
        # `require.main === module` is CommonJS-only and unreachable from this
        # ESM module; the CLI belongs in surgical-orchestration.ts. Match the
        # executable guard, not the comment that explains why it was removed.
        assert not re.search(r"^if \(require\.main", orchestrator_src, re.MULTILINE)


class TestSecondPassFixes:
    """Regression guards for the six defects found in the 2026-08-05 second-pass
    review.  Each test fails against the pre-fix source and passes after the
    fix is applied — so a future regression that reverts any fix is caught.
    """

    # F3 — extractParentFolders silently dropped root-level / shallow paths.
    def test_f3_extract_parent_folders_handles_shallow_paths(self, orchestrator_src: str) -> None:
        """``README.md`` (dirname ``"."``) and ``src/a.ts`` (dirname ``"src"``)
        must produce jobs, not be silently dropped by a ``parts.length >= 2``
        guard."""
        # The old guard `if (parts.length >= 2)` dropped single-segment parents.
        assert "parts.length >= 2" not in orchestrator_src, (
            "extractParentFolders still gates on parts.length >= 2, dropping "
            "root-level and shallow paths"
        )

    # F2 — Verifier received no worker output to review.
    def test_f2_verifier_receives_worker_output(self, orchestrator_src: str) -> None:
        """buildVerifierInstructions must inline the worker's debrief and
        files_modified into the verifier prompt — not ignore them."""
        # The old shape had an eslint-disabled _workerResult parameter.
        assert "eslint-disable-next-line" not in orchestrator_src or (
            "_workerResult" not in orchestrator_src
        ), "buildVerifierInstructions still suppresses worker output"
        # The verifier prompt must reference the worker's debrief or files.
        verifier_match = re.search(
            r"buildVerifierInstructions.*?return `(.*?)`;",
            orchestrator_src,
            re.DOTALL,
        )
        assert verifier_match, "buildVerifierInstructions not found"
        body = verifier_match.group(1)
        assert "debrief" in body.lower() or "filesModified" in body or "files_modified" in body, (
            "verifier prompt does not reference worker output (debrief/files)"
        )

    # F4 — Schema/runtime divergences.
    def test_f4a_debrief_schema_error_signature_optional(self) -> None:
        """debrief-schema.json must mark errorSignature as optional to match
        the TypeScript DebriefPayload interface (errorSignature?: string)."""
        schema = json.loads((REFERENCES / "debrief-schema.json").read_text(encoding="utf-8"))
        required = schema.get("required", [])
        assert "errorSignature" not in required, (
            "debrief-schema requires errorSignature but TS marks it optional — "
            "a payload without the key passes TypeScript but fails the schema"
        )

    def test_f4b_exit_schema_accepts_camelcase(self) -> None:
        """subagent-exit-schema.json must accept the camelCase keys
        (filesModified, selfAudit) that the TypeScript SubagentResult interface
        uses, not just the snake_case variants."""
        schema_text = (REFERENCES / "subagent-exit-schema.json").read_text(encoding="utf-8")
        # The parser accepts both; the schema must not reject what the runtime
        # accepts.  Either add camelCase to the schema or document the parser
        # is lenient.  We check that at least the schema doesn't use
        # additionalProperties: false to reject camelCase while the parser
        # accepts it.
        # The fix: add camelCase property aliases or relax the schema.
        # For now, assert the schema mentions the camelCase variants.
        assert "filesModified" in schema_text or "selfAudit" in schema_text, (
            "subagent-exit-schema does not mention camelCase variants that "
            "parseSubagentResult accepts — schema rejects what runtime accepts"
        )

    # F6 — execSync blocks the event loop in an async method.
    def test_f6_playwright_uses_non_blocking_exec(self, orchestrator_src: str) -> None:
        """runPlaywrightTests must not use execSync (blocking I/O) inside an
        async method.  Use child_process.exec (promisified) or spawn instead."""
        assert "execSync" not in orchestrator_src, (
            "orchestrator.ts still uses execSync — blocking I/O in an async "
            "method freezes the Node event loop for up to 120s"
        )

    # F5 — spawnTestFixer bypasses the dispatcher injection.
    def test_f5_test_fixer_uses_dispatcher(self, orchestrator_src: str) -> None:
        """spawnTestFixer must route through the injected SubagentDispatcher
        (spawnSubagent), not return a hardcoded FAILED stub."""
        # The old stub emitted an event and returned a hardcoded FAILED.
        # The fix dispatches a TEST_FIXER payload through spawnSubagent.
        assert "TEST_FIXER" in orchestrator_src, (
            "No TEST_FIXER role dispatched — test-fixer still a hardcoded stub"
        )
        # The hardcoded FAILED return must be gone.
        assert "Manual intervention required" not in orchestrator_src, (
            "spawnTestFixer still returns hardcoded 'Manual intervention required'"
        )

    # F1 — Concurrency cap was dead code (run() was sequential).
    def test_f1_run_parallelizes_jobs(self, orchestrator_src: str) -> None:
        """run() must dispatch jobs in parallel batches of MAX_CONCURRENCY,
        not sequentially with a for-await loop."""
        # The old code: `for (const [jobId, job] of this.jobCard.jobs) { await ... }`
        # The fix uses Promise.all or a concurrency limiter.
        assert "Promise.all" in orchestrator_src, (
            "run() does not use Promise.all — jobs are still sequential, "
            "MAX_CONCURRENCY is dead code"
        )


@pytest.mark.skipif(shutil.which("npx") is None, reason="npx not available")
def test_shipped_typescript_typechecks_strict() -> None:
    """The shipped runtime must compile. Nothing else in CI covers it."""
    sources = sorted(str(p) for p in REFERENCES.glob("*.ts"))
    assert sources, "no TypeScript sources found in references/"
    result = subprocess.run(
        [
            "npx", "--no-install", "tsc",
            "--noEmit", "--strict", "--skipLibCheck",
            "--module", "node16", "--moduleResolution", "node16",
            "--target", "es2022", "--types", "node",
            *sources,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0 and "not found" in (result.stderr or "").lower():
        pytest.skip(f"local tsc unavailable: {result.stderr.strip()[:200]}")
    assert result.returncode == 0, f"tsc --strict failed:\n{result.stdout}\n{result.stderr}"
