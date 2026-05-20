import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCKFILE_FIX = ROOT / ".github" / "workflows" / "nix-lockfile-fix.yml"
NIX_WORKFLOW = ROOT / ".github" / "workflows" / "nix.yml"
OSV_WORKFLOW = ROOT / ".github" / "workflows" / "osv-scanner.yml"


def _job_section(text: str, job_name: str) -> str:
    marker = f"\n  {job_name}:\n"
    start = text.index(marker)
    next_job = text.find("\n  ", start + len(marker))
    while next_job != -1:
        line_end = text.find("\n", next_job + 1)
        line = text[next_job + 1 : line_end if line_end != -1 else len(text)]
        if line.startswith("  ") and not line.startswith("    ") and line.rstrip().endswith(":"):
            return text[start:next_job]
        next_job = text.find("\n  ", next_job + 1)
    return text[start:]


def test_pr_lockfile_generation_runs_without_secrets_or_write_credentials():
    text = LOCKFILE_FIX.read_text()
    generate = _job_section(text, "generate_pr_fix_patch")

    assert "permissions:\n      contents: read\n      pull-requests: read" in generate
    assert "persist-credentials: false" in generate
    assert "secrets.CACHIX_AUTH_TOKEN" not in generate
    assert "./.github/actions/nix-setup" not in generate
    assert "nix run .#fix-lockfiles" in generate
    assert "HASH_LINE = re.compile" in generate


def test_pr_lockfile_write_job_does_not_execute_pr_controlled_code():
    text = LOCKFILE_FIX.read_text()
    apply = _job_section(text, "apply_pr_fix")

    assert "permissions:\n      contents: write" in apply
    assert "nix run .#fix-lockfiles" not in apply
    assert "./.github/actions/nix-setup" not in apply
    assert "CACHIX_AUTH_TOKEN" not in apply
    assert "git apply patch/lockfile-fix.patch" in apply
    assert "HASH_LINE = re.compile" in apply
    assert "TARGET_REF: ${{ needs.resolve_pr_fix.outputs.target_ref }}" in apply
    assert 'git push origin "HEAD:${TARGET_REF}"' in apply
    assert "git push origin HEAD:${{ needs.resolve_pr_fix.outputs.target_ref }}" not in apply


def test_pr_lockfile_fork_prs_do_not_receive_write_token_pushes():
    text = LOCKFILE_FIX.read_text()

    assert "can_push" in _job_section(text, "resolve_pr_fix")
    assert "assertSafeRef(pr.head.ref)" in text
    assert "ref.startsWith('-')" in text
    assert "pr.head.repo.full_name === baseFullName ? 'true' : 'false'" in text
    assert "needs.resolve_pr_fix.outputs.can_push == 'true'" in _job_section(text, "apply_pr_fix")
    assert "needs.resolve_pr_fix.outputs.can_push != 'true'" in _job_section(text, "comment_pr_patch_available")


def test_lockfile_patch_validation_rejects_metadata_changes():
    text = LOCKFILE_FIX.read_text()
    generate = _job_section(text, "generate_pr_fix_patch")
    apply = _job_section(text, "apply_pr_fix")

    for section in (generate, apply):
        assert "old mode " in section
        assert "new mode " in section
        assert "GIT binary patch" in section
        assert "Patch changes non-hash content or unsafe metadata" in section
        assert "INDEX_LINE" in section


def test_nix_pr_checks_do_not_load_local_actions_or_cachix_secret():
    text = NIX_WORKFLOW.read_text()

    assert "./.github/actions/nix-setup" not in text
    assert "persist-credentials: false" in text
    assert "github.event_name == 'push'" in text
    assert "authToken: ${{ secrets.CACHIX_AUTH_TOKEN }}" in text
    assert "github.event_name == 'pull_request'" in text


def test_nix_setup_uses_pinned_external_action_in_ci():
    for workflow in (LOCKFILE_FIX, NIX_WORKFLOW):
        text = workflow.read_text()
        assert "cachix/install-nix-action@08dcb3a5e62fa31e2da3d490afc4176ef55ecd72 # v30" in text
        assert "./.github/actions/nix-setup" not in text


def test_osv_scanner_covers_all_tracked_lockfiles():
    text = OSV_WORKFLOW.read_text()
    tracked = subprocess.run(
        ["git", "ls-files", "uv.lock", "*package-lock.json"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()

    # OSV only supports lockfile inputs here; every tracked uv/package lockfile
    # should be listed explicitly so new dependency roots are not silently missed.
    for lockfile in tracked:
        assert f"--lockfile={lockfile}" in text
