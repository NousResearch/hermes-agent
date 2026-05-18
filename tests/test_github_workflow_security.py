from pathlib import Path


def test_nix_lockfile_fix_does_not_run_pr_controlled_local_action() -> None:
    workflow = Path(".github/workflows/nix-lockfile-fix.yml").read_text()
    _, pr_fix_job = workflow.split("  # ── PR fix (manual / checkbox)", maxsplit=1)

    assert "uses: ./.github/actions/nix-setup" not in pr_fix_job
    assert (
        "DeterminateSystems/nix-installer-action@ef8a148080ab6020fd15196c2084a2eea5ff2d25"
        in pr_fix_job
    )
    assert "cachix/cachix-action@1eb2ef646ac0255473d23a5907ad7b04ce94065c" in pr_fix_job
