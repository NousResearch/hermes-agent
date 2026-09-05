"""Regression contract for specialized Install & Update E2E acceptance."""

from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS = REPO_ROOT / "tests" / "install" / "install-update-e2e.sh"
REUSABLE = REPO_ROOT / ".github" / "workflows" / "install-e2e-run.yml"
SCHEDULED = REPO_ROOT / ".github" / "workflows" / "install-e2e.yml"
ACCEPTANCE = REPO_ROOT / ".github" / "workflows" / "install-e2e-acceptance.yml"


def test_harness_exposes_explicit_browser_acceptance_mode() -> None:
    result = subprocess.run(
        ["bash", str(HARNESS), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--include-browser" in result.stdout
    text = HARNESS.read_text()
    assert "--include-browser) INCLUDE_BROWSER=true" in text
    assert '[ "$INCLUDE_BROWSER" = false ]' in text


def test_flag_probes_cannot_false_negative_via_pipefail_sigpipe() -> None:
    text = HARNESS.read_text()

    # `grep -q` exits after the first match. With `set -o pipefail`, piping a
    # large producer into it can report the producer's SIGPIPE instead of the
    # successful match and silently change which installer flags are exercised.
    assert "| grep -qF" not in text
    assert 'grep -qF -- "$flag" <<<"$help"' in text
    assert 'grep -qF -- "$flag" <<<"$script"' in text


def test_installer_authorities_keep_browser_and_npm_in_scope() -> None:
    reusable = REUSABLE.read_text()
    scheduled = SCHEDULED.read_text()
    acceptance = ACCEPTANCE.read_text()

    assert "include-browser:" in reusable
    assert "args+=(--include-browser)" in reusable
    assert "include-browser: true" in scheduled
    assert "include-browser: ${{ matrix.route == 'installer' }}" in acceptance


def test_pull_request_acceptance_executes_both_flag_probe_routes() -> None:
    acceptance = ACCEPTANCE.read_text()

    assert "route: [installer, update]" in acceptance
    assert "route: ${{ matrix.route }}" in acceptance
    assert "install-ref: ${{ matrix.install-ref }}" in acceptance
    assert "include-browser: ${{ matrix.route == 'installer' }}" in acceptance
    assert "include-browser: true" not in acceptance


def test_pull_request_receipt_binds_to_the_submitted_head() -> None:
    reusable = REUSABLE.read_text()
    acceptance = ACCEPTANCE.read_text()
    head_expression = (
        "${{ github.event.pull_request.head.sha || github.sha }}"
    )

    assert f"ref: {head_expression}" in reusable
    assert f"EXPECTED_SHA: {head_expression}" in reusable
    assert "steps.candidate.outputs.sha" in reusable
    assert f"ref: {head_expression}" in acceptance
    assert (
        'git fetch --force --tags --no-recurse-submodules \\\n'
        '            "https://github.com/${GITHUB_REPOSITORY}.git"'
        in acceptance
    )


def test_uv_fixture_does_not_hardcode_the_runner_architecture() -> None:
    text = HARNESS.read_text()

    assert 'case "$(uname -m)" in' in text
    assert 'UV_TARGET="x86_64-unknown-linux-gnu"' in text
    assert 'UV_TARGET="aarch64-unknown-linux-gnu"' in text
    assert '_uv_rel="github/uv/releases/download/$UV_VER/uv-$UV_TARGET.tar.gz"' in text
    assert "uv-x86_64-unknown-linux-gnu.tar.gz" not in text
