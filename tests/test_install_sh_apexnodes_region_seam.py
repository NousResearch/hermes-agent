"""Seam-test for the ApexNodes region-detect / China-mirror overlay.

The region self-detection + domestic mirror / COS source downgrade was extracted
out of the hot install.sh / install.ps1 files into our own namespace under
scripts/lib/apexnodes-region-detect.{sh,ps1} (see apex_overlay/README.md and
docs/OVERLAY-SEAM-AUDIT.md). The installer keeps only a self-locating `source`
hook plus a few one-line call sites.

A `source`-based seam is only safe while three invariants hold, and an upstream
bump (or a careless edit) could silently break any of them — turning the China
install path into a quiet fall-through to blocked github.com / pypi.org. This
test makes each invariant a LOUD failure instead:

  1. The installer still sources the lib (the seam is wired, not disarmed).
  2. The lib still defines the public functions the installer calls.
  3. The lib still ships as a sibling of the installer in EVERY materialized
     copy that carries our CN logic (desktop bundle + PyPI wheel), since the
     installer sources it from a path relative to its own on-disk location.

It deliberately does NOT re-test the detection heuristic itself (that lives in
the lib and is covered by its own behavioral smoke); it pins the SEAM.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"
LIB_SH = REPO_ROOT / "scripts" / "lib" / "apexnodes-region-detect.sh"
LIB_PS1 = REPO_ROOT / "scripts" / "lib" / "apexnodes-region-detect.ps1"

# Public surface the installers call. If the lib renames any of these, the
# installer's command-guarded call sites become permanent no-ops (CN silently
# off) — so pin the names here.
SH_PUBLIC_FUNCS = (
    "apexnodes_resolve_region",
    "apexnodes_apply_cn_mirror_env",
    "apexnodes_cn_enabled",
    "apexnodes_install_uv_from_cos",
    "apexnodes_download_runtime_tarball",
)
PS1_PUBLIC_FUNCS = (
    "Resolve-ApexRegion",
    "Set-ApexCnMirrorEnv",
    "Test-CnEnabled",
    "Install-UvFromCos",
    "Install-RuntimeFromCos",
)


# --------------------------------------------------------------------------
# 1. The lib files exist and define their public surface.
# --------------------------------------------------------------------------
def test_region_lib_files_exist() -> None:
    assert LIB_SH.is_file(), f"missing overlay seam lib: {LIB_SH}"
    assert LIB_PS1.is_file(), f"missing overlay seam lib: {LIB_PS1}"


def test_sh_lib_defines_public_functions() -> None:
    text = LIB_SH.read_text()
    for fn in SH_PUBLIC_FUNCS:
        assert f"{fn}()" in text, f"{LIB_SH.name} must define {fn}()"


def test_ps1_lib_defines_public_functions() -> None:
    text = LIB_PS1.read_text()
    for fn in PS1_PUBLIC_FUNCS:
        assert f"function {fn}" in text, f"{LIB_PS1.name} must define function {fn}"


# --------------------------------------------------------------------------
# 2. The installers source the lib (seam wired) and call its functions.
# --------------------------------------------------------------------------
def test_install_sh_sources_region_lib() -> None:
    text = INSTALL_SH.read_text()
    # Self-locating source: derive the script dir, then source lib/<file> from it.
    assert "lib/apexnodes-region-detect.sh" in text, "install.sh must reference the seam lib path"
    assert 'BASH_SOURCE[0]' in text or "$0" in text, "install.sh must self-locate to find the lib"
    assert ". \"$_APEX_REGION_LIB\"" in text, "install.sh must source the seam lib"
    # Present-guard so a lib-absent context (curl|bash from upstream) is a no-op,
    # not a hard error.
    assert 'if [ -f "$_APEX_REGION_LIB" ]' in text, "install.sh must guard the source on file presence"


def test_install_sh_calls_lib_functions_with_guard() -> None:
    text = INSTALL_SH.read_text()
    # Each call site must be guarded by `command -v <fn>` so the upstream
    # (lib-absent) path stays a clean no-op.
    for fn in (
        "apexnodes_resolve_region",
        "apexnodes_install_uv_from_cos",
        "apexnodes_cn_enabled",
        "apexnodes_download_runtime_tarball",
    ):
        assert fn in text, f"install.sh must call {fn}"
        assert f"command -v {fn}" in text, f"install.sh must guard the {fn} call site with command -v"


def test_install_ps1_dot_sources_region_lib() -> None:
    text = INSTALL_PS1.read_text()
    assert "lib\\apexnodes-region-detect.ps1" in text, "install.ps1 must reference the seam lib path"
    assert "$PSScriptRoot" in text, "install.ps1 must self-locate via $PSScriptRoot to find the lib"
    assert ". $ApexRegionLib" in text, "install.ps1 must dot-source the seam lib"
    assert "Test-Path $ApexRegionLib" in text, "install.ps1 must guard the dot-source on file presence"


def test_install_ps1_calls_lib_functions_with_guard() -> None:
    text = INSTALL_PS1.read_text()
    for fn in (
        "Resolve-ApexRegion",
        "Install-UvFromCos",
        "Test-CnEnabled",
        "Install-RuntimeFromCos",
    ):
        assert fn in text, f"install.ps1 must call {fn}"
        assert f"Get-Command {fn}" in text, f"install.ps1 must guard the {fn} call site with Get-Command"


# --------------------------------------------------------------------------
# 3. The lib ships as a sibling in every materialized copy with CN logic.
# --------------------------------------------------------------------------
def test_pypi_wheel_packaging_ships_the_lib() -> None:
    """The wheel bundles install.{sh,ps1}; it must ship the seam lib too, or a
    wheel-driven install (hermes_cli/dep_ensure) loses the CN downgrade."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    # package-data glob must cover scripts/lib/*.{sh,ps1}.
    assert "scripts/lib/*.sh" in pyproject, "pyproject package-data must ship scripts/lib/*.sh in the wheel"
    assert "scripts/lib/*.ps1" in pyproject, "pyproject package-data must ship scripts/lib/*.ps1 in the wheel"

    workflow = (REPO_ROOT / ".github" / "workflows" / "upload_to_pypi.yml").read_text()
    # The bundling step must copy the libs next to the installers.
    assert "scripts/lib/apexnodes-region-detect.sh" in workflow
    assert "scripts/lib/apexnodes-region-detect.ps1" in workflow


def test_desktop_bundle_stages_the_lib() -> None:
    """The packaged desktop ships install.{sh,ps1} from process.resourcesPath; the
    installer sources lib/ relative to itself, so the lib must be staged to
    resourcesPath/lib/ (build/lib -> lib extraResources)."""
    stage = (REPO_ROOT / "apps" / "desktop" / "scripts" / "stage-install-script.cjs").read_text()
    assert "apexnodes-region-detect.sh" in stage, "desktop staging must stage the .sh seam lib"
    assert "apexnodes-region-detect.ps1" in stage, "desktop staging must stage the .ps1 seam lib"

    pkg = (REPO_ROOT / "apps" / "desktop" / "package.json").read_text()
    # extraResources must map the staged build/lib into the app's resources/lib.
    assert '"from": "build/lib"' in pkg and '"to": "lib"' in pkg, (
        "apps/desktop/package.json extraResources must ship build/lib -> lib so the "
        "bundled installer finds the seam lib as a sibling"
    )
