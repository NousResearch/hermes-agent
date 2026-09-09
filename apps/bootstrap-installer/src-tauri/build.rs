fn main() {
    // -----------------------------------------------------------------
    // Windows application manifest. See north-forge-setup.manifest for
    // rationale — it declares level="asInvoker" so Windows's
    // installer-detection heuristic doesn't refuse to launch us without
    // UAC elevation (bootstrap-north-forge.ps1 only writes sibling
    // -venv / -data folders on the user's own drive).
    //
    // The GitHub-download install-script machinery the upstream Hermes
    // installer had here (BUILD_PIN_COMMIT / BUILD_PIN_BRANCH baked in at
    // compile time, git-dir rerun triggers) is gone: North Forge's
    // bootstrap script is already in the checkout on the drive, so there
    // is nothing to pin or fetch.
    // -----------------------------------------------------------------
    #[cfg(target_os = "windows")]
    let attrs = {
        let manifest = include_str!("north-forge-setup.manifest");
        let win = tauri_build::WindowsAttributes::new().app_manifest(manifest);
        tauri_build::Attributes::new().windows_attributes(win)
    };

    #[cfg(not(target_os = "windows"))]
    let attrs = tauri_build::Attributes::new();

    tauri_build::try_build(attrs).expect("failed to run tauri-build");
}
