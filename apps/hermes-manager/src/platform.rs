//! Platform integration planning for PATH and shell setup.
//!
//! This module keeps the first platform-integration slice side-effect free:
//! callers can inspect the planned PATH/profile changes before a later command
//! applies them through OS-specific mechanisms.

use std::fs;
use std::path::{Path, PathBuf};

use crate::{ManagerError, Result};

/// Start marker for Hermes-managed shell profile content.
pub const HERMES_PROFILE_BEGIN: &str = "# >>> Hermes Agent PATH >>>";

/// End marker for Hermes-managed shell profile content.
pub const HERMES_PROFILE_END: &str = "# <<< Hermes Agent PATH <<<";

/// Planned PATH mutation for a Hermes CLI binary directory.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct PathUpdatePlan {
    pub hermes_bin: PathBuf,
    pub path_entries: Vec<PathBuf>,
    pub changed: bool,
    pub next_path: String,
}

/// Planned Windows shortcut pointing at the packaged Hermes desktop app.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ShortcutPlan {
    pub path: PathBuf,
    pub target: PathBuf,
    #[serde(rename = "workingDirectory")]
    pub working_directory: PathBuf,
    #[serde(rename = "iconLocation")]
    pub icon_location: String,
    pub description: String,
}

/// Compute the PATH that would make the Hermes command available.
pub fn plan_path_update(
    install_root: &Path,
    current_path: Option<String>,
    windows: bool,
) -> PathUpdatePlan {
    plan_path_update_with_extra_entries(install_root, &[], current_path, windows)
}

/// Compute a PATH update that also exposes additional managed tool directories.
pub fn plan_path_update_with_extra_entries(
    install_root: &Path,
    extra_entries: &[PathBuf],
    current_path: Option<String>,
    windows: bool,
) -> PathUpdatePlan {
    let hermes_bin = if windows {
        install_root.join("venv").join("Scripts")
    } else {
        install_root.join("venv").join("bin")
    };
    let mut path_entries = vec![hermes_bin.clone()];
    for entry in extra_entries {
        let duplicate = path_entries.iter().any(|candidate| {
            path_eq_for_platform(&candidate.display().to_string(), entry, windows)
        });
        if !duplicate {
            path_entries.push(entry.clone());
        }
    }
    let delimiter = if windows { ';' } else { ':' };
    let current_path = current_path.unwrap_or_default();
    let mut parts = split_path_like(&current_path, delimiter);
    let mut changed = false;

    for entry in path_entries.iter().rev() {
        let already_present = parts
            .iter()
            .any(|part| path_eq_for_platform(part, entry, windows));
        if !already_present {
            parts.insert(0, entry.display().to_string());
            changed = true;
        }
    }

    PathUpdatePlan {
        hermes_bin,
        path_entries,
        changed,
        next_path: parts.join(&delimiter.to_string()),
    }
}

/// Return a shell profile line users can apply on Unix-like platforms.
pub fn shell_profile_hint(plan: &PathUpdatePlan) -> String {
    posix_shell_profile_hint(plan)
}

fn posix_shell_profile_hint(plan: &PathUpdatePlan) -> String {
    let entries = if plan.path_entries.is_empty() {
        plan.hermes_bin.display().to_string()
    } else {
        plan.path_entries
            .iter()
            .map(|entry| entry.display().to_string())
            .collect::<Vec<_>>()
            .join(":")
    };
    format!("export PATH=\"{entries}:$PATH\"")
}

fn fish_shell_profile_hint(plan: &PathUpdatePlan) -> String {
    let entries = if plan.path_entries.is_empty() {
        vec![plan.hermes_bin.clone()]
    } else {
        plan.path_entries.clone()
    };
    entries
        .iter()
        .map(|entry| format!("fish_add_path \"{}\"", entry.display()))
        .collect::<Vec<_>>()
        .join("\n")
}

fn shell_profile_hint_for_profile(profile_path: &Path, plan: &PathUpdatePlan) -> String {
    if profile_path.file_name().and_then(|value| value.to_str()) == Some("config.fish") {
        return fish_shell_profile_hint(plan);
    }
    posix_shell_profile_hint(plan)
}

/// Write or replace the Hermes-managed PATH block in a shell profile file.
pub fn write_shell_profile_update(profile_path: &Path, plan: &PathUpdatePlan) -> Result<()> {
    let existing = match fs::read_to_string(profile_path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(err) => return Err(ManagerError::io(profile_path, err)),
    };
    let block = format!(
        "{HERMES_PROFILE_BEGIN}\n{}\n{HERMES_PROFILE_END}\n",
        shell_profile_hint_for_profile(profile_path, plan)
    );
    let next = replace_managed_block(&existing, &block);
    if let Some(parent) = profile_path.parent() {
        fs::create_dir_all(parent).map_err(|err| ManagerError::io(parent, err))?;
    }
    fs::write(profile_path, next).map_err(|err| ManagerError::io(profile_path, err))
}

/// Write a managed Unix launcher that sanitizes Python environment variables.
pub fn write_unix_launcher(launcher_path: &Path, target_path: &Path) -> Result<bool> {
    if !target_path.is_file() {
        return Ok(false);
    }

    let script = format!(
        "#!/usr/bin/env bash\nunset PYTHONPATH\nunset PYTHONHOME\nexec \"{}\" \"$@\"\n",
        target_path.display()
    );
    let existing = match fs::read_to_string(launcher_path) {
        Ok(text) => Some(text),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
        Err(err) => return Err(ManagerError::io(launcher_path, err)),
    };
    let content_changed = existing.as_deref() != Some(script.as_str());
    if content_changed {
        if let Some(parent) = launcher_path.parent() {
            fs::create_dir_all(parent).map_err(|err| ManagerError::io(parent, err))?;
        }
        fs::write(launcher_path, script).map_err(|err| ManagerError::io(launcher_path, err))?;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let metadata =
            fs::metadata(launcher_path).map_err(|err| ManagerError::io(launcher_path, err))?;
        let mut permissions = metadata.permissions();
        let mode = permissions.mode();
        if mode & 0o111 != 0o111 {
            permissions.set_mode(mode | 0o755);
            fs::set_permissions(launcher_path, permissions)
                .map_err(|err| ManagerError::io(launcher_path, err))?;
            return Ok(true);
        }
    }

    Ok(content_changed)
}

/// Plan Start Menu and Desktop shortcuts for a packaged Hermes desktop executable.
pub fn plan_windows_shortcuts(
    target_exe: &Path,
    programs_dir: &Path,
    desktop_dir: &Path,
    icon_exists: bool,
) -> Vec<ShortcutPlan> {
    let working_directory = target_exe
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .to_path_buf();
    let icon_path = working_directory.join("resources").join("icon.ico");
    let icon_location = if icon_exists {
        format!("{},0", icon_path.display())
    } else {
        format!("{},0", target_exe.display())
    };
    [programs_dir, desktop_dir]
        .into_iter()
        .map(|dir| ShortcutPlan {
            path: dir.join("Hermes.lnk"),
            target: target_exe.to_path_buf(),
            working_directory: working_directory.clone(),
            icon_location: icon_location.clone(),
            description: "Hermes Agent".to_string(),
        })
        .collect()
}

/// Return planned shortcut paths that currently exist.
pub fn existing_shortcut_paths(plans: &[ShortcutPlan]) -> Vec<PathBuf> {
    plans
        .iter()
        .filter(|plan| plan.path.exists())
        .map(|plan| plan.path.clone())
        .collect()
}

/// Create or replace the planned Windows shortcuts.
#[cfg(target_os = "windows")]
pub fn write_windows_shortcuts(plans: &[ShortcutPlan]) -> Result<()> {
    for plan in plans {
        write_one_windows_shortcut(plan)?;
    }
    Ok(())
}

/// Return an actionable error on non-Windows platforms.
#[cfg(not(target_os = "windows"))]
pub fn write_windows_shortcuts(_plans: &[ShortcutPlan]) -> Result<()> {
    Err(ManagerError::InvalidManifest(
        "write-shortcuts is only supported on Windows".to_string(),
    ))
}

/// Remove planned Windows shortcuts that still point at Hermes.
#[cfg(target_os = "windows")]
pub fn remove_windows_shortcuts(plans: &[ShortcutPlan]) -> Result<Vec<PathBuf>> {
    let mut removed = Vec::new();
    for plan in plans {
        if !plan.path.exists() {
            continue;
        }
        if remove_one_windows_shortcut(plan)? {
            removed.push(plan.path.clone());
        }
    }
    Ok(removed)
}

/// Return an actionable error on non-Windows platforms.
#[cfg(not(target_os = "windows"))]
pub fn remove_windows_shortcuts(_plans: &[ShortcutPlan]) -> Result<Vec<PathBuf>> {
    Err(ManagerError::InvalidManifest(
        "remove-shortcuts is only supported on Windows".to_string(),
    ))
}

#[cfg(target_os = "windows")]
fn write_one_windows_shortcut(plan: &ShortcutPlan) -> Result<()> {
    use std::os::windows::process::CommandExt;

    let script = r#"
& {
    param($LinkPath, $TargetPath, $WorkingDirectory, $IconLocation, $Description)
    $ErrorActionPreference = 'Stop'
    $parent = Split-Path -Parent $LinkPath
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($LinkPath)
    $shortcut.TargetPath = $TargetPath
    $shortcut.WorkingDirectory = $WorkingDirectory
    $shortcut.IconLocation = $IconLocation
    $shortcut.Description = $Description
    $shortcut.Save()
}
"#;

    let status = std::process::Command::new("powershell.exe")
        .args([
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ])
        .arg(&plan.path)
        .arg(&plan.target)
        .arg(&plan.working_directory)
        .arg(&plan.icon_location)
        .arg(&plan.description)
        .creation_flags(0x0800_0000)
        .status()
        .map_err(|err| ManagerError::io(&plan.path, err))?;
    if status.success() {
        Ok(())
    } else {
        Err(ManagerError::InvalidManifest(format!(
            "failed to create shortcut {}: exit {}",
            plan.path.display(),
            status
        )))
    }
}

#[cfg(target_os = "windows")]
fn remove_one_windows_shortcut(plan: &ShortcutPlan) -> Result<bool> {
    use std::os::windows::process::CommandExt;

    let script = r#"
& {
    param($LinkPath, $TargetPath)
    $ErrorActionPreference = 'Stop'
    if (-not (Test-Path -LiteralPath $LinkPath)) { exit 2 }
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($LinkPath)
    if (-not [string]::Equals($shortcut.TargetPath, $TargetPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        exit 3
    }
    Remove-Item -LiteralPath $LinkPath -Force
}
"#;

    let status = std::process::Command::new("powershell.exe")
        .args([
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ])
        .arg(&plan.path)
        .arg(&plan.target)
        .creation_flags(0x0800_0000)
        .status()
        .map_err(|err| ManagerError::io(&plan.path, err))?;
    match status.code() {
        Some(0) => Ok(true),
        Some(2 | 3) => Ok(false),
        _ => Err(ManagerError::InvalidManifest(format!(
            "failed to remove shortcut {}: exit {}",
            plan.path.display(),
            status
        ))),
    }
}

/// Read the current user's Windows PATH registry value.
#[cfg(target_os = "windows")]
pub fn read_windows_user_path() -> Result<Option<String>> {
    match read_windows_user_env_var("Path")? {
        Some(value) => Ok(Some(value)),
        None => read_windows_user_env_var("PATH"),
    }
}

/// Read a current-user Windows environment variable.
#[cfg(target_os = "windows")]
pub fn read_windows_user_env_var(name: &str) -> Result<Option<String>> {
    use winreg::enums::HKEY_CURRENT_USER;
    use winreg::RegKey;

    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let environment = match hkcu.open_subkey("Environment") {
        Ok(key) => key,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(ManagerError::io("HKCU\\Environment", err)),
    };
    match environment.get_value::<String, _>(name) {
        Ok(value) => Ok(Some(value)),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(ManagerError::io(format!("HKCU\\Environment\\{name}"), err)),
    }
}

/// Return no registry PATH off Windows.
#[cfg(not(target_os = "windows"))]
pub fn read_windows_user_path() -> Result<Option<String>> {
    Ok(None)
}

/// Return no registry environment variable off Windows.
#[cfg(not(target_os = "windows"))]
pub fn read_windows_user_env_var(_name: &str) -> Result<Option<String>> {
    Ok(None)
}

/// Write the planned PATH value to the current user's Windows environment.
#[cfg(target_os = "windows")]
pub fn write_windows_user_path_update(plan: &PathUpdatePlan) -> Result<bool> {
    if !plan.changed {
        return Ok(false);
    }

    write_windows_user_env_var("Path", &plan.next_path)?;
    broadcast_windows_environment_change();
    Ok(true)
}

/// Write a current-user Windows environment variable.
#[cfg(target_os = "windows")]
pub fn write_windows_user_env_var(name: &str, value: &str) -> Result<()> {
    use winreg::enums::HKEY_CURRENT_USER;
    use winreg::RegKey;

    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let (environment, _) = hkcu
        .create_subkey("Environment")
        .map_err(|err| ManagerError::io("HKCU\\Environment", err))?;
    environment
        .set_value(name, &value)
        .map_err(|err| ManagerError::io(format!("HKCU\\Environment\\{name}"), err))
}

/// Return an actionable error on non-Windows platforms.
#[cfg(not(target_os = "windows"))]
pub fn write_windows_user_path_update(_plan: &PathUpdatePlan) -> Result<bool> {
    Err(ManagerError::InvalidManifest(
        "write-user-path is only supported on Windows".to_string(),
    ))
}

/// Return an actionable error on non-Windows platforms.
#[cfg(not(target_os = "windows"))]
pub fn write_windows_user_env_var(_name: &str, _value: &str) -> Result<()> {
    Err(ManagerError::InvalidManifest(
        "Windows user environment writes are only supported on Windows".to_string(),
    ))
}

#[cfg(target_os = "windows")]
fn broadcast_windows_environment_change() {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        SendMessageTimeoutW, HWND_BROADCAST, SMTO_ABORTIFHUNG, WM_SETTINGCHANGE,
    };

    let environment = "Environment\0".encode_utf16().collect::<Vec<_>>();
    unsafe {
        SendMessageTimeoutW(
            HWND_BROADCAST,
            WM_SETTINGCHANGE,
            0,
            environment.as_ptr() as isize,
            SMTO_ABORTIFHUNG,
            5000,
            std::ptr::null_mut(),
        );
    }
}

fn replace_managed_block(existing: &str, block: &str) -> String {
    if let Some(begin) = existing.find(HERMES_PROFILE_BEGIN) {
        if let Some(end_offset) = existing[begin..].find(HERMES_PROFILE_END) {
            let end = begin + end_offset + HERMES_PROFILE_END.len();
            let mut next = String::new();
            next.push_str(existing[..begin].trim_end_matches(['\r', '\n']));
            if !next.is_empty() {
                next.push('\n');
            }
            next.push_str(block);
            let suffix = existing[end..].trim_start_matches(['\r', '\n']);
            if !suffix.is_empty() {
                next.push_str(suffix);
                if !next.ends_with('\n') {
                    next.push('\n');
                }
            }
            return next;
        }
    }

    let mut next = existing.trim_end_matches(['\r', '\n']).to_string();
    if !next.is_empty() {
        next.push_str("\n\n");
    }
    next.push_str(block);
    next
}

fn split_path_like(value: &str, delimiter: char) -> Vec<String> {
    value
        .split(delimiter)
        .filter(|part| !part.trim().is_empty())
        .map(str::to_string)
        .collect()
}

fn path_eq_for_platform(left: &str, right: &Path, windows: bool) -> bool {
    let right = right.display().to_string();
    if windows {
        left.eq_ignore_ascii_case(&right)
    } else {
        left == right
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn path_plan_prepends_hermes_bin_once() {
        let install_root = PathBuf::from("C:/Users/example/hermes/hermes-agent");
        let hermes_bin = install_root.join("venv").join("Scripts");
        let plan = plan_path_update(
            &install_root,
            Some(format!("{};C:/Windows/System32", hermes_bin.display())),
            true,
        );

        assert_eq!(plan.hermes_bin, hermes_bin);
        assert_eq!(plan.changed, false);
        assert_eq!(
            plan.next_path,
            format!("{};C:/Windows/System32", plan.hermes_bin.display())
        );
    }

    #[test]
    fn path_plan_prepends_missing_hermes_bin() {
        let install_root = PathBuf::from("/home/user/.hermes/hermes-agent");
        let plan = plan_path_update(&install_root, Some("/usr/bin:/bin".to_string()), false);

        assert_eq!(plan.hermes_bin, install_root.join("venv").join("bin"));
        assert_eq!(plan.changed, true);
        assert_eq!(
            plan.next_path,
            format!("{}:/usr/bin:/bin", plan.hermes_bin.display())
        );
    }

    #[test]
    fn path_plan_prepends_unix_managed_tool_bin() {
        let install_root = PathBuf::from("/home/user/.hermes/hermes-agent");
        let managed_tools = PathBuf::from("/home/user/.hermes/bin");
        let plan = plan_path_update_with_extra_entries(
            &install_root,
            &[managed_tools.clone()],
            Some("/usr/bin:/bin".to_string()),
            false,
        );

        assert_eq!(plan.hermes_bin, install_root.join("venv").join("bin"));
        assert_eq!(
            plan.path_entries,
            vec![plan.hermes_bin.clone(), managed_tools]
        );
        assert_eq!(plan.changed, true);
        assert_eq!(
            plan.next_path,
            format!(
                "{}:/home/user/.hermes/bin:/usr/bin:/bin",
                plan.hermes_bin.display()
            )
        );
    }

    #[test]
    fn shell_profile_hint_uses_export_line_for_unix() {
        let plan = PathUpdatePlan {
            hermes_bin: PathBuf::from("/home/user/.hermes/hermes-agent/venv/bin"),
            path_entries: vec![
                PathBuf::from("/home/user/.hermes/hermes-agent/venv/bin"),
                PathBuf::from("/home/user/.hermes/bin"),
            ],
            changed: true,
            next_path: String::new(),
        };

        let hint = shell_profile_hint(&plan);

        assert!(hint.contains(
            "export PATH=\"/home/user/.hermes/hermes-agent/venv/bin:/home/user/.hermes/bin:$PATH\""
        ));
    }

    #[test]
    fn write_shell_profile_update_appends_managed_block_once() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let profile = dir.path().join(".bashrc");
        std::fs::write(&profile, "alias ll='ls -la'\n").unwrap();
        let plan = PathUpdatePlan {
            hermes_bin: PathBuf::from("/home/user/.hermes/hermes-agent/venv/bin"),
            path_entries: vec![PathBuf::from("/home/user/.hermes/hermes-agent/venv/bin")],
            changed: true,
            next_path: String::new(),
        };

        write_shell_profile_update(&profile, &plan).unwrap();
        write_shell_profile_update(&profile, &plan).unwrap();

        let text = std::fs::read_to_string(&profile).unwrap();
        assert_eq!(text.matches(HERMES_PROFILE_BEGIN).count(), 1);
        assert!(text.contains("alias ll='ls -la'"));
        assert!(text.contains("export PATH=\"/home/user/.hermes/hermes-agent/venv/bin:$PATH\""));
    }

    #[test]
    fn write_shell_profile_update_uses_fish_add_path_for_fish_config() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let config_dir = dir.path().join(".config").join("fish");
        let profile = config_dir.join("config.fish");
        std::fs::create_dir_all(&config_dir).unwrap();
        std::fs::write(&profile, "set -gx EDITOR vim\n").unwrap();
        let plan = PathUpdatePlan {
            hermes_bin: PathBuf::from("/home/user/.hermes/hermes-agent/venv/bin"),
            path_entries: vec![
                PathBuf::from("/home/user/.hermes/hermes-agent/venv/bin"),
                PathBuf::from("/home/user/.hermes/bin"),
            ],
            changed: true,
            next_path: String::new(),
        };

        write_shell_profile_update(&profile, &plan).unwrap();
        write_shell_profile_update(&profile, &plan).unwrap();

        let text = std::fs::read_to_string(&profile).unwrap();
        assert_eq!(text.matches(HERMES_PROFILE_BEGIN).count(), 1);
        assert!(text.contains("set -gx EDITOR vim"));
        assert!(text.contains("fish_add_path \"/home/user/.hermes/hermes-agent/venv/bin\""));
        assert!(text.contains("fish_add_path \"/home/user/.hermes/bin\""));
        assert!(!text.contains("export PATH="));
    }

    #[test]
    fn write_shell_profile_update_replaces_existing_managed_block() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let profile = dir.path().join(".zshrc");
        std::fs::write(
            &profile,
            format!("{HERMES_PROFILE_BEGIN}\nold\n{HERMES_PROFILE_END}\n"),
        )
        .unwrap();
        let plan = PathUpdatePlan {
            hermes_bin: PathBuf::from("/new/hermes/bin"),
            path_entries: vec![PathBuf::from("/new/hermes/bin")],
            changed: true,
            next_path: String::new(),
        };

        write_shell_profile_update(&profile, &plan).unwrap();

        let text = std::fs::read_to_string(&profile).unwrap();
        assert!(!text.contains("old"));
        assert!(text.contains("export PATH=\"/new/hermes/bin:$PATH\""));
    }

    #[test]
    fn write_unix_launcher_creates_python_env_cleaning_shim() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let target = dir
            .path()
            .join("hermes-agent")
            .join("venv")
            .join("bin")
            .join("hermes");
        let launcher = dir.path().join("bin").join("hermes");
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::write(&target, "#!/bin/sh\n").unwrap();

        let changed = write_unix_launcher(&launcher, &target).unwrap();
        let unchanged = write_unix_launcher(&launcher, &target).unwrap();

        let text = std::fs::read_to_string(&launcher).unwrap();
        assert!(changed);
        assert!(!unchanged);
        assert!(text.starts_with("#!/usr/bin/env bash\n"));
        assert!(text.contains("unset PYTHONPATH\n"));
        assert!(text.contains("unset PYTHONHOME\n"));
        assert!(text.contains(&format!("exec \"{}\" \"$@\"\n", target.display())));

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&launcher).unwrap().permissions().mode();
            assert_eq!(mode & 0o111, 0o111);
        }
    }

    #[test]
    fn write_unix_launcher_skips_missing_target() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let target = dir
            .path()
            .join("hermes-agent")
            .join("venv")
            .join("bin")
            .join("hermes");
        let launcher = dir.path().join("bin").join("hermes");

        let changed = write_unix_launcher(&launcher, &target).unwrap();

        assert!(!changed);
        assert!(!launcher.exists());
    }

    #[test]
    fn plan_windows_shortcuts_points_to_packaged_exe_and_icon() {
        let target = PathBuf::from("C:/hermes/apps/desktop/release/win-unpacked/Hermes.exe");
        let plans = plan_windows_shortcuts(
            &target,
            Path::new("C:/Users/example/AppData/Roaming/Microsoft/Windows/Start Menu/Programs"),
            Path::new("C:/Users/example/Desktop"),
            true,
        );

        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].path.file_name().unwrap(), "Hermes.lnk");
        assert_eq!(plans[0].target, target);
        assert_eq!(
            plans[0].working_directory,
            PathBuf::from("C:/hermes/apps/desktop/release/win-unpacked")
        );
        assert!(plans[0]
            .icon_location
            .replace('\\', "/")
            .ends_with("resources/icon.ico,0"));
        assert_eq!(plans[0].description, "Hermes Agent");
    }

    #[test]
    fn existing_shortcut_paths_reports_only_present_planned_links() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let target = dir.path().join("Hermes.exe");
        let programs = dir.path().join("Programs");
        let desktop = dir.path().join("Desktop");
        std::fs::create_dir_all(&programs).expect("programs dir should be created");
        std::fs::write(programs.join("Hermes.lnk"), "stub").expect("shortcut should be created");
        let plans = plan_windows_shortcuts(&target, &programs, &desktop, false);

        let existing = existing_shortcut_paths(&plans);

        assert_eq!(existing, vec![programs.join("Hermes.lnk")]);
    }

    #[test]
    #[cfg(not(target_os = "windows"))]
    fn windows_user_path_write_reports_unsupported_off_windows() {
        let plan = PathUpdatePlan {
            hermes_bin: PathBuf::from("/home/user/.hermes/hermes-agent/venv/bin"),
            path_entries: vec![PathBuf::from("/home/user/.hermes/hermes-agent/venv/bin")],
            changed: true,
            next_path: String::new(),
        };

        let err = write_windows_user_path_update(&plan).unwrap_err();

        assert!(err.to_string().contains("only supported on Windows"));
    }
}
