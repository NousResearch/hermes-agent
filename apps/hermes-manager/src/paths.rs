//! Cross-platform path helpers for Hermes-managed resources.

use std::ffi::OsString;
use std::path::PathBuf;

/// Environment variable that overrides the default Hermes home.
pub const HERMES_HOME_ENV: &str = "HERMES_HOME";

/// Directory name used under the operating-system home directory.
pub const HERMES_DIR_NAME: &str = ".hermes";

/// Resolve Hermes home from an explicit override or the process environment.
pub fn hermes_home(explicit: Option<PathBuf>) -> PathBuf {
    hermes_home_from_env(
        explicit,
        std::env::var_os(HERMES_HOME_ENV),
        std::env::var_os("LOCALAPPDATA"),
        os_home_dir(),
    )
}

fn hermes_home_from_env(
    explicit: Option<PathBuf>,
    env_home: Option<OsString>,
    local_app_data: Option<OsString>,
    home: Option<OsString>,
) -> PathBuf {
    if let Some(path) = explicit {
        return path;
    }

    if let Some(path) = env_home {
        if !path.is_empty() {
            return PathBuf::from(path);
        }
    }

    default_hermes_home_from_env(local_app_data, home, cfg!(target_os = "windows"))
}

/// Return the default Hermes home for the current platform.
pub fn default_hermes_home() -> PathBuf {
    default_hermes_home_from_env(
        std::env::var_os("LOCALAPPDATA"),
        os_home_dir(),
        cfg!(target_os = "windows"),
    )
}

fn os_home_dir() -> Option<OsString> {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("USERPROFILE")
            .or_else(|| {
                let drive = std::env::var_os("HOMEDRIVE")?;
                let path = std::env::var_os("HOMEPATH")?;
                let mut home = PathBuf::from(drive);
                home.push(path);
                Some(home.into_os_string())
            })
            .or_else(|| std::env::var_os("HOME"))
    }

    #[cfg(not(target_os = "windows"))]
    {
        std::env::var_os("HOME")
    }
}

fn default_hermes_home_from_env(
    local_app_data: Option<OsString>,
    home: Option<OsString>,
    is_windows: bool,
) -> PathBuf {
    if is_windows {
        if let Some(local_app_data) = local_app_data {
            if !local_app_data.is_empty() {
                return PathBuf::from(local_app_data).join("hermes");
            }
        }

        if let Some(home) = home {
            if !home.is_empty() {
                return PathBuf::from(home)
                    .join("AppData")
                    .join("Local")
                    .join("hermes");
            }
        }

        return PathBuf::from("AppData").join("Local").join("hermes");
    }

    if let Some(home) = home {
        if !home.is_empty() {
            return PathBuf::from(home).join(HERMES_DIR_NAME);
        }
    }

    PathBuf::from(HERMES_DIR_NAME)
}

/// Runtime source checkout directory managed by Hermes.
pub fn agent_root(hermes_home: &std::path::Path) -> PathBuf {
    hermes_home.join("hermes-agent")
}

/// Runtime directories that Hermes installers own and may recreate.
pub fn managed_runtime_roots(hermes_home: &std::path::Path) -> Vec<PathBuf> {
    vec![
        agent_root(hermes_home),
        hermes_home.join("bin"),
        hermes_home.join("node"),
        hermes_home.join("git"),
    ]
}

/// Manager metadata directory.
pub fn manager_state_dir(hermes_home: &std::path::Path) -> PathBuf {
    hermes_home.join("manager")
}

/// Installed-files manifest path.
pub fn installed_manifest_path(hermes_home: &std::path::Path) -> PathBuf {
    manager_state_dir(hermes_home).join("installed-files.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_home_wins() {
        let home = hermes_home(Some(PathBuf::from("D:/tmp/hermes-test")));
        assert_eq!(home, PathBuf::from("D:/tmp/hermes-test"));
    }

    #[test]
    fn agent_root_is_under_hermes_home() {
        let home = PathBuf::from("/tmp/hermes");
        assert_eq!(agent_root(&home), PathBuf::from("/tmp/hermes/hermes-agent"));
    }

    #[test]
    fn managed_runtime_roots_are_under_hermes_home() {
        let home = PathBuf::from("/tmp/hermes");
        assert_eq!(
            managed_runtime_roots(&home),
            vec![
                PathBuf::from("/tmp/hermes/hermes-agent"),
                PathBuf::from("/tmp/hermes/bin"),
                PathBuf::from("/tmp/hermes/node"),
                PathBuf::from("/tmp/hermes/git"),
            ]
        );
    }

    #[test]
    fn installed_manifest_lives_under_manager_state() {
        let home = PathBuf::from("/tmp/hermes");
        assert_eq!(
            installed_manifest_path(&home),
            PathBuf::from("/tmp/hermes/manager/installed-files.json")
        );
    }

    #[test]
    fn explicit_env_home_wins_before_default_home() {
        let home = hermes_home_from_env(
            None,
            Some("D:/env/hermes".into()),
            Some("C:/Users/alice/AppData/Local".into()),
            Some("C:/Users/alice".into()),
        );
        assert_eq!(home, PathBuf::from("D:/env/hermes"));
    }

    #[test]
    fn windows_local_app_data_wins_for_default_home() {
        let home = default_hermes_home_from_env(
            Some("C:/Users/alice/AppData/Local".into()),
            Some("C:/Users/alice".into()),
            true,
        );
        assert_eq!(home, PathBuf::from("C:/Users/alice/AppData/Local/hermes"));
    }

    #[test]
    fn windows_blank_local_app_data_falls_back_under_home_app_data_local() {
        let home =
            default_hermes_home_from_env(Some("".into()), Some("C:/Users/alice".into()), true);
        assert_eq!(home, PathBuf::from("C:/Users/alice/AppData/Local/hermes"));
    }

    #[test]
    fn unix_default_home_uses_dot_hermes() {
        let home = default_hermes_home_from_env(
            Some("/ignored/localappdata".into()),
            Some("/home/alice".into()),
            false,
        );
        assert_eq!(home, PathBuf::from("/home/alice/.hermes"));
    }
}
