//! Smoke tests for the hermes-manager command-line binary.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn manager_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_hermes-manager"))
}

fn run_manager(args: &[&str]) -> String {
    let output = Command::new(manager_binary())
        .args(args)
        .output()
        .expect("manager command should run");
    assert!(
        output.status.success(),
        "manager failed\nstatus: {:?}\nstdout: {}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("stdout should be utf-8")
}

fn create_runtime_dirs(hermes_home: &Path) -> Vec<PathBuf> {
    let paths = hermes_manager::paths::managed_runtime_roots(hermes_home);
    for path in &paths {
        fs::create_dir_all(path).expect("runtime dir should be created");
    }
    paths
}

fn create_runtime_files(hermes_home: &Path) -> Vec<PathBuf> {
    let paths = hermes_manager::paths::managed_runtime_files(hermes_home);
    for path in &paths {
        fs::write(path, "managed").expect("runtime file should be created");
    }
    paths
}

#[test]
fn cli_smoke_manages_runtime_metadata_repair_and_lite_uninstall() {
    let temp = tempfile::tempdir().expect("tempdir should be created");
    let hermes_home = temp.path().join("hermes");
    fs::create_dir_all(&hermes_home).expect("Hermes home should be created");
    let user_config = hermes_home.join("config.yaml");
    fs::write(&user_config, "model: test").expect("user config should be created");

    let runtime_dirs = create_runtime_dirs(&hermes_home);
    let runtime_files = create_runtime_files(&hermes_home);
    let runtime_path_count = runtime_dirs.len() + runtime_files.len();
    let hermes_home_text = hermes_home.display().to_string();

    let install_out = run_manager(&["--hermes-home", &hermes_home_text, "install-metadata"]);
    assert!(install_out.contains("install_metadata=ok"));
    assert!(hermes_home
        .join("manager")
        .join("installed-files.json")
        .is_file());

    let dry_run_out = run_manager(&[
        "--hermes-home",
        &hermes_home_text,
        "--json",
        "uninstall-lite",
        "--dry-run",
    ]);
    let dry_run: serde_json::Value =
        serde_json::from_str(&dry_run_out).expect("dry-run output should be json");
    assert_eq!(dry_run["command"], "uninstall-lite");
    assert_eq!(dry_run["dryRun"], true);
    assert_eq!(
        dry_run["paths"]
            .as_array()
            .expect("paths should be array")
            .len(),
        runtime_path_count
    );

    let uninstall_out = run_manager(&[
        "--hermes-home",
        &hermes_home_text,
        "--json",
        "uninstall-lite",
    ]);
    let uninstall: serde_json::Value =
        serde_json::from_str(&uninstall_out).expect("uninstall output should be json");
    assert_eq!(uninstall["command"], "uninstall-lite");
    for path in &runtime_dirs {
        assert!(!path.exists(), "{} should be removed", path.display());
    }
    for path in &runtime_files {
        assert!(!path.exists(), "{} should be removed", path.display());
    }
    assert!(user_config.exists());

    let repaired_runtime_dirs = create_runtime_dirs(&hermes_home);
    let repaired_runtime_files = create_runtime_files(&hermes_home);
    let repaired_path_count = repaired_runtime_dirs.len() + repaired_runtime_files.len();
    let repair_out = run_manager(&["--hermes-home", &hermes_home_text, "--json", "repair-clean"]);
    let repair: serde_json::Value =
        serde_json::from_str(&repair_out).expect("repair output should be json");
    assert_eq!(repair["command"], "repair-clean");
    assert_eq!(
        repair["paths"]
            .as_array()
            .expect("paths should be array")
            .len(),
        repaired_path_count
    );
    for path in repaired_runtime_dirs {
        assert!(!path.exists(), "{} should be removed", path.display());
    }
    for path in repaired_runtime_files {
        assert!(!path.exists(), "{} should be removed", path.display());
    }
    assert!(user_config.exists());
}
