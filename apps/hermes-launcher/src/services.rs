//! Service restart hooks (§2.4).
//!
//! After a flip, the updater signals running gateways to drain-then-exit-75.
//! The supervisor (systemd/s6/desktop) restarts the fresh process, which
//! resolves `current` and comes up on the new version.
//!
//! Notification plumbing: when invoked by gateway `/update` (--notify-file),
//! writes the same `.update_exit_code` / `.update_output.txt` files the
//! gateway watcher already polls — byte-compatible, zero gateway changes.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Read the gateway PID from `$HERMES_HOME/gateway.pid`.
fn read_gateway_pid(hermes_home: &Path) -> Option<u32> {
    let pid_path = hermes_home.join("gateway.pid");
    let content = std::fs::read_to_string(&pid_path).ok()?;
    // The pid file may be JSON or a plain integer
    let trimmed = content.trim();
    if let Ok(pid) = trimmed.parse::<u32>() {
        return Some(pid);
    }
    // Try JSON
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(pid) = json.get("pid").and_then(|v| v.as_u64()) {
            return Some(pid as u32);
        }
    }
    None
}

/// Signal the gateway to drain-then-exit-75.
///
/// Sends SIGUSR1 (the existing restart signal — gateway/run.py:20900).
/// If no gateway is running (no pid file), just prints a message.
pub fn restart_gateway(hermes_home: &Path) -> Result<()> {
    match read_gateway_pid(hermes_home) {
        Some(pid) => {
            #[cfg(unix)]
            {
                // Check if the process exists
                if !pid_exists(pid) {
                    println!("  gateway pid {} is not running — no restart needed", pid);
                    return Ok(());
                }
                // Send SIGUSR1 (the existing restart signal)
                let result = unsafe { libc::kill(pid as i32, libc::SIGUSR1) };
                if result != 0 {
                    eprintln!(
                        "  warn: failed to signal gateway pid {} — restart manually",
                        pid
                    );
                } else {
                    println!("  gateway pid {} signaled (SIGUSR1) — drain + restart", pid);
                }
            }
            #[cfg(not(unix))]
            {
                // On Windows, we can't send POSIX signals.
                // The gateway has its own restart mechanism; just print.
                println!(
                    "  gateway pid {} — restart manually (Windows doesn't support SIGUSR1)",
                    pid
                );
            }
        }
        None => {
            println!("  no gateway running — restart your gateway to pick up the new version");
        }
    }
    Ok(())
}

/// Check if a PID is alive.
#[cfg(unix)]
fn pid_exists(pid: u32) -> bool {
    // kill(pid, 0) checks existence without sending a signal
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

/// Write the notification files for gateway `/update` IPC.
///
/// Writes `.update_exit_code` (exit code) and `.update_output.txt` (message)
/// in `$HERMES_HOME`, byte-compatible with the existing gateway watcher.
pub fn write_notify_files(
    hermes_home: &Path,
    exit_code: i32,
    message: &str,
    notify_file: Option<&str>,
) -> Result<()> {
    // If a specific notify file was requested, write there
    if let Some(path) = notify_file {
        std::fs::write(path, format!("{}\n", exit_code))?;
        return Ok(());
    }

    // Otherwise write the standard gateway IPC files
    let exit_code_path = hermes_home.join(".update_exit_code");
    let output_path = hermes_home.join(".update_output.txt");

    std::fs::write(&exit_code_path, format!("{}\n", exit_code))
        .with_context(|| format!("cannot write {}", exit_code_path.display()))?;
    std::fs::write(&output_path, message)
        .with_context(|| format!("cannot write {}", output_path.display()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_gateway_pid_plain_integer() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("gateway.pid"), "12345\n").unwrap();
        let pid = read_gateway_pid(tmp.path());
        assert_eq!(pid, Some(12345));
    }

    #[test]
    fn test_read_gateway_pid_json() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("gateway.pid"),
            r#"{"pid": 99999, "start_time": 123456}"#,
        )
        .unwrap();
        let pid = read_gateway_pid(tmp.path());
        assert_eq!(pid, Some(99999));
    }

    #[test]
    fn test_read_gateway_pid_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let pid = read_gateway_pid(tmp.path());
        assert_eq!(pid, None);
    }

    #[test]
    fn test_read_gateway_pid_invalid() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("gateway.pid"), "not a pid\n").unwrap();
        let pid = read_gateway_pid(tmp.path());
        assert_eq!(pid, None);
    }

    #[test]
    fn test_write_notify_files() {
        let tmp = tempfile::tempdir().unwrap();
        write_notify_files(tmp.path(), 0, "update complete", None).unwrap();
        let exit_code = std::fs::read_to_string(tmp.path().join(".update_exit_code")).unwrap();
        let output = std::fs::read_to_string(tmp.path().join(".update_output.txt")).unwrap();
        assert_eq!(exit_code.trim(), "0");
        assert_eq!(output, "update complete");
    }

    #[test]
    fn test_write_notify_files_with_custom_path() {
        let tmp = tempfile::tempdir().unwrap();
        let notify_path = tmp.path().join("custom-notify");
        write_notify_files(tmp.path(), 0, "done", Some(notify_path.to_str().unwrap())).unwrap();
        let content = std::fs::read_to_string(&notify_path).unwrap();
        assert_eq!(content.trim(), "0");
        // Should NOT have written the standard files
        assert!(!tmp.path().join(".update_exit_code").exists());
    }

    #[test]
    fn test_restart_gateway_no_pid() {
        let tmp = tempfile::tempdir().unwrap();
        // No gateway.pid — should print a message, not error
        let result = restart_gateway(tmp.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_restart_gateway_dead_pid() {
        let tmp = tempfile::tempdir().unwrap();
        // Write a PID that doesn't exist (99999 is very unlikely to be real)
        std::fs::write(tmp.path().join("gateway.pid"), "99999\n").unwrap();
        let result = restart_gateway(tmp.path());
        assert!(result.is_ok()); // should not error, just print a message
    }
}
