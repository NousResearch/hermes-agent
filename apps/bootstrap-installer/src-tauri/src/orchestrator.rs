//! Rust-side bootstrap orchestration planning.
//!
//! This module starts Phase 4 by keeping low-risk install state probes and
//! stage planning in Rust while the actual stage execution still falls back to
//! `install.ps1` / `install.sh` until individual stages reach parity.

use crate::events::StageInfo;
use anyhow::{anyhow, Context, Result};
use chrono::{SecondsFormat, Utc};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// PATH probe result for one external tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolProbe {
    pub name: String,
    pub path: Option<PathBuf>,
}

/// Read-only install state gathered before script-backed stages run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallStateReport {
    pub hermes_home: PathBuf,
    pub install_root: PathBuf,
    pub bootstrap_marker_exists: bool,
    pub tools: Vec<ToolProbe>,
}

/// Stage execution plan for the current mixed Rust/script bootstrap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedStage {
    pub name: String,
    pub rust_probe: bool,
    pub script_fallback: bool,
}

/// Build a read-only state report without including user config or session data.
pub fn install_state_report(hermes_home: &Path, tools: Vec<ToolProbe>) -> InstallStateReport {
    let install_root = hermes_home.join("hermes-agent");
    InstallStateReport {
        hermes_home: hermes_home.to_path_buf(),
        bootstrap_marker_exists: crate::paths::likely_bootstrap_marker(&install_root).exists(),
        install_root,
        tools,
    }
}

/// Probe the current process environment for install-time helper tools.
pub fn probe_install_state(hermes_home: &Path) -> InstallStateReport {
    let tools = ["uv", "git", "node", "npm", "python"]
        .into_iter()
        .map(probe_tool)
        .collect();
    install_state_report(hermes_home, tools)
}

/// Build the initial mixed execution plan from the script manifest.
pub fn build_stage_plan(stages: &[StageInfo], _include_desktop: bool) -> Vec<PlannedStage> {
    stages
        .iter()
        .map(|stage| PlannedStage {
            name: stage.name.clone(),
            rust_probe: is_probe_owned_stage(&stage.name),
            script_fallback: true,
        })
        .collect()
}

/// Return a compact log line for the current Rust orchestration coverage.
pub fn summarize_plan(report: &InstallStateReport, plan: &[PlannedStage]) -> String {
    let tool_summary = report
        .tools
        .iter()
        .map(|tool| match &tool.path {
            Some(path) => format!("{}={}", tool.name, path.display()),
            None => format!("{}=missing", tool.name),
        })
        .collect::<Vec<_>>()
        .join(", ");
    let probe_count = plan.iter().filter(|stage| stage.rust_probe).count();
    format!(
        "[bootstrap] rust orchestrator: install_root={} marker_exists={} probe_stages={}/{} tools=[{}]",
        report.install_root.display(),
        report.bootstrap_marker_exists,
        probe_count,
        plan.len(),
        tool_summary
    )
}

/// Write the bootstrap-complete marker consumed by the desktop launcher.
pub fn write_bootstrap_marker(
    install_root: &Path,
    pinned_commit: Option<&str>,
    pinned_branch: Option<&str>,
) -> Result<serde_json::Value> {
    if !install_root.is_dir() {
        return Err(anyhow!(
            "install root does not exist: {}",
            install_root.display()
        ));
    }

    let commit = pinned_commit
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .or_else(|| resolve_git_head(install_root))
        .unwrap_or_default();
    let branch = pinned_branch
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("main");

    let marker = serde_json::json!({
        "schemaVersion": 1,
        "pinnedCommit": commit,
        "pinnedBranch": branch,
        "completedAt": Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
    });
    let marker_path = install_root.join(".hermes-bootstrap-complete");
    let text = serde_json::to_string_pretty(&marker)
        .context("serializing bootstrap marker")?
        + "\n";
    std::fs::write(&marker_path, text)
        .with_context(|| format!("writing bootstrap marker {}", marker_path.display()))?;
    Ok(marker)
}

/// Create Hermes home config directories and initial template files.
pub fn configure_templates(hermes_home: &Path, install_root: &Path) -> Result<serde_json::Value> {
    if !install_root.is_dir() {
        return Err(anyhow!(
            "install root does not exist: {}",
            install_root.display()
        ));
    }

    for dir in [
        "cron",
        "sessions",
        "logs",
        "pairing",
        "hooks",
        "image_cache",
        "audio_cache",
        "memories",
        "skills",
    ] {
        fs::create_dir_all(hermes_home.join(dir))
            .with_context(|| format!("creating Hermes home directory {dir}"))?;
    }

    let env_created = ensure_file_from_template(
        &hermes_home.join(".env"),
        &install_root.join(".env.example"),
        Some(""),
    )?;
    let config_created = ensure_file_from_template(
        &hermes_home.join("config.yaml"),
        &install_root.join("cli-config.yaml.example"),
        None,
    )?;
    let soul_created = ensure_soul_file(&hermes_home.join("SOUL.md"))?;
    let skills_sync = sync_bundled_skills(hermes_home, install_root)?;

    Ok(serde_json::json!({
        "hermesHome": hermes_home,
        "envCreated": env_created,
        "configCreated": config_created,
        "soulCreated": soul_created,
        "skillsSync": skills_sync,
    }))
}

fn ensure_file_from_template(dest: &Path, template: &Path, empty_fallback: Option<&str>) -> Result<bool> {
    if dest.exists() {
        return Ok(false);
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    if template.exists() {
        fs::copy(template, dest).with_context(|| {
            format!(
                "copying template {} to {}",
                template.display(),
                dest.display()
            )
        })?;
        return Ok(true);
    }
    if let Some(contents) = empty_fallback {
        fs::write(dest, contents).with_context(|| format!("creating {}", dest.display()))?;
        return Ok(true);
    }
    Ok(false)
}

fn ensure_soul_file(path: &Path) -> Result<bool> {
    if path.exists() {
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::write(path, SOUL_TEMPLATE).with_context(|| format!("creating {}", path.display()))?;
    Ok(true)
}

fn sync_bundled_skills(hermes_home: &Path, install_root: &Path) -> Result<&'static str> {
    let script = install_root.join("tools").join("skills_sync.py");
    let python = if cfg!(target_os = "windows") {
        install_root.join("venv").join("Scripts").join("python.exe")
    } else {
        install_root.join("venv").join("bin").join("python")
    };
    if python.exists() && script.exists() {
        let (env_key, env_value) = skills_sync_env(hermes_home);
        let status = Command::new(&python)
            .arg(&script)
            .env(env_key, env_value)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if matches!(status, Ok(status) if status.success()) {
            return Ok("python");
        }
    }

    let bundled = install_root.join("skills");
    let user_skills = hermes_home.join("skills");
    if bundled.is_dir() && !user_skills_has_non_manifest_entries(&user_skills)? {
        copy_dir_contents(&bundled, &user_skills)?;
        return Ok("copied");
    }
    Ok("skipped")
}

fn skills_sync_env(hermes_home: &Path) -> (&'static str, PathBuf) {
    ("HERMES_HOME", hermes_home.to_path_buf())
}

fn user_skills_has_non_manifest_entries(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    for entry in fs::read_dir(path).with_context(|| format!("reading {}", path.display()))? {
        let entry = entry.with_context(|| format!("reading entry under {}", path.display()))?;
        if entry.file_name() != ".bundled_manifest" {
            return Ok(true);
        }
    }
    Ok(false)
}

fn copy_dir_contents(src: &Path, dest: &Path) -> Result<()> {
    fs::create_dir_all(dest).with_context(|| format!("creating {}", dest.display()))?;
    for entry in fs::read_dir(src).with_context(|| format!("reading {}", src.display()))? {
        let entry = entry.with_context(|| format!("reading entry under {}", src.display()))?;
        let src_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_contents(&src_path, &dest_path)?;
        } else {
            if let Some(parent) = dest_path.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
            fs::copy(&src_path, &dest_path).with_context(|| {
                format!("copying {} to {}", src_path.display(), dest_path.display())
            })?;
        }
    }
    Ok(())
}

fn resolve_git_head(install_root: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["-c", "windows.appendAtomically=false", "rev-parse", "HEAD"])
        .current_dir(install_root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

const SOUL_TEMPLATE: &str = r#"# Hermes Agent Persona

<!--
This file defines the agent's personality and tone.
The agent will embody whatever you write here.
Edit this to customize how Hermes communicates with you.

Examples:
  - "You are a warm, playful assistant who uses kaomoji occasionally."
  - "You are a concise technical expert. No fluff, just facts."
  - "You speak like a friendly coworker who happens to know everything."

This file is loaded fresh each message -- no restart needed.
Delete the contents (or this file) to use the default personality.
-->
"#;

fn probe_tool(name: &str) -> ToolProbe {
    let path_env = std::env::var_os("PATH").unwrap_or_default();
    let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string());
    ToolProbe {
        name: name.to_string(),
        path: find_executable_on_path(name, path_env, &pathext),
    }
}

fn is_probe_owned_stage(name: &str) -> bool {
    matches!(
        name,
        "uv" | "python" | "git" | "node" | "system-packages" | "path"
    )
}

fn find_executable_on_path<P>(name: &str, path_env: P, pathext: &str) -> Option<PathBuf>
where
    P: AsRef<OsStr>,
{
    let candidates = executable_candidates(name, pathext);
    for dir in std::env::split_paths(path_env.as_ref()) {
        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

fn executable_candidates(name: &str, pathext: &str) -> Vec<String> {
    let has_extension = Path::new(name).extension().is_some();
    if has_extension {
        return vec![name.to_string()];
    }

    let mut out = vec![name.to_string()];
    for ext in pathext.split(';') {
        let ext = ext.trim();
        if ext.is_empty() {
            continue;
        }
        out.push(format!("{name}{ext}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::StageInfo;
    use std::path::PathBuf;

    fn stage(name: &str) -> StageInfo {
        StageInfo {
            name: name.to_string(),
            title: format!("Stage {name}"),
            category: "install".to_string(),
            needs_user_input: false,
        }
    }

    #[test]
    fn find_executable_on_path_uses_windows_pathext_candidates() {
        let root = std::env::temp_dir().join(format!(
            "hermes-orchestrator-path-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let exe = root.join("uv.EXE");
        std::fs::write(&exe, b"stub").unwrap();

        let found = find_executable_on_path("uv", &root, ".COM;.EXE;.BAT").unwrap();

        assert_eq!(found, exe);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn build_stage_plan_marks_known_probe_owned_stages_without_skipping_scripts() {
        let stages = vec![stage("uv"), stage("repository"), stage("configure")];
        let plan = build_stage_plan(&stages, false);

        assert_eq!(plan.len(), 3);
        assert_eq!(plan[0].name, "uv");
        assert_eq!(plan[0].rust_probe, true);
        assert_eq!(plan[0].script_fallback, true);
        assert_eq!(plan[1].name, "repository");
        assert_eq!(plan[1].rust_probe, false);
        assert_eq!(plan[2].script_fallback, true);
    }

    #[test]
    fn install_state_report_keeps_user_data_out_of_the_report() {
        let hermes_home = PathBuf::from("C:/Users/example/AppData/Local/hermes");
        let report = install_state_report(&hermes_home, Vec::new());

        assert_eq!(report.hermes_home, hermes_home);
        assert_eq!(report.install_root, hermes_home.join("hermes-agent"));
        assert!(report.tools.is_empty());
    }

    #[test]
    fn write_bootstrap_marker_uses_pin_and_default_branch_without_bom() {
        let root = std::env::temp_dir().join(format!(
            "hermes-marker-test-{}",
            std::process::id()
        ));
        let install_root = root.join("hermes-agent");
        std::fs::create_dir_all(&install_root).unwrap();

        let marker = write_bootstrap_marker(&install_root, Some("abcdef123"), None).unwrap();
        let marker_path = install_root.join(".hermes-bootstrap-complete");
        let bytes = std::fs::read(&marker_path).unwrap();

        assert!(!bytes.starts_with(&[0xef, 0xbb, 0xbf]));
        assert_eq!(marker["schemaVersion"], 1);
        assert_eq!(marker["pinnedCommit"], "abcdef123");
        assert_eq!(marker["pinnedBranch"], "main");
        assert!(marker["completedAt"].as_str().unwrap().ends_with('Z'));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn write_bootstrap_marker_rejects_missing_install_root() {
        let root = std::env::temp_dir().join(format!(
            "hermes-marker-missing-{}",
            std::process::id()
        ));

        let err = write_bootstrap_marker(&root.join("hermes-agent"), Some("abcdef123"), None)
            .unwrap_err();

        assert!(err.to_string().contains("install root does not exist"));
    }

    #[test]
    fn configure_templates_preserves_user_files_and_copies_skill_fallback() {
        let root = std::env::temp_dir().join(format!(
            "hermes-config-test-{}",
            std::process::id()
        ));
        let hermes_home = root.join("home");
        let install_root = root.join("hermes-agent");
        std::fs::create_dir_all(install_root.join("skills").join("demo")).unwrap();
        std::fs::write(install_root.join(".env.example"), "TOKEN=\n").unwrap();
        std::fs::write(install_root.join("cli-config.yaml.example"), "model: test\n").unwrap();
        std::fs::write(install_root.join("skills").join("demo").join("SKILL.md"), "# Demo\n")
            .unwrap();
        std::fs::create_dir_all(&hermes_home).unwrap();
        std::fs::write(hermes_home.join(".env"), "USER=1\n").unwrap();

        let report = configure_templates(&hermes_home, &install_root).unwrap();

        assert_eq!(report["envCreated"], false);
        assert_eq!(std::fs::read_to_string(hermes_home.join(".env")).unwrap(), "USER=1\n");
        assert_eq!(
            std::fs::read_to_string(hermes_home.join("config.yaml")).unwrap(),
            "model: test\n"
        );
        assert!(hermes_home.join("cron").is_dir());
        assert!(hermes_home.join("sessions").is_dir());
        assert!(hermes_home.join("skills").join("demo").join("SKILL.md").exists());
        let soul_bytes = std::fs::read(hermes_home.join("SOUL.md")).unwrap();
        assert!(!soul_bytes.starts_with(&[0xef, 0xbb, 0xbf]));
        assert_eq!(report["skillsSync"], "copied");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn skills_sync_env_sets_hermes_home_for_python_child() {
        let hermes_home = PathBuf::from("C:/Users/example/AppData/Local/hermes");
        let env = skills_sync_env(&hermes_home);

        assert_eq!(env.0, "HERMES_HOME");
        assert_eq!(env.1, hermes_home);
    }
}
