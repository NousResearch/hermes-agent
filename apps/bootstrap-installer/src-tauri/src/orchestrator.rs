//! Rust-side bootstrap orchestration planning.
//!
//! This module starts Phase 4 by keeping low-risk install state probes and
//! stage planning in Rust while the actual stage execution still falls back to
//! `install.ps1` / `install.sh` until individual stages reach parity.

use crate::events::{Manifest, StageInfo};
use crate::install_script::ScriptKind;
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
    pub execution: StageExecutionMode,
    pub rust_probe: bool,
    pub script_fallback: bool,
}

/// Native Python virtual environment stage execution plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonVenvStagePlan {
    pub uv: PathBuf,
    pub cwd: PathBuf,
    pub venv: PathBuf,
}

/// Native Python runtime stage execution plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonRuntimeStagePlan {
    pub uv: PathBuf,
}

/// Native Python dependency sync stage execution plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonDependenciesStagePlan {
    pub uv: PathBuf,
    pub cwd: PathBuf,
    pub venv: PathBuf,
    pub python: PathBuf,
    pub lockfile: PathBuf,
}

/// Native Node dependency stage execution plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeDependenciesStagePlan {
    pub npm: PathBuf,
    pub npx: Option<PathBuf>,
    pub cwd: PathBuf,
    pub browser_tools: bool,
    pub tui_dir: Option<PathBuf>,
}

/// Native desktop build stage execution plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DesktopBuildStagePlan {
    pub npm: PathBuf,
    pub cwd: PathBuf,
    pub desktop_dir: PathBuf,
}

/// Native Windows Node runtime installation plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowsNodeRuntimeStagePlan {
    pub version_major: u32,
    pub archive_name: String,
    pub download_url: String,
    pub archive_path: PathBuf,
    pub install_dir: PathBuf,
    pub node_exe: PathBuf,
}

/// Native Windows uv runtime installation plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowsUvRuntimeStagePlan {
    pub archive_name: String,
    pub download_url: String,
    pub archive_path: PathBuf,
    pub install_dir: PathBuf,
    pub uv_exe: PathBuf,
}

/// Native Windows Git runtime installation plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowsGitRuntimeStagePlan {
    pub tag: &'static str,
    pub version: &'static str,
    pub archive_name: String,
    pub download_url: String,
    pub archive_path: PathBuf,
    pub install_dir: PathBuf,
    pub git_exe: PathBuf,
    pub bash_exe: PathBuf,
    pub is_zip: bool,
}

/// Messaging-platform SDK requirement derived from user configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlatformSdkRequirement {
    pub env_var: &'static str,
    pub import_name: &'static str,
    pub pip_spec: &'static str,
}

/// Native platform SDK verification stage execution plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformSdkStagePlan {
    pub python: PathBuf,
    pub requirements: Vec<PlatformSdkRequirement>,
}

/// How a bootstrap stage is currently handled by the Rust orchestrator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageExecutionMode {
    Native,
    NativeWithScriptFallback,
    ProbeThenScript,
    Script,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BootstrapArchiveSourceKind {
    Bundled,
    Cache,
}

impl BootstrapArchiveSourceKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Bundled => "bundled",
            Self::Cache => "cache",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedBootstrapArchive {
    path: PathBuf,
    cache_path: PathBuf,
    kind: BootstrapArchiveSourceKind,
}

/// Build the bootstrap stage manifest without invoking the platform script.
pub fn native_bootstrap_manifest(kind: ScriptKind, include_desktop: bool) -> Manifest {
    let stages = match kind {
        ScriptKind::Ps1 => windows_manifest_stages(include_desktop),
        ScriptKind::Sh => unix_manifest_stages(include_desktop),
    };
    Manifest {
        stages,
        protocol_version: Some(1),
    }
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
        .map(|stage| {
            let execution = stage_execution_mode(&stage.name);
            PlannedStage {
                name: stage.name.clone(),
                execution,
                rust_probe: execution == StageExecutionMode::ProbeThenScript,
                script_fallback: matches!(
                    execution,
                    StageExecutionMode::NativeWithScriptFallback
                        | StageExecutionMode::ProbeThenScript
                        | StageExecutionMode::Script
                ),
            }
        })
        .collect()
}

/// Return a Rust-side skip result for stages that must be handled by UI.
pub fn interactive_stage_skip_result(stage: &StageInfo) -> Option<crate::events::StageResultPayload> {
    if !stage.needs_user_input {
        return None;
    }
    Some(crate::events::StageResultPayload {
        stage: stage.name.clone(),
        ok: true,
        skipped: true,
        reason: Some("requires user input; handled by post-install UI".to_string()),
        data: None,
    })
}

/// Return a Rust-side skip result for tool stages already satisfied locally.
#[cfg(test)]
pub fn satisfied_tool_stage_skip_result<P>(
    stage: &StageInfo,
    hermes_home: &Path,
    path_env: P,
    pathext: &str,
) -> Option<crate::events::StageResultPayload>
where
    P: AsRef<OsStr>,
{
    satisfied_tool_stage_skip_result_with_node_probe(
        stage,
        hermes_home,
        path_env,
        pathext,
        |_| false,
    )
}

fn satisfied_tool_stage_skip_result_with_node_probe<P, F>(
    stage: &StageInfo,
    hermes_home: &Path,
    path_env: P,
    pathext: &str,
    node_version_ok: F,
) -> Option<crate::events::StageResultPayload>
where
    P: AsRef<OsStr>,
    F: Fn(&Path) -> bool,
{
    let available = match stage.name.as_str() {
        name if name.eq_ignore_ascii_case("uv") => {
            managed_tool_path(hermes_home, "uv").is_file()
                || find_executable_on_path("uv", path_env.as_ref(), pathext).is_some()
        }
        name if name.eq_ignore_ascii_case("git") => {
            find_executable_on_path("git", path_env.as_ref(), pathext).is_some()
        }
        name if name.eq_ignore_ascii_case("node") => {
            let node = find_node_executable(hermes_home, path_env.as_ref(), pathext);
            let npm = find_npm_executable(hermes_home, path_env.as_ref(), pathext);
            matches!(node, Some(path) if npm.is_some() && node_version_ok(&path))
        }
        name if name.eq_ignore_ascii_case("system-packages") => {
            find_executable_on_path("rg", path_env.as_ref(), pathext).is_some()
                && find_executable_on_path("ffmpeg", path_env.as_ref(), pathext).is_some()
        }
        _ => false,
    };
    if !available {
        return None;
    }
    Some(crate::events::StageResultPayload {
        stage: stage.name.clone(),
        ok: true,
        skipped: true,
        reason: Some("required tool already available".to_string()),
        data: None,
    })
}

/// Return a Rust-side skip result for tool stages satisfied in this process.
pub fn satisfied_tool_stage_skip_result_from_env(
    stage: &StageInfo,
    hermes_home: &Path,
) -> Option<crate::events::StageResultPayload> {
    let path_env = std::env::var_os("PATH").unwrap_or_default();
    let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string());
    satisfied_tool_stage_skip_result_with_node_probe(
        stage,
        hermes_home,
        path_env,
        &pathext,
        node_version_satisfies_build,
    )
}

/// Return a Rust-side skip result when the required Python runtime exists.
pub fn python_stage_skip_result(
    stage: &StageInfo,
    hermes_home: &Path,
) -> Option<crate::events::StageResultPayload> {
    python_stage_skip_result_with_probe(stage, || python_runtime_available(hermes_home))
}

fn python_stage_skip_result_with_probe<F>(
    stage: &StageInfo,
    python_runtime_available: F,
) -> Option<crate::events::StageResultPayload>
where
    F: FnOnce() -> bool,
{
    if !stage.name.eq_ignore_ascii_case("python") {
        return None;
    }
    if !python_runtime_available() {
        return None;
    }
    Some(crate::events::StageResultPayload {
        stage: stage.name.clone(),
        ok: true,
        skipped: true,
        reason: Some("required Python runtime already available".to_string()),
        data: None,
    })
}

fn python_runtime_available(hermes_home: &Path) -> bool {
    let path_env = std::env::var_os("PATH").unwrap_or_default();
    let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string());
    let Ok(uv) = uv_tool_path(hermes_home, path_env, &pathext) else {
        return false;
    };
    Command::new(uv)
        .args(["python", "find", "3.11"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Return a Rust-side skip result for node-deps when npm is absent.
pub fn node_deps_skip_result<P>(
    stage: &StageInfo,
    hermes_home: &Path,
    path_env: P,
    pathext: &str,
) -> Option<crate::events::StageResultPayload>
where
    P: AsRef<OsStr>,
{
    if !stage.name.eq_ignore_ascii_case("node-deps") {
        return None;
    }
    if find_npm_executable(hermes_home, path_env, pathext).is_some() {
        return None;
    }
    Some(crate::events::StageResultPayload {
        stage: stage.name.clone(),
        ok: true,
        skipped: true,
        reason: Some("npm not available".to_string()),
        data: None,
    })
}

/// Return a Rust-side skip result for node-deps in this process.
pub fn node_deps_skip_result_from_env(
    stage: &StageInfo,
    hermes_home: &Path,
) -> Option<crate::events::StageResultPayload> {
    let path_env = std::env::var_os("PATH").unwrap_or_default();
    let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string());
    node_deps_skip_result(stage, hermes_home, path_env, &pathext)
}

/// Return a Rust-side skip result for desktop builds without the desktop package.
pub fn desktop_stage_skip_result(
    stage: &StageInfo,
    install_root: &Path,
) -> Option<crate::events::StageResultPayload> {
    if !stage.name.eq_ignore_ascii_case("desktop") {
        return None;
    }
    if install_root
        .join("apps")
        .join("desktop")
        .join("package.json")
        .is_file()
    {
        return None;
    }
    Some(crate::events::StageResultPayload {
        stage: stage.name.clone(),
        ok: true,
        skipped: true,
        reason: Some("apps/desktop not present".to_string()),
        data: None,
    })
}

/// Return a Rust-side skip result for platform SDK verification with no tokens.
pub fn platform_sdks_skip_result(
    stage: &StageInfo,
    hermes_home: &Path,
) -> Option<crate::events::StageResultPayload> {
    if !stage.name.eq_ignore_ascii_case("platform-sdks") {
        return None;
    }
    if platform_env_has_configured_tokens(&hermes_home.join(".env")) {
        return None;
    }
    Some(crate::events::StageResultPayload {
        stage: stage.name.clone(),
        ok: true,
        skipped: true,
        reason: Some("no messaging platform tokens configured".to_string()),
        data: None,
    })
}

/// Build the native platform SDK verification stage plan.
pub fn platform_sdk_stage_plan(
    hermes_home: &Path,
    install_root: &Path,
) -> Result<PlatformSdkStagePlan> {
    let env_path = hermes_home.join(".env");
    let env_text = fs::read_to_string(&env_path)
        .with_context(|| format!("reading {}", env_path.display()))?;
    let requirements = platform_sdk_requirements_from_env(&env_text);
    if requirements.is_empty() {
        return Err(anyhow!("no messaging platform tokens configured"));
    }
    let python = venv_python_path(&install_root.join("venv"));
    if !python.is_file() {
        return Err(anyhow!("venv Python not found at {}", python.display()));
    }
    Ok(PlatformSdkStagePlan {
        python,
        requirements,
    })
}

/// Verify and install configured messaging platform SDKs natively.
pub fn install_platform_sdks_stage(
    hermes_home: &Path,
    install_root: &Path,
) -> Result<serde_json::Value> {
    let plan = platform_sdk_stage_plan(hermes_home, install_root)?;
    let missing = plan
        .requirements
        .iter()
        .copied()
        .filter(|sdk| !python_import_available(&plan.python, sdk.import_name))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(serde_json::json!({
            "python": plan.python,
            "checked": plan.requirements.len(),
            "installed": [],
        }));
    }
    ensure_pip_available(&plan.python)?;
    for sdk in &missing {
        install_pip_spec(&plan.python, sdk.pip_spec)?;
    }
    Ok(serde_json::json!({
        "python": plan.python,
        "checked": plan.requirements.len(),
        "installed": missing.iter().map(|sdk| sdk.pip_spec).collect::<Vec<_>>(),
    }))
}

fn platform_env_has_configured_tokens(env_path: &Path) -> bool {
    let Ok(text) = fs::read_to_string(env_path) else {
        return false;
    };
    !platform_sdk_requirements_from_env(&text).is_empty()
}

fn platform_sdk_requirements_from_env(text: &str) -> Vec<PlatformSdkRequirement> {
    const SDK_MAP: [PlatformSdkRequirement; 5] = [
        PlatformSdkRequirement {
            env_var: "TELEGRAM_BOT_TOKEN",
            import_name: "telegram",
            pip_spec: "python-telegram-bot[webhooks]>=22.6,<23",
        },
        PlatformSdkRequirement {
            env_var: "DISCORD_BOT_TOKEN",
            import_name: "discord",
            pip_spec: "discord.py[voice]>=2.7.1,<3",
        },
        PlatformSdkRequirement {
            env_var: "SLACK_BOT_TOKEN",
            import_name: "slack_sdk",
            pip_spec: "slack-sdk>=3.27.0,<4",
        },
        PlatformSdkRequirement {
            env_var: "SLACK_APP_TOKEN",
            import_name: "slack_bolt",
            pip_spec: "slack-bolt>=1.18.0,<2",
        },
        PlatformSdkRequirement {
            env_var: "WHATSAPP_ENABLED",
            import_name: "qrcode",
            pip_spec: "qrcode>=7.0,<8",
        },
    ];
    SDK_MAP
        .into_iter()
        .filter(|sdk| env_has_configured_platform_value(text, sdk.env_var))
        .collect()
}

fn env_has_configured_platform_value(text: &str, env_var: &str) -> bool {
    text.lines().any(|line| {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            return false;
        }
        let Some((key, value)) = trimmed.split_once('=') else {
            return false;
        };
        if key != env_var {
            return false;
        }
        let value = value.trim();
        if value.is_empty() || value.eq_ignore_ascii_case("your-token-here") {
            return false;
        }
        if env_var == "WHATSAPP_ENABLED" {
            return value.eq_ignore_ascii_case("true");
        }
        true
    })
}

fn python_import_available(python: &Path, import_name: &str) -> bool {
    Command::new(python)
        .args(["-c", &format!("import {import_name}")])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn ensure_pip_available(python: &Path) -> Result<()> {
    let has_pip = Command::new(python)
        .args(["-m", "pip", "--version"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    if has_pip {
        return Ok(());
    }
    let status = Command::new(python)
        .args(["-m", "ensurepip", "--upgrade"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("running ensurepip with {}", python.display()))?;
    if status.success() {
        Ok(())
    } else {
        Err(anyhow!("ensurepip failed with exit {:?}", status.code()))
    }
}

fn install_pip_spec(python: &Path, spec: &str) -> Result<()> {
    let status = Command::new(python)
        .args(["-m", "pip", "install", spec])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("installing {spec} with {}", python.display()))?;
    if status.success() {
        Ok(())
    } else {
        Err(anyhow!("pip install {spec} failed with exit {:?}", status.code()))
    }
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
    let native_count = plan
        .iter()
        .filter(|stage| {
            matches!(
                stage.execution,
                StageExecutionMode::Native | StageExecutionMode::NativeWithScriptFallback
            )
        })
        .count();
    let probe_count = plan
        .iter()
        .filter(|stage| stage.execution == StageExecutionMode::ProbeThenScript)
        .count();
    let script_count = plan
        .iter()
        .filter(|stage| stage.execution == StageExecutionMode::Script)
        .count();
    format!(
        concat!(
            "[bootstrap] rust orchestrator: install_root={} marker_exists={} ",
            "native_stages={} probe_stages={} script_stages={} total_stages={} tools=[{}]"
        ),
        report.install_root.display(),
        report.bootstrap_marker_exists,
        native_count,
        probe_count,
        script_count,
        plan.len(),
        tool_summary
    )
}

fn windows_manifest_stages(include_desktop: bool) -> Vec<StageInfo> {
    let mut stages = vec![
        stage_info("uv", "Installing uv package manager", "prereqs", false),
        stage_info("python", "Verifying Python 3.11", "prereqs", false),
        stage_info("git", "Installing Git", "prereqs", false),
        stage_info("node", "Detecting Node.js", "prereqs", false),
        stage_info(
            "system-packages",
            "Installing ripgrep and ffmpeg",
            "prereqs",
            false,
        ),
        stage_info("repository", "Cloning Hermes repository", "install", false),
        stage_info(
            "venv",
            "Creating Python virtual environment",
            "install",
            false,
        ),
        stage_info(
            "dependencies",
            "Installing Python dependencies",
            "install",
            false,
        ),
        stage_info("node-deps", "Installing Node.js dependencies", "install", false),
    ];
    if include_desktop {
        stages.push(stage_info("desktop", "Building desktop app", "install", false));
    }
    stages.extend([
        stage_info("path", "Adding Hermes to PATH", "finalize", false),
        stage_info(
            "config-templates",
            "Writing configuration templates",
            "finalize",
            false,
        ),
        stage_info(
            "platform-sdks",
            "Installing messaging platform SDKs",
            "finalize",
            false,
        ),
        stage_info(
            "bootstrap-marker",
            "Marking install complete",
            "finalize",
            false,
        ),
        stage_info(
            "configure",
            "Configuring API keys and models",
            "post-install",
            true,
        ),
        stage_info(
            "gateway",
            "Starting messaging gateway",
            "post-install",
            true,
        ),
    ]);
    stages
}

fn unix_manifest_stages(include_desktop: bool) -> Vec<StageInfo> {
    let mut stages = vec![
        stage_info("prerequisites", "System prerequisites", "runtime", false),
        stage_info("repository", "Download Hermes Agent", "runtime", false),
        stage_info(
            "venv",
            "Create Python virtual environment",
            "runtime",
            false,
        ),
        stage_info(
            "python-deps",
            "Install Python dependencies",
            "runtime",
            false,
        ),
        stage_info(
            "node-deps",
            "Install browser-tool dependencies",
            "runtime",
            false,
        ),
        stage_info("path", "Install hermes command", "runtime", false),
        stage_info("config", "Prepare config and skills", "configuration", false),
        stage_info(
            "setup",
            "Configure API keys and settings",
            "configuration",
            true,
        ),
        stage_info(
            "gateway",
            "Configure gateway service",
            "configuration",
            true,
        ),
    ];
    if include_desktop {
        stages.push(stage_info("desktop", "Build desktop app", "runtime", false));
    }
    stages.push(stage_info("complete", "Finish install", "runtime", false));
    stages
}

fn stage_info(name: &str, title: &str, category: &str, needs_user_input: bool) -> StageInfo {
    StageInfo {
        name: name.to_string(),
        title: title.to_string(),
        category: category.to_string(),
        needs_user_input,
    }
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

/// Stamp the install method used by status and update recommendations.
pub fn write_install_method_stamp(hermes_home: &Path) -> Result<serde_json::Value> {
    let stamp_path = hermes_home.join(".install_method");
    if let Some(parent) = stamp_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::write(&stamp_path, "git\n")
        .with_context(|| format!("writing install method stamp {}", stamp_path.display()))?;
    Ok(serde_json::json!({
        "installMethod": "git",
        "stampPath": stamp_path.display().to_string(),
    }))
}

/// Build the native Python runtime stage plan.
pub fn python_runtime_stage_plan<P>(
    hermes_home: &Path,
    path_env: P,
    pathext: &str,
) -> Result<PythonRuntimeStagePlan>
where
    P: AsRef<OsStr>,
{
    Ok(PythonRuntimeStagePlan {
        uv: uv_tool_path(hermes_home, path_env, pathext)?,
    })
}

/// Install the required Python runtime natively through uv.
pub fn install_python_runtime_stage(hermes_home: &Path) -> Result<serde_json::Value> {
    let path_env = std::env::var_os("PATH").unwrap_or_default();
    let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string());
    let plan = python_runtime_stage_plan(hermes_home, path_env, &pathext)?;
    let status = Command::new(&plan.uv)
        .args(["python", "install", "3.11"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("running {}", plan.uv.display()))?;
    if !status.success() {
        return Err(anyhow!(
            "uv python install failed with exit {:?}",
            status.code()
        ));
    }
    let output = Command::new(&plan.uv)
        .args(["python", "find", "3.11"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .with_context(|| format!("locating Python with {}", plan.uv.display()))?;
    if !output.status.success() {
        return Err(anyhow!("uv python find failed after install"));
    }
    let python = String::from_utf8(output.stdout)
        .context("decoding uv python find output")?
        .trim()
        .to_string();
    if python.is_empty() {
        return Err(anyhow!("uv python find returned an empty path after install"));
    }
    Ok(serde_json::json!({
        "uv": plan.uv,
        "python": python,
    }))
}

/// Build a Windows Node runtime plan from the Node.js latest-v22.x index.
pub fn windows_node_runtime_stage_plan_from_index(
    hermes_home: &Path,
    arch: &str,
    index_html: &str,
) -> Result<WindowsNodeRuntimeStagePlan> {
    let version_major = 22;
    let archive_name = latest_windows_node_archive_name(index_html, version_major, arch)
        .ok_or_else(|| anyhow!("Node.js v{version_major} Windows {arch} archive not found"))?;
    windows_node_runtime_stage_plan_from_archive_name(hermes_home, archive_name)
}

fn windows_node_runtime_stage_plan_from_archive_name(
    hermes_home: &Path,
    archive_name: String,
) -> Result<WindowsNodeRuntimeStagePlan> {
    let version_major = node_archive_version_tuple(&archive_name).0;
    if version_major == 0 {
        return Err(anyhow!("invalid Node.js archive name: {archive_name}"));
    }
    let download_url =
        format!("https://nodejs.org/dist/latest-v{version_major}.x/{archive_name}");
    let archive_path = bootstrap_archive_cache_path(hermes_home, &archive_name);
    let install_dir = hermes_home.join("node");
    let node_exe = install_dir.join("node.exe");
    Ok(WindowsNodeRuntimeStagePlan {
        version_major,
        archive_name,
        download_url,
        archive_path,
        install_dir,
        node_exe,
    })
}

/// Install the Windows Node.js runtime from the official portable ZIP archive.
pub async fn install_windows_node_runtime_stage(
    hermes_home: &Path,
    bundled_tools_dir: Option<&Path>,
) -> Result<serde_json::Value> {
    if !cfg!(target_os = "windows") {
        return Err(anyhow!("native Node runtime stage is only available on Windows"));
    }
    let arch = windows_node_arch_slug();
    if arch != "x64" && arch != "arm64" && arch != "x86" {
        return Err(anyhow!("unsupported Windows architecture for Node.js: {arch}"));
    }
    let version_major = 22;
    let plan = if let Some(archive_name) =
        latest_bundled_windows_node_archive_name(bundled_tools_dir, version_major, &arch)
    {
        windows_node_runtime_stage_plan_from_archive_name(hermes_home, archive_name)?
    } else {
        let index_url = format!("https://nodejs.org/dist/latest-v{version_major}.x/");
        let index_html = reqwest::Client::new()
            .get(&index_url)
            .header("User-Agent", "Hermes-Setup")
            .send()
            .await
            .with_context(|| format!("GET {index_url}"))?
            .text()
            .await
            .with_context(|| format!("reading body of {index_url}"))?;
        windows_node_runtime_stage_plan_from_index(hermes_home, &arch, &index_html)?
    };
    let archive_source =
        resolve_bootstrap_archive_source(hermes_home, bundled_tools_dir, &plan.archive_name);
    if archive_source.kind == BootstrapArchiveSourceKind::Cache {
        crate::artifact::download_to_cache(
            crate::artifact::DownloadSpec {
                url: plan.download_url.clone(),
                user_agent: "Hermes-Setup",
                expected_sha256: None,
            },
            &archive_source.path,
        )
        .await
        .with_context(|| format!("downloading {}", plan.archive_name))?;
    }
    install_windows_node_archive(&archive_source.path, &plan.install_dir)?;
    if !node_version_satisfies_build(&plan.node_exe) {
        return Err(anyhow!(
            "installed Node.js does not satisfy desktop build requirements"
        ));
    }
    prepend_process_path(&plan.install_dir);
    persist_windows_path_entry(&plan.install_dir)?;
    Ok(serde_json::json!({
        "node": plan.node_exe,
        "archive": plan.archive_name,
        "archiveSource": archive_source.kind.as_str(),
        "installDir": plan.install_dir,
    }))
}

/// Build a Windows uv runtime plan from the GitHub release asset matrix.
pub fn windows_uv_runtime_stage_plan(
    hermes_home: &Path,
    arch: &str,
) -> Result<WindowsUvRuntimeStagePlan> {
    let archive_name = windows_uv_archive_name(arch)
        .ok_or_else(|| anyhow!("unsupported Windows architecture for uv: {arch}"))?
        .to_string();
    let download_url =
        format!("https://github.com/astral-sh/uv/releases/latest/download/{archive_name}");
    let archive_path = hermes_home
        .join("bootstrap-cache")
        .join(&archive_name);
    let install_dir = hermes_home.join("bin");
    let uv_exe = install_dir.join("uv.exe");
    Ok(WindowsUvRuntimeStagePlan {
        archive_name,
        download_url,
        archive_path,
        install_dir,
        uv_exe,
    })
}

/// Install uv natively on Windows from a bundled or official GitHub release ZIP.
pub async fn install_windows_uv_runtime_stage(
    hermes_home: &Path,
    bundled_tools_dir: Option<&Path>,
) -> Result<serde_json::Value> {
    if !cfg!(target_os = "windows") {
        return Err(anyhow!("native uv stage is only available on Windows"));
    }
    let arch = windows_node_arch_slug();
    let plan = windows_uv_runtime_stage_plan(hermes_home, &arch)?;
    let archive_source =
        resolve_bootstrap_archive_source(hermes_home, bundled_tools_dir, &plan.archive_name);
    if archive_source.kind == BootstrapArchiveSourceKind::Cache {
        crate::artifact::download_to_cache(
            crate::artifact::DownloadSpec {
                url: plan.download_url.clone(),
                user_agent: "Hermes-Setup",
                expected_sha256: None,
            },
            &archive_source.path,
        )
        .await
        .with_context(|| format!("downloading {}", plan.archive_name))?;
    }
    install_windows_uv_archive(&archive_source.path, &plan.install_dir)?;
    let status = Command::new(&plan.uv_exe)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("checking {}", plan.uv_exe.display()))?;
    if !status.success() {
        return Err(anyhow!("installed uv failed version check"));
    }
    Ok(serde_json::json!({
        "uv": plan.uv_exe,
        "archive": plan.archive_name,
        "archiveSource": archive_source.kind.as_str(),
    }))
}

/// Build a Windows Git runtime plan matching install.ps1's pinned release.
pub fn windows_git_runtime_stage_plan(
    hermes_home: &Path,
    arch: &str,
) -> Result<WindowsGitRuntimeStagePlan> {
    let tag = "v2.54.0.windows.1";
    let version = "2.54.0";
    let (archive_name, is_zip, bash_relative) = match arch {
        "arm64" => (
            format!("PortableGit-{version}-arm64.7z.exe"),
            false,
            PathBuf::from("bin").join("bash.exe"),
        ),
        "x64" => (
            format!("PortableGit-{version}-64-bit.7z.exe"),
            false,
            PathBuf::from("bin").join("bash.exe"),
        ),
        "x86" => (
            format!("MinGit-{version}-32-bit.zip"),
            true,
            PathBuf::from("usr").join("bin").join("bash.exe"),
        ),
        other => return Err(anyhow!("unsupported Windows architecture for Git: {other}")),
    };
    let download_url =
        format!("https://github.com/git-for-windows/git/releases/download/{tag}/{archive_name}");
    let archive_path = hermes_home
        .join("bootstrap-cache")
        .join(&archive_name);
    let install_dir = hermes_home.join("git");
    let git_exe = install_dir.join("cmd").join("git.exe");
    let bash_exe = install_dir.join(bash_relative);
    Ok(WindowsGitRuntimeStagePlan {
        tag,
        version,
        archive_name,
        download_url,
        archive_path,
        install_dir,
        git_exe,
        bash_exe,
        is_zip,
    })
}

/// Install Git for Windows natively before falling back to PowerShell.
pub async fn install_windows_git_runtime_stage(
    hermes_home: &Path,
    bundled_tools_dir: Option<&Path>,
) -> Result<serde_json::Value> {
    if !cfg!(target_os = "windows") {
        return Err(anyhow!("native Git stage is only available on Windows"));
    }
    let arch = windows_node_arch_slug();
    let plan = windows_git_runtime_stage_plan(hermes_home, &arch)?;
    let archive_source =
        resolve_bootstrap_archive_source(hermes_home, bundled_tools_dir, &plan.archive_name);
    if archive_source.kind == BootstrapArchiveSourceKind::Cache {
        crate::artifact::download_to_cache(
            crate::artifact::DownloadSpec {
                url: plan.download_url.clone(),
                user_agent: "Hermes-Setup",
                expected_sha256: None,
            },
            &archive_source.path,
        )
        .await
        .with_context(|| format!("downloading {}", plan.archive_name))?;
    }
    let install_plan = WindowsGitRuntimeStagePlan {
        archive_path: archive_source.path.clone(),
        ..plan
    };
    install_windows_git_archive(&install_plan)?;
    if !install_plan.git_exe.is_file() {
        return Err(anyhow!(
            "Git extraction did not produce {}",
            install_plan.git_exe.display()
        ));
    }
    let path_entries = [
        install_plan.install_dir.join("cmd"),
        install_plan.install_dir.join("bin"),
        install_plan.install_dir.join("usr").join("bin"),
    ];
    prepend_process_paths(&path_entries);
    persist_windows_path_entries(&path_entries)?;
    let bash = find_windows_git_bash(&install_plan.install_dir);
    if let Some(bash) = &bash {
        persist_windows_env_var("HERMES_GIT_BASH_PATH", &bash.display().to_string())?;
        std::env::set_var("HERMES_GIT_BASH_PATH", bash);
    }
    Ok(serde_json::json!({
        "git": install_plan.git_exe,
        "bash": bash,
        "archive": install_plan.archive_name,
        "archiveSource": archive_source.kind.as_str(),
    }))
}

/// Build the native Python virtual environment stage plan.
pub fn python_venv_stage_plan<P>(
    install_root: &Path,
    hermes_home: &Path,
    path_env: P,
    pathext: &str,
) -> Result<PythonVenvStagePlan>
where
    P: AsRef<OsStr>,
{
    if !install_root.is_dir() {
        return Err(anyhow!(
            "install root does not exist: {}",
            install_root.display()
        ));
    }
    let uv = uv_tool_path(hermes_home, path_env, pathext)?;
    let venv = install_root.join("venv");
    Ok(PythonVenvStagePlan {
        uv,
        cwd: install_root.to_path_buf(),
        venv,
    })
}

/// Create the Python virtual environment natively through uv.
pub fn create_python_venv_stage(
    install_root: &Path,
    hermes_home: &Path,
) -> Result<serde_json::Value> {
    let path_env = std::env::var_os("PATH").unwrap_or_default();
    let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string());
    let plan = python_venv_stage_plan(install_root, hermes_home, path_env, &pathext)?;
    if plan.venv.exists() {
        fs::remove_dir_all(&plan.venv)
            .with_context(|| format!("removing existing venv {}", plan.venv.display()))?;
    }
    let status = Command::new(&plan.uv)
        .args(["venv", "venv", "--python", "3.11"])
        .current_dir(&plan.cwd)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("running {}", plan.uv.display()))?;
    if !status.success() {
        return Err(anyhow!(
            "uv venv failed with exit {:?}",
            status.code()
        ));
    }
    let python = venv_python_path(&plan.venv);
    if !python.is_file() {
        return Err(anyhow!(
            "uv venv completed but Python was not created at {}",
            python.display()
        ));
    }
    Ok(serde_json::json!({
        "uv": plan.uv,
        "venv": plan.venv,
        "python": python,
    }))
}

/// Build the native Python dependencies stage plan.
pub fn python_dependencies_stage_plan<P>(
    install_root: &Path,
    hermes_home: &Path,
    path_env: P,
    pathext: &str,
) -> Result<PythonDependenciesStagePlan>
where
    P: AsRef<OsStr>,
{
    if !install_root.is_dir() {
        return Err(anyhow!(
            "install root does not exist: {}",
            install_root.display()
        ));
    }
    let lockfile = install_root.join("uv.lock");
    if !lockfile.is_file() {
        return Err(anyhow!("uv.lock not found at {}", lockfile.display()));
    }
    let uv = uv_tool_path(hermes_home, path_env, pathext)?;
    let venv = install_root.join("venv");
    let python = venv_python_path(&venv);
    Ok(PythonDependenciesStagePlan {
        uv,
        cwd: install_root.to_path_buf(),
        venv,
        python,
        lockfile,
    })
}

/// Install Python dependencies through the hash-verified uv.lock path.
pub fn sync_python_dependencies_stage(
    install_root: &Path,
    hermes_home: &Path,
) -> Result<serde_json::Value> {
    let path_env = std::env::var_os("PATH").unwrap_or_default();
    let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string());
    let plan = python_dependencies_stage_plan(install_root, hermes_home, path_env, &pathext)?;
    let status = Command::new(&plan.uv)
        .args(["sync", "--extra", "all", "--locked"])
        .current_dir(&plan.cwd)
        .env("VIRTUAL_ENV", &plan.venv)
        .env("UV_PYTHON", &plan.python)
        .env("UV_PROJECT_ENVIRONMENT", &plan.venv)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("running {}", plan.uv.display()))?;
    if !status.success() {
        return Err(anyhow!("uv sync failed with exit {:?}", status.code()));
    }
    let baseline = Command::new(&plan.python)
        .args(["-c", "import dotenv, openai, rich, prompt_toolkit"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("checking baseline imports with {}", plan.python.display()))?;
    if !baseline.success() {
        return Err(anyhow!(
            "baseline imports failed after uv sync with exit {:?}",
            baseline.code()
        ));
    }
    Ok(serde_json::json!({
        "uv": plan.uv,
        "venv": plan.venv,
        "python": plan.python,
        "tier": "hash-verified (uv.lock)",
    }))
}

/// Build the native Node dependencies stage plan.
pub fn node_dependencies_stage_plan<P>(
    install_root: &Path,
    hermes_home: &Path,
    path_env: P,
    pathext: &str,
) -> Result<NodeDependenciesStagePlan>
where
    P: AsRef<OsStr>,
{
    let npm = find_npm_executable(hermes_home, path_env.as_ref(), pathext)
        .ok_or_else(|| anyhow!("npm is not available"))?;
    let npx = find_npx_executable(&npm, path_env, pathext);
    let tui_dir = install_root.join("ui-tui");
    let tui_dir = if tui_dir.join("package.json").is_file() {
        Some(tui_dir)
    } else {
        None
    };
    Ok(NodeDependenciesStagePlan {
        npm,
        npx,
        cwd: install_root.to_path_buf(),
        browser_tools: install_root.join("package.json").is_file(),
        tui_dir,
    })
}

/// Install Node dependencies natively on Windows before falling back to PowerShell.
pub fn install_windows_node_dependencies_stage(
    install_root: &Path,
    hermes_home: &Path,
) -> Result<serde_json::Value> {
    if !cfg!(target_os = "windows") {
        return Err(anyhow!("native node-deps stage is only available on Windows"));
    }
    let path_env = std::env::var_os("PATH").unwrap_or_default();
    let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string());
    let plan = node_dependencies_stage_plan(install_root, hermes_home, path_env, &pathext)?;
    if plan.browser_tools {
        run_node_dependency_command(&plan.npm, ["install", "--silent"], &plan.cwd)
            .context("installing root Node dependencies")?;
        let npx = plan
            .npx
            .as_ref()
            .ok_or_else(|| anyhow!("npx is not available"))?;
        run_node_dependency_command(
            npx,
            ["--yes", "playwright", "install", "chromium"],
            &plan.cwd,
        )
        .context("installing Playwright Chromium")?;
    }
    if let Some(tui_dir) = &plan.tui_dir {
        run_node_dependency_command(&plan.npm, ["install", "--silent"], tui_dir)
            .context("installing TUI Node dependencies")?;
    }
    Ok(serde_json::json!({
        "npm": plan.npm,
        "npx": plan.npx,
        "browserTools": plan.browser_tools,
        "tui": plan.tui_dir.is_some(),
    }))
}

/// Build the native desktop app stage plan.
pub fn desktop_build_stage_plan<P>(
    install_root: &Path,
    hermes_home: &Path,
    path_env: P,
    pathext: &str,
) -> Result<DesktopBuildStagePlan>
where
    P: AsRef<OsStr>,
{
    let npm = find_npm_executable(hermes_home, path_env, pathext)
        .ok_or_else(|| anyhow!("npm is not available"))?;
    let desktop_dir = install_root.join("apps").join("desktop");
    if !desktop_dir.join("package.json").is_file() {
        return Err(anyhow!(
            "desktop package not found at {}",
            desktop_dir.join("package.json").display()
        ));
    }
    Ok(DesktopBuildStagePlan {
        npm,
        cwd: install_root.to_path_buf(),
        desktop_dir,
    })
}

/// Build the desktop app natively on Windows before falling back to PowerShell.
pub fn build_windows_desktop_stage(
    install_root: &Path,
    hermes_home: &Path,
) -> Result<serde_json::Value> {
    if !cfg!(target_os = "windows") {
        return Err(anyhow!("native desktop stage is only available on Windows"));
    }
    let path_env = std::env::var_os("PATH").unwrap_or_default();
    let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string());
    let plan = desktop_build_stage_plan(install_root, hermes_home, path_env, &pathext)?;
    if run_node_dependency_command(&plan.npm, ["ci"], &plan.cwd).is_err() {
        run_node_dependency_command(&plan.npm, ["install"], &plan.cwd)
            .context("installing desktop workspace Node dependencies")?;
    }
    run_desktop_pack_command(&plan.npm, &plan.desktop_dir)?;
    let desktop_exe = find_built_windows_desktop_exe(install_root)
        .ok_or_else(|| anyhow!("desktop build completed but Hermes.exe was not found"))?;
    Ok(serde_json::json!({
        "npm": plan.npm,
        "desktopDir": plan.desktop_dir,
        "desktopExe": desktop_exe,
    }))
}

/// Build a native Windows PATH stage report without mutating user state.
pub fn windows_path_stage_plan(
    hermes_home: &Path,
    install_root: &Path,
    current_user_path: Option<String>,
    current_user_hermes_home: Option<String>,
) -> serde_json::Value {
    let path_plan = hermes_manager::platform::plan_path_update(install_root, current_user_path, true);
    let desired_home = hermes_home.display().to_string();
    let hermes_home_changed = current_user_hermes_home
        .as_deref()
        .map(|value| !value.eq_ignore_ascii_case(&desired_home))
        .unwrap_or(true);

    serde_json::json!({
        "hermesHome": hermes_home,
        "hermesBin": path_plan.hermes_bin,
        "pathChanged": path_plan.changed,
        "hermesHomeChanged": hermes_home_changed,
        "applied": false,
    })
}

/// Apply the native Windows PATH stage.
#[cfg(target_os = "windows")]
pub fn configure_windows_path_stage(hermes_home: &Path, install_root: &Path) -> Result<serde_json::Value> {
    let current_user_path = hermes_manager::platform::read_windows_user_path()?;
    let current_user_hermes_home =
        hermes_manager::platform::read_windows_user_env_var("HERMES_HOME")?;
    let mut report = windows_path_stage_plan(
        hermes_home,
        install_root,
        current_user_path.clone(),
        current_user_hermes_home.clone(),
    );
    let path_plan = hermes_manager::platform::plan_path_update(install_root, current_user_path, true);
    let path_applied = hermes_manager::platform::write_windows_user_path_update(&path_plan)?;
    let desired_home = hermes_home.display().to_string();
    let home_changed = current_user_hermes_home
        .as_deref()
        .map(|value| !value.eq_ignore_ascii_case(&desired_home))
        .unwrap_or(true);
    if home_changed {
        hermes_manager::platform::write_windows_user_env_var("HERMES_HOME", &desired_home)?;
    }
    std::env::set_var("HERMES_HOME", &desired_home);
    refresh_process_path(install_root);
    report["applied"] = serde_json::Value::Bool(path_applied || home_changed);
    Ok(report)
}

/// Reject accidental Windows PATH calls on Unix builds.
#[cfg(not(target_os = "windows"))]
pub fn configure_windows_path_stage(_hermes_home: &Path, _install_root: &Path) -> Result<serde_json::Value> {
    Err(anyhow!("native PATH stage is only available on Windows"))
}

/// Apply the native Unix shell-profile PATH stage.
pub fn configure_unix_path_stage(install_root: &Path) -> Result<serde_json::Value> {
    let profile_path = default_unix_profile_path()
        .ok_or_else(|| anyhow!("could not resolve a writable Unix shell profile path"))?;
    configure_unix_path_stage_with_profile(install_root, &profile_path, std::env::var("PATH").ok())
}

fn configure_unix_path_stage_with_profile(
    install_root: &Path,
    profile_path: &Path,
    current_path: Option<String>,
) -> Result<serde_json::Value> {
    let plan = hermes_manager::platform::plan_path_update(install_root, current_path, false);
    let before = std::fs::read_to_string(profile_path).ok();
    hermes_manager::platform::write_shell_profile_update(profile_path, &plan)
        .map_err(|err| anyhow!("writing Unix shell profile update: {err}"))?;
    let after = std::fs::read_to_string(profile_path).ok();
    std::env::set_var("PATH", &plan.next_path);
    Ok(serde_json::json!({
        "profilePath": profile_path.display().to_string(),
        "hermesBin": plan.hermes_bin,
        "pathChanged": plan.changed,
        "profileChanged": before != after,
        "applied": plan.changed || before != after,
    }))
}

fn default_unix_profile_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    let shell = std::env::var("SHELL").unwrap_or_default();
    let name = if shell.ends_with("zsh") { ".zshrc" } else { ".profile" };
    Some(home.join(name))
}

#[cfg(target_os = "windows")]
fn refresh_process_path(install_root: &Path) {
    let current = std::env::var("PATH").ok();
    let plan = hermes_manager::platform::plan_path_update(install_root, current, true);
    std::env::set_var("PATH", plan.next_path);
}

/// Build the repository archive selector used by the native fresh-install path.
pub fn repository_archive_spec(commit: Option<&str>, branch: Option<&str>) -> crate::repo_archive::RepoArchiveSpec {
    crate::repo_archive::RepoArchiveSpec {
        owner: "NousResearch".to_string(),
        repo: "hermes-agent".to_string(),
        commit: commit.filter(|value| !value.trim().is_empty()).map(str::to_string),
        branch: branch.filter(|value| !value.trim().is_empty()).map(str::to_string),
    }
}

/// Download a GitHub archive into a fresh install root and prepare best-effort Git metadata.
pub async fn install_repository_archive_fresh(
    install_root: &Path,
    commit: Option<&str>,
    branch: Option<&str>,
) -> Result<serde_json::Value> {
    if install_root.exists() {
        return Err(anyhow!(
            "install root already exists; native archive fallback is fresh-install only: {}",
            install_root.display()
        ));
    }

    let spec = repository_archive_spec(commit, branch);
    let archive_path =
        crate::repo_archive::download_and_extract_fresh(&spec, &crate::paths::bootstrap_cache_dir(), install_root)
            .await?;
    let git_initialized = initialize_archive_git_repo(install_root);
    let source_marker = crate::repo_archive::write_archive_source_marker(
        install_root,
        &spec,
        &archive_path,
        git_initialized,
    )?;

    Ok(serde_json::json!({
        "installRoot": install_root,
        "archive": archive_path,
        "gitInitialized": git_initialized,
        "source": source_marker,
    }))
}

fn initialize_archive_git_repo(install_root: &Path) -> bool {
    if !run_git(install_root, ["init"]) {
        return false;
    }
    let _ = run_git(install_root, ["config", "windows.appendAtomically", "false"]);
    let _ = run_git(install_root, ["config", "core.autocrlf", "false"]);
    let _ = run_git(
        install_root,
        ["remote", "add", "origin", "https://github.com/NousResearch/hermes-agent.git"],
    );
    true
}

fn run_git<const N: usize>(install_root: &Path, args: [&str; N]) -> bool {
    Command::new("git")
        .args(["-c", "windows.appendAtomically=false"])
        .args(args)
        .current_dir(install_root)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
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

fn venv_python_path(venv: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    }
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

fn managed_tool_path(hermes_home: &Path, name: &str) -> PathBuf {
    let filename = if cfg!(target_os = "windows") {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    hermes_home.join("bin").join(filename)
}

fn uv_tool_path<P>(hermes_home: &Path, path_env: P, pathext: &str) -> Result<PathBuf>
where
    P: AsRef<OsStr>,
{
    let managed = managed_tool_path(hermes_home, "uv");
    if managed.is_file() {
        return Ok(managed);
    }
    find_executable_on_path("uv", path_env, pathext)
        .ok_or_else(|| anyhow!("uv is not available"))
}

fn latest_windows_node_archive_name(
    index_html: &str,
    version_major: u32,
    arch: &str,
) -> Option<String> {
    let prefix = format!("node-v{version_major}.");
    let suffix = format!("-win-{arch}.zip");
    index_html
        .split('"')
        .flat_map(|part| part.split_whitespace())
        .filter_map(|part| {
            let name = part.trim_matches(|ch: char| {
                matches!(ch, '<' | '>' | '\'' | '"' | '=')
            });
            if name.starts_with(&prefix) && name.ends_with(&suffix) {
                Some(name.to_string())
            } else {
                None
            }
        })
        .max_by(|left, right| compare_node_archive_versions(left, right))
}

fn latest_bundled_windows_node_archive_name(
    bundled_tools_dir: Option<&Path>,
    version_major: u32,
    arch: &str,
) -> Option<String> {
    let bundled_tools_dir = bundled_tools_dir?;
    let entries = fs::read_dir(bundled_tools_dir).ok()?;
    entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| windows_node_archive_matches(name, version_major, arch))
        .max_by(|left, right| compare_node_archive_versions(left, right))
}

fn windows_node_archive_matches(name: &str, version_major: u32, arch: &str) -> bool {
    let expected_suffix = format!("-win-{arch}.zip");
    name.starts_with("node-v")
        && name.ends_with(&expected_suffix)
        && node_archive_version_tuple(name).0 == version_major
}

fn bootstrap_archive_cache_path(hermes_home: &Path, archive_name: &str) -> PathBuf {
    hermes_home.join("bootstrap-cache").join(archive_name)
}

fn resolve_bootstrap_archive_source(
    hermes_home: &Path,
    bundled_tools_dir: Option<&Path>,
    archive_name: &str,
) -> ResolvedBootstrapArchive {
    let cache_path = bootstrap_archive_cache_path(hermes_home, archive_name);
    let bundled_path = bundled_tools_dir
        .map(|dir| dir.join(archive_name))
        .filter(|path| path.is_file());
    if let Some(path) = bundled_path {
        return ResolvedBootstrapArchive {
            path,
            cache_path,
            kind: BootstrapArchiveSourceKind::Bundled,
        };
    }
    ResolvedBootstrapArchive {
        path: cache_path.clone(),
        cache_path,
        kind: BootstrapArchiveSourceKind::Cache,
    }
}

fn windows_uv_archive_name(arch: &str) -> Option<&'static str> {
    match arch {
        "x64" => Some("uv-x86_64-pc-windows-msvc.zip"),
        "arm64" => Some("uv-aarch64-pc-windows-msvc.zip"),
        "x86" => Some("uv-i686-pc-windows-msvc.zip"),
        _ => None,
    }
}

fn compare_node_archive_versions(left: &str, right: &str) -> std::cmp::Ordering {
    node_archive_version_tuple(left).cmp(&node_archive_version_tuple(right))
}

fn node_archive_version_tuple(name: &str) -> (u32, u32, u32) {
    let version = name
        .strip_prefix("node-v")
        .and_then(|rest| rest.split_once('-').map(|(version, _)| version))
        .unwrap_or_default();
    let mut parts = version
        .split('.')
        .filter_map(|part| part.parse::<u32>().ok());
    (
        parts.next().unwrap_or_default(),
        parts.next().unwrap_or_default(),
        parts.next().unwrap_or_default(),
    )
}

fn install_windows_node_archive(archive_path: &Path, install_dir: &Path) -> Result<()> {
    let parent = install_dir.parent().ok_or_else(|| {
        anyhow!(
            "Node install directory has no parent: {}",
            install_dir.display()
        )
    })?;
    fs::create_dir_all(parent)
        .with_context(|| format!("creating Node install parent {}", parent.display()))?;
    let tmp_dir = install_dir.with_extension("extracting");
    remove_path_if_exists(&tmp_dir)?;
    fs::create_dir_all(&tmp_dir)
        .with_context(|| format!("creating Node extraction directory {}", tmp_dir.display()))?;

    let result: Result<()> = (|| {
        crate::artifact::extract_zip_archive(archive_path, &tmp_dir)?;
        let extracted_root = single_child_dir(&tmp_dir)?;
        remove_path_if_exists(install_dir)?;
        fs::rename(&extracted_root, install_dir).with_context(|| {
            format!(
                "moving Node runtime {} to {}",
                extracted_root.display(),
                install_dir.display()
            )
        })?;
        Ok(())
    })();
    let cleanup = remove_path_if_exists(&tmp_dir);
    result?;
    cleanup
}

fn install_windows_uv_archive(archive_path: &Path, install_dir: &Path) -> Result<()> {
    fs::create_dir_all(install_dir)
        .with_context(|| format!("creating uv install dir {}", install_dir.display()))?;
    let tmp_dir = install_dir.join("uv-extracting");
    remove_path_if_exists(&tmp_dir)?;
    fs::create_dir_all(&tmp_dir)
        .with_context(|| format!("creating uv extraction directory {}", tmp_dir.display()))?;

    let result: Result<()> = (|| {
        crate::artifact::extract_zip_archive(archive_path, &tmp_dir)?;
        let uv = find_file_named(&tmp_dir, "uv.exe")?;
        fs::copy(&uv, install_dir.join("uv.exe")).with_context(|| {
            format!(
                "copying uv binary {} to {}",
                uv.display(),
                install_dir.join("uv.exe").display()
            )
        })?;
        if let Ok(uvx) = find_file_named(&tmp_dir, "uvx.exe") {
            fs::copy(&uvx, install_dir.join("uvx.exe")).with_context(|| {
                format!(
                    "copying uvx binary {} to {}",
                    uvx.display(),
                    install_dir.join("uvx.exe").display()
                )
            })?;
        }
        Ok(())
    })();
    let cleanup = remove_path_if_exists(&tmp_dir);
    result?;
    cleanup
}

fn install_windows_git_archive(plan: &WindowsGitRuntimeStagePlan) -> Result<()> {
    remove_path_if_exists(&plan.install_dir)?;
    fs::create_dir_all(&plan.install_dir)
        .with_context(|| format!("creating Git install dir {}", plan.install_dir.display()))?;
    if plan.is_zip {
        crate::artifact::extract_zip_archive(&plan.archive_path, &plan.install_dir)?;
        return Ok(());
    }
    let output_arg = format!("-o{}", plan.install_dir.display());
    let status = Command::new(&plan.archive_path)
        .args([output_arg.as_str(), "-y"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("extracting {}", plan.archive_path.display()))?;
    if status.success() {
        return Ok(());
    }
    Err(anyhow!(
        "PortableGit extraction failed with exit {:?}",
        status.code()
    ))
}

fn find_windows_git_bash(install_dir: &Path) -> Option<PathBuf> {
    [
        install_dir.join("bin").join("bash.exe"),
        install_dir.join("usr").join("bin").join("bash.exe"),
    ]
    .into_iter()
    .find(|path| path.is_file())
}

fn find_file_named(root: &Path, name: &str) -> Result<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
            let entry = entry.with_context(|| format!("reading entry under {}", dir.display()))?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path
                .file_name()
                .and_then(|value| value.to_str())
                .map(|value| value.eq_ignore_ascii_case(name))
                .unwrap_or(false)
            {
                return Ok(path);
            }
        }
    }
    Err(anyhow!("{} not found under {}", name, root.display()))
}

fn single_child_dir(parent: &Path) -> Result<PathBuf> {
    let mut dirs = fs::read_dir(parent)
        .with_context(|| format!("reading {}", parent.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    dirs.sort();
    if dirs.len() != 1 {
        return Err(anyhow!(
            "expected exactly one extracted directory under {}, found {}",
            parent.display(),
            dirs.len()
        ));
    }
    Ok(dirs.remove(0))
}

fn remove_path_if_exists(path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    if path.is_dir() {
        fs::remove_dir_all(path)
            .with_context(|| format!("removing directory {}", path.display()))
    } else {
        fs::remove_file(path)
            .with_context(|| format!("removing file {}", path.display()))
    }
}

fn windows_node_arch_slug() -> String {
    let arch = std::env::var("PROCESSOR_ARCHITEW6432")
        .or_else(|_| std::env::var("PROCESSOR_ARCHITECTURE"))
        .unwrap_or_else(|_| std::env::consts::ARCH.to_string());
    match arch.to_ascii_lowercase().as_str() {
        "amd64" | "x86_64" => "x64".to_string(),
        "arm64" | "aarch64" => "arm64".to_string(),
        "x86" | "i386" | "i686" => "x86".to_string(),
        other => other.to_string(),
    }
}

fn prepend_process_path(entry: &Path) {
    let current = std::env::var_os("PATH").unwrap_or_default();
    let mut parts = vec![entry.to_path_buf()];
    parts.extend(std::env::split_paths(&current));
    if let Ok(next) = std::env::join_paths(parts) {
        std::env::set_var("PATH", next);
    }
}

fn prepend_process_paths(entries: &[PathBuf]) {
    let current = std::env::var_os("PATH").unwrap_or_default();
    let mut parts = entries.to_vec();
    parts.extend(std::env::split_paths(&current));
    if let Ok(next) = std::env::join_paths(parts) {
        std::env::set_var("PATH", next);
    }
}

#[cfg(target_os = "windows")]
fn persist_windows_path_entry(entry: &Path) -> Result<()> {
    persist_windows_path_entries(&[entry.to_path_buf()])
}

#[cfg(not(target_os = "windows"))]
fn persist_windows_path_entry(_entry: &Path) -> Result<()> {
    Err(anyhow!("Windows PATH persistence is only available on Windows"))
}

#[cfg(target_os = "windows")]
fn persist_windows_path_entries(entries: &[PathBuf]) -> Result<()> {
    let current = hermes_manager::platform::read_windows_user_path()?;
    let mut parts = current
        .as_deref()
        .unwrap_or_default()
        .split(';')
        .filter(|part| !part.trim().is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let mut changed = false;
    for entry in entries.iter().rev() {
        let entry_text = entry.display().to_string();
        let exists = parts
            .iter()
            .any(|part| part.eq_ignore_ascii_case(&entry_text));
        if !exists {
            parts.insert(0, entry_text);
            changed = true;
        }
    }
    if changed {
        persist_windows_env_var("Path", &parts.join(";"))?;
    }
    Ok(())
}

#[cfg(not(target_os = "windows"))]
fn persist_windows_path_entries(_entries: &[PathBuf]) -> Result<()> {
    Err(anyhow!("Windows PATH persistence is only available on Windows"))
}

#[cfg(target_os = "windows")]
fn persist_windows_env_var(name: &str, value: &str) -> Result<()> {
    hermes_manager::platform::write_windows_user_env_var(name, value)?;
    Ok(())
}

#[cfg(not(target_os = "windows"))]
fn persist_windows_env_var(_name: &str, _value: &str) -> Result<()> {
    Err(anyhow!("Windows environment persistence is only available on Windows"))
}

fn find_node_executable<P>(hermes_home: &Path, path_env: P, pathext: &str) -> Option<PathBuf>
where
    P: AsRef<OsStr>,
{
    let managed = if cfg!(target_os = "windows") {
        hermes_home.join("node").join("node.exe")
    } else {
        hermes_home.join("node").join("bin").join("node")
    };
    if managed.is_file() {
        return Some(managed);
    }
    find_executable_on_path("node", path_env, pathext)
}

fn find_npm_executable<P>(hermes_home: &Path, path_env: P, pathext: &str) -> Option<PathBuf>
where
    P: AsRef<OsStr>,
{
    let managed = if cfg!(target_os = "windows") {
        hermes_home.join("node").join("npm.cmd")
    } else {
        hermes_home.join("node").join("bin").join("npm")
    };
    if managed.is_file() {
        return Some(managed);
    }
    find_executable_on_path("npm", path_env, pathext)
}

fn find_npx_executable<P>(npm: &Path, path_env: P, pathext: &str) -> Option<PathBuf>
where
    P: AsRef<OsStr>,
{
    let npm_dir = npm.parent()?;
    for candidate in executable_candidates("npx", pathext) {
        let path = npm_dir.join(candidate);
        if path.is_file() {
            return Some(path);
        }
    }
    find_executable_on_path("npx", path_env, pathext)
}

fn run_node_dependency_command<const N: usize>(
    command: &Path,
    args: [&str; N],
    cwd: &Path,
) -> Result<()> {
    let status = Command::new(command)
        .args(args)
        .current_dir(cwd)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("running {}", command.display()))?;
    if status.success() {
        return Ok(());
    }
    Err(anyhow!(
        "{} failed with exit {:?}",
        command.display(),
        status.code()
    ))
}

fn run_desktop_pack_command(npm: &Path, desktop_dir: &Path) -> Result<()> {
    let status = Command::new(npm)
        .args(["run", "pack"])
        .current_dir(desktop_dir)
        .env("CSC_IDENTITY_AUTO_DISCOVERY", "false")
        .env("WIN_CSC_LINK", "")
        .env("WIN_CSC_KEY_PASSWORD", "")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("running {} run pack", npm.display()))?;
    if status.success() {
        return Ok(());
    }
    Err(anyhow!(
        "{} run pack failed with exit {:?}",
        npm.display(),
        status.code()
    ))
}

fn find_built_windows_desktop_exe(install_root: &Path) -> Option<PathBuf> {
    let release = install_root.join("apps").join("desktop").join("release");
    [
        release.join("win-unpacked").join("Hermes.exe"),
        release.join("win-arm64-unpacked").join("Hermes.exe"),
    ]
    .into_iter()
    .find(|path| path.is_file())
}

fn node_version_satisfies_build(node: &Path) -> bool {
    let output = Command::new(node)
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output();
    let Ok(output) = output else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    let Ok(version) = String::from_utf8(output.stdout) else {
        return false;
    };
    node_version_string_satisfies_build(version.trim())
}

fn node_version_string_satisfies_build(version: &str) -> bool {
    let cleaned = version
        .trim_start_matches('v')
        .split_once('-')
        .map(|(base, _)| base)
        .unwrap_or(version);
    let parts = cleaned
        .split('.')
        .filter_map(|part| part.parse::<u32>().ok())
        .collect::<Vec<_>>();
    if parts.len() < 2 {
        return false;
    }
    let major = parts[0];
    let minor = parts[1];
    (major == 20 && minor >= 19) || (major == 22 && minor >= 12) || major > 22
}

fn stage_execution_mode(name: &str) -> StageExecutionMode {
    if matches!(
        name.to_ascii_lowercase().as_str(),
        "repository" | "python" | "venv" | "dependencies" | "python-deps" | "platform-sdks"
    ) {
        return StageExecutionMode::NativeWithScriptFallback;
    }
    if cfg!(target_os = "windows") && name.eq_ignore_ascii_case("node-deps") {
        return StageExecutionMode::NativeWithScriptFallback;
    }
    if cfg!(target_os = "windows") && name.eq_ignore_ascii_case("desktop") {
        return StageExecutionMode::NativeWithScriptFallback;
    }
    if cfg!(target_os = "windows") && name.eq_ignore_ascii_case("node") {
        return StageExecutionMode::NativeWithScriptFallback;
    }
    if cfg!(target_os = "windows") && name.eq_ignore_ascii_case("uv") {
        return StageExecutionMode::NativeWithScriptFallback;
    }
    if cfg!(target_os = "windows") && name.eq_ignore_ascii_case("git") {
        return StageExecutionMode::NativeWithScriptFallback;
    }
    if matches!(
        name.to_ascii_lowercase().as_str(),
        "bootstrap-marker" | "config" | "config-templates" | "complete" | "path"
    ) {
        return StageExecutionMode::Native;
    }
    if matches!(
        name.to_ascii_lowercase().as_str(),
        "uv" | "git" | "node" | "system-packages" | "node-deps" | "desktop"
    ) {
        return StageExecutionMode::ProbeThenScript;
    }
    StageExecutionMode::Script
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
    fn build_stage_plan_classifies_native_probe_and_script_stages() {
        let stages = vec![
            stage("repository"),
            stage("path"),
            stage("python"),
            stage("uv"),
            stage("git"),
            stage("node"),
            stage("platform-sdks"),
            stage("node-deps"),
            stage("desktop"),
            stage("venv"),
            stage("dependencies"),
            stage("python-deps"),
        ];
        let plan = build_stage_plan(&stages, false);

        assert_eq!(plan.len(), 12);
        assert_eq!(plan[0].name, "repository");
        assert_eq!(plan[0].execution, StageExecutionMode::NativeWithScriptFallback);
        assert_eq!(plan[0].script_fallback, true);
        assert_eq!(plan[1].name, "path");
        assert_eq!(plan[1].execution, StageExecutionMode::Native);
        assert_eq!(plan[1].script_fallback, false);
        assert_eq!(plan[2].name, "python");
        assert_eq!(plan[2].execution, StageExecutionMode::NativeWithScriptFallback);
        assert_eq!(plan[2].script_fallback, true);
        let uv_execution = if cfg!(target_os = "windows") {
            StageExecutionMode::NativeWithScriptFallback
        } else {
            StageExecutionMode::ProbeThenScript
        };
        assert_eq!(plan[3].name, "uv");
        assert_eq!(plan[3].execution, uv_execution);
        assert_eq!(plan[3].rust_probe, !cfg!(target_os = "windows"));
        assert_eq!(plan[3].script_fallback, true);
        let git_execution = if cfg!(target_os = "windows") {
            StageExecutionMode::NativeWithScriptFallback
        } else {
            StageExecutionMode::ProbeThenScript
        };
        assert_eq!(plan[4].name, "git");
        assert_eq!(plan[4].execution, git_execution);
        assert_eq!(plan[4].rust_probe, !cfg!(target_os = "windows"));
        assert_eq!(plan[4].script_fallback, true);
        let node_execution = if cfg!(target_os = "windows") {
            StageExecutionMode::NativeWithScriptFallback
        } else {
            StageExecutionMode::ProbeThenScript
        };
        assert_eq!(plan[5].name, "node");
        assert_eq!(plan[5].execution, node_execution);
        assert_eq!(plan[5].rust_probe, !cfg!(target_os = "windows"));
        assert_eq!(plan[5].script_fallback, true);
        assert_eq!(plan[6].name, "platform-sdks");
        assert_eq!(plan[6].execution, StageExecutionMode::NativeWithScriptFallback);
        assert_eq!(plan[6].script_fallback, true);
        let node_deps_execution = if cfg!(target_os = "windows") {
            StageExecutionMode::NativeWithScriptFallback
        } else {
            StageExecutionMode::ProbeThenScript
        };
        assert_eq!(plan[7].name, "node-deps");
        assert_eq!(plan[7].execution, node_deps_execution);
        assert_eq!(plan[7].rust_probe, !cfg!(target_os = "windows"));
        assert_eq!(plan[7].script_fallback, true);
        assert_eq!(plan[8].name, "desktop");
        let desktop_execution = if cfg!(target_os = "windows") {
            StageExecutionMode::NativeWithScriptFallback
        } else {
            StageExecutionMode::ProbeThenScript
        };
        assert_eq!(plan[8].execution, desktop_execution);
        assert_eq!(plan[8].rust_probe, !cfg!(target_os = "windows"));
        assert_eq!(plan[8].script_fallback, true);
        assert_eq!(plan[9].name, "venv");
        assert_eq!(plan[9].execution, StageExecutionMode::NativeWithScriptFallback);
        assert_eq!(plan[9].script_fallback, true);
        assert_eq!(plan[10].name, "dependencies");
        assert_eq!(plan[10].execution, StageExecutionMode::NativeWithScriptFallback);
        assert_eq!(plan[10].script_fallback, true);
        assert_eq!(plan[11].name, "python-deps");
        assert_eq!(plan[11].execution, StageExecutionMode::NativeWithScriptFallback);
        assert_eq!(plan[11].script_fallback, true);
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
    fn summarize_plan_reports_native_probe_and_script_coverage() {
        let hermes_home = PathBuf::from("C:/Users/example/AppData/Local/hermes");
        let report = install_state_report(
            &hermes_home,
            vec![ToolProbe {
                name: "uv".to_string(),
                path: None,
            }],
        );
        let stages = vec![stage("repository"), stage("path"), stage("uv"), stage("venv")];
        let plan = build_stage_plan(&stages, false);

        let summary = summarize_plan(&report, &plan);

        let native_count = if cfg!(target_os = "windows") { 4 } else { 3 };
        let probe_count = if cfg!(target_os = "windows") { 0 } else { 1 };
        assert!(summary.contains(&format!("native_stages={native_count}")));
        assert!(summary.contains(&format!("probe_stages={probe_count}")));
        assert!(summary.contains("script_stages=0"));
        assert!(summary.contains("total_stages=4"));
        assert!(summary.contains("uv=missing"));
    }

    #[test]
    fn native_bootstrap_manifest_matches_windows_stage_contract() {
        let manifest = native_bootstrap_manifest(crate::install_script::ScriptKind::Ps1, true);
        let names = manifest
            .stages
            .iter()
            .map(|stage| stage.name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(manifest.protocol_version, Some(1));
        assert_eq!(
            names,
            vec![
                "uv",
                "python",
                "git",
                "node",
                "system-packages",
                "repository",
                "venv",
                "dependencies",
                "node-deps",
                "desktop",
                "path",
                "config-templates",
                "platform-sdks",
                "bootstrap-marker",
                "configure",
                "gateway",
            ]
        );
        assert_eq!(manifest.stages[9].title, "Building desktop app");
        assert!(manifest.stages[14].needs_user_input);
    }

    #[test]
    fn native_bootstrap_manifest_matches_unix_stage_contract() {
        let manifest = native_bootstrap_manifest(crate::install_script::ScriptKind::Sh, false);
        let names = manifest
            .stages
            .iter()
            .map(|stage| stage.name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(manifest.protocol_version, Some(1));
        assert_eq!(
            names,
            vec![
                "prerequisites",
                "repository",
                "venv",
                "python-deps",
                "node-deps",
                "path",
                "config",
                "setup",
                "gateway",
                "complete",
            ]
        );
        assert_eq!(manifest.stages[6].title, "Prepare config and skills");
        assert!(manifest.stages[7].needs_user_input);
    }

    #[test]
    fn interactive_stage_skip_result_only_skips_user_input_stages() {
        let setup = stage_info("setup", "Configure API keys and settings", "configuration", true);
        let path = stage_info("path", "Install hermes command", "runtime", false);

        let skipped = interactive_stage_skip_result(&setup).unwrap();

        assert_eq!(skipped.stage, "setup");
        assert_eq!(skipped.ok, true);
        assert_eq!(skipped.skipped, true);
        assert_eq!(
            skipped.reason.as_deref(),
            Some("requires user input; handled by post-install UI")
        );
        assert!(skipped.data.is_none());
        assert!(interactive_stage_skip_result(&path).is_none());
    }

    #[test]
    fn satisfied_tool_stage_skip_result_skips_only_when_tools_are_present() {
        let root = std::env::temp_dir().join(format!(
            "hermes-tool-skip-test-{}",
            std::process::id()
        ));
        let hermes_home = root.join("home");
        let tools = root.join("tools");
        std::fs::create_dir_all(hermes_home.join("bin")).unwrap();
        std::fs::create_dir_all(&tools).unwrap();
        std::fs::write(hermes_home.join("bin").join("uv.exe"), b"uv").unwrap();
        std::fs::write(tools.join("git.exe"), b"git").unwrap();
        std::fs::write(tools.join("node.exe"), b"node").unwrap();
        std::fs::write(tools.join("npm.cmd"), b"npm").unwrap();
        std::fs::write(tools.join("rg.exe"), b"rg").unwrap();

        let uv = stage_info("uv", "Installing uv package manager", "prereqs", false);
        let git = stage_info("git", "Installing Git", "prereqs", false);
        let node = stage_info("node", "Detecting Node.js", "prereqs", false);
        let system_packages = stage_info(
            "system-packages",
            "Installing ripgrep and ffmpeg",
            "prereqs",
            false,
        );
        let venv = stage_info(
            "venv",
            "Creating Python virtual environment",
            "install",
            false,
        );

        let uv_result = satisfied_tool_stage_skip_result(&uv, &hermes_home, &tools, ".EXE").unwrap();
        let git_result = satisfied_tool_stage_skip_result(&git, &hermes_home, &tools, ".EXE").unwrap();
        let old_node_result = satisfied_tool_stage_skip_result(&node, &hermes_home, &tools, ".EXE;.CMD");

        assert_eq!(uv_result.stage, "uv");
        assert_eq!(uv_result.ok, true);
        assert_eq!(uv_result.skipped, true);
        assert_eq!(
            uv_result.reason.as_deref(),
            Some("required tool already available")
        );
        assert_eq!(git_result.stage, "git");
        assert!(old_node_result.is_none());
        let new_node_result = satisfied_tool_stage_skip_result_with_node_probe(
            &node,
            &hermes_home,
            &tools,
            ".EXE;.CMD",
            |_| true,
        )
        .unwrap();
        assert_eq!(new_node_result.stage, "node");
        assert_eq!(new_node_result.reason.as_deref(), Some("required tool already available"));
        assert!(satisfied_tool_stage_skip_result(
            &system_packages,
            &hermes_home,
            &tools,
            ".EXE"
        )
        .is_none());
        std::fs::write(tools.join("ffmpeg.exe"), b"ffmpeg").unwrap();
        assert!(satisfied_tool_stage_skip_result(
            &system_packages,
            &hermes_home,
            &tools,
            ".EXE"
        )
        .is_some());
        assert!(satisfied_tool_stage_skip_result(&venv, &hermes_home, &tools, ".EXE").is_none());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn node_deps_skip_result_skips_only_when_npm_is_absent() {
        let root = std::env::temp_dir().join(format!(
            "hermes-node-deps-skip-test-{}",
            std::process::id()
        ));
        let hermes_home = root.join("home");
        let tools = root.join("tools");
        std::fs::create_dir_all(&tools).unwrap();

        let node_deps = stage_info("node-deps", "Installing Node.js dependencies", "install", false);
        let skipped = node_deps_skip_result(&node_deps, &hermes_home, &tools, ".EXE;.CMD")
            .unwrap();

        assert_eq!(skipped.stage, "node-deps");
        assert_eq!(skipped.ok, true);
        assert_eq!(skipped.skipped, true);
        assert_eq!(skipped.reason.as_deref(), Some("npm not available"));
        std::fs::write(tools.join("npm.cmd"), b"npm").unwrap();
        assert!(node_deps_skip_result(&node_deps, &hermes_home, &tools, ".EXE;.CMD").is_none());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn windows_node_runtime_stage_plan_parses_latest_v22_zip() {
        let html = r#"
            <a href="node-v22.18.0-win-x64.zip">node-v22.18.0-win-x64.zip</a>
            <a href="node-v22.19.1-win-x64.zip">node-v22.19.1-win-x64.zip</a>
            <a href="node-v22.19.1-win-arm64.zip">node-v22.19.1-win-arm64.zip</a>
        "#;
        let root = std::env::temp_dir().join(format!(
            "hermes-node-runtime-plan-test-{}",
            std::process::id()
        ));
        let hermes_home = root.join("home");

        let plan = windows_node_runtime_stage_plan_from_index(&hermes_home, "x64", html).unwrap();

        assert_eq!(plan.version_major, 22);
        assert_eq!(plan.archive_name, "node-v22.19.1-win-x64.zip");
        assert_eq!(
            plan.download_url,
            "https://nodejs.org/dist/latest-v22.x/node-v22.19.1-win-x64.zip"
        );
        assert_eq!(plan.install_dir, hermes_home.join("node"));
        assert_eq!(plan.node_exe, hermes_home.join("node").join("node.exe"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn bootstrap_archive_source_prefers_bundled_resource_over_cache() {
        let root = std::env::temp_dir().join(format!(
            "hermes-bootstrap-archive-source-test-{}",
            std::process::id()
        ));
        let hermes_home = root.join("home");
        let bundled = root.join("resources").join("bootstrap-tools");
        std::fs::create_dir_all(&bundled).unwrap();
        std::fs::write(bundled.join("uv-x86_64-pc-windows-msvc.zip"), b"uv").unwrap();

        let source = resolve_bootstrap_archive_source(
            &hermes_home,
            Some(&bundled),
            "uv-x86_64-pc-windows-msvc.zip",
        );

        assert_eq!(source.kind, BootstrapArchiveSourceKind::Bundled);
        assert_eq!(
            source.path,
            bundled.join("uv-x86_64-pc-windows-msvc.zip")
        );
        assert_eq!(
            source.cache_path,
            hermes_home
                .join("bootstrap-cache")
                .join("uv-x86_64-pc-windows-msvc.zip")
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn bootstrap_archive_source_falls_back_to_cache_when_resource_is_absent() {
        let root = std::env::temp_dir().join(format!(
            "hermes-bootstrap-archive-cache-test-{}",
            std::process::id()
        ));
        let hermes_home = root.join("home");
        let bundled = root.join("resources").join("bootstrap-tools");
        std::fs::create_dir_all(&bundled).unwrap();

        let source = resolve_bootstrap_archive_source(
            &hermes_home,
            Some(&bundled),
            "uv-x86_64-pc-windows-msvc.zip",
        );

        assert_eq!(source.kind, BootstrapArchiveSourceKind::Cache);
        assert_eq!(
            source.path,
            hermes_home
                .join("bootstrap-cache")
                .join("uv-x86_64-pc-windows-msvc.zip")
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn bundled_node_archive_picker_uses_latest_matching_v22_arch() {
        let root = std::env::temp_dir().join(format!(
            "hermes-bundled-node-archive-test-{}",
            std::process::id()
        ));
        let bundled = root.join("resources").join("bootstrap-tools");
        std::fs::create_dir_all(&bundled).unwrap();
        for name in [
            "node-v22.18.0-win-x64.zip",
            "node-v22.20.1-win-arm64.zip",
            "node-v22.19.1-win-x64.zip",
            "node-v21.7.3-win-x64.zip",
            "node-v22.19.2-win-x86.zip",
        ] {
            std::fs::write(bundled.join(name), b"node").unwrap();
        }

        let picked = latest_bundled_windows_node_archive_name(Some(&bundled), 22, "x64");

        assert_eq!(picked.as_deref(), Some("node-v22.19.1-win-x64.zip"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn node_dependencies_stage_plan_prefers_managed_tools_and_detects_packages() {
        let root = std::env::temp_dir().join(format!(
            "hermes-node-deps-plan-test-{}",
            std::process::id()
        ));
        let hermes_home = root.join("home");
        let install_root = root.join("checkout");
        let node_home = hermes_home.join("node");
        std::fs::create_dir_all(&node_home).unwrap();
        std::fs::create_dir_all(install_root.join("ui-tui")).unwrap();
        let npm_name = if cfg!(target_os = "windows") { "npm.cmd" } else { "bin/npm" };
        let npx_name = if cfg!(target_os = "windows") { "npx.CMD" } else { "bin/npx" };
        let npm = hermes_home.join("node").join(npm_name);
        let npx = hermes_home.join("node").join(npx_name);
        std::fs::create_dir_all(npm.parent().unwrap()).unwrap();
        std::fs::write(&npm, b"npm").unwrap();
        std::fs::write(&npx, b"npx").unwrap();
        std::fs::write(install_root.join("package.json"), b"{}").unwrap();
        std::fs::write(install_root.join("ui-tui").join("package.json"), b"{}").unwrap();

        let plan =
            node_dependencies_stage_plan(&install_root, &hermes_home, "", ".EXE;.CMD").unwrap();

        assert_eq!(plan.npm, npm);
        assert_eq!(plan.npx.as_deref(), Some(npx.as_path()));
        assert_eq!(plan.cwd, install_root);
        assert_eq!(plan.browser_tools, true);
        assert_eq!(plan.tui_dir.as_deref(), Some(install_root.join("ui-tui").as_path()));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn desktop_stage_skip_result_skips_when_desktop_package_is_absent() {
        let root = std::env::temp_dir().join(format!(
            "hermes-desktop-skip-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();

        let desktop = stage_info("desktop", "Building desktop app", "install", false);
        let skipped = desktop_stage_skip_result(&desktop, &root).unwrap();

        assert_eq!(skipped.stage, "desktop");
        assert_eq!(skipped.ok, true);
        assert_eq!(skipped.skipped, true);
        assert_eq!(
            skipped.reason.as_deref(),
            Some("apps/desktop not present")
        );
        std::fs::create_dir_all(root.join("apps").join("desktop")).unwrap();
        std::fs::write(
            root.join("apps").join("desktop").join("package.json"),
            b"{}",
        )
        .unwrap();
        assert!(desktop_stage_skip_result(&desktop, &root).is_none());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn desktop_build_stage_plan_requires_npm_and_desktop_package() {
        let root = std::env::temp_dir().join(format!(
            "hermes-desktop-build-plan-test-{}",
            std::process::id()
        ));
        let hermes_home = root.join("home");
        let install_root = root.join("checkout");
        let desktop_dir = install_root.join("apps").join("desktop");
        std::fs::create_dir_all(&desktop_dir).unwrap();
        std::fs::write(desktop_dir.join("package.json"), b"{}").unwrap();

        assert!(
            desktop_build_stage_plan(&install_root, &hermes_home, "", ".EXE;.CMD").is_err()
        );

        let npm_name = if cfg!(target_os = "windows") { "npm.cmd" } else { "bin/npm" };
        let npm = hermes_home.join("node").join(npm_name);
        std::fs::create_dir_all(npm.parent().unwrap()).unwrap();
        std::fs::write(&npm, b"npm").unwrap();

        let plan =
            desktop_build_stage_plan(&install_root, &hermes_home, "", ".EXE;.CMD").unwrap();

        assert_eq!(plan.npm, npm);
        assert_eq!(plan.cwd, install_root);
        assert_eq!(plan.desktop_dir, desktop_dir);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn python_stage_skip_result_uses_probe_without_affecting_other_stages() {
        let python = stage_info("python", "Verifying Python 3.11", "prereqs", false);
        let uv = stage_info("uv", "Installing uv package manager", "prereqs", false);

        let skipped = python_stage_skip_result_with_probe(&python, || true).unwrap();

        assert_eq!(skipped.stage, "python");
        assert_eq!(skipped.ok, true);
        assert_eq!(skipped.skipped, true);
        assert_eq!(
            skipped.reason.as_deref(),
            Some("required Python runtime already available")
        );
        assert!(python_stage_skip_result_with_probe(&python, || false).is_none());
        assert!(python_stage_skip_result_with_probe(&uv, || true).is_none());
    }

    #[test]
    fn windows_uv_runtime_stage_plan_maps_arch_to_release_asset() {
        let root = std::env::temp_dir().join(format!(
            "hermes-uv-runtime-plan-test-{}",
            std::process::id()
        ));
        let hermes_home = root.join("home");

        let plan = windows_uv_runtime_stage_plan(&hermes_home, "x64").unwrap();

        assert_eq!(plan.archive_name, "uv-x86_64-pc-windows-msvc.zip");
        assert_eq!(
            plan.download_url,
            "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
        );
        assert_eq!(plan.install_dir, hermes_home.join("bin"));
        assert_eq!(plan.uv_exe, hermes_home.join("bin").join("uv.exe"));

        let arm_plan = windows_uv_runtime_stage_plan(&hermes_home, "arm64").unwrap();
        assert_eq!(arm_plan.archive_name, "uv-aarch64-pc-windows-msvc.zip");
        let x86_plan = windows_uv_runtime_stage_plan(&hermes_home, "x86").unwrap();
        assert_eq!(x86_plan.archive_name, "uv-i686-pc-windows-msvc.zip");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn windows_git_runtime_stage_plan_matches_pinned_portable_git_assets() {
        let root = std::env::temp_dir().join(format!(
            "hermes-git-runtime-plan-test-{}",
            std::process::id()
        ));
        let hermes_home = root.join("home");

        let x64 = windows_git_runtime_stage_plan(&hermes_home, "x64").unwrap();

        assert_eq!(x64.tag, "v2.54.0.windows.1");
        assert_eq!(x64.version, "2.54.0");
        assert_eq!(x64.archive_name, "PortableGit-2.54.0-64-bit.7z.exe");
        assert_eq!(
            x64.download_url,
            concat!(
                "https://github.com/git-for-windows/git/releases/download/",
                "v2.54.0.windows.1/PortableGit-2.54.0-64-bit.7z.exe"
            )
        );
        assert_eq!(x64.install_dir, hermes_home.join("git"));
        assert_eq!(x64.git_exe, hermes_home.join("git").join("cmd").join("git.exe"));
        assert_eq!(x64.bash_exe, hermes_home.join("git").join("bin").join("bash.exe"));
        assert_eq!(x64.is_zip, false);

        let arm = windows_git_runtime_stage_plan(&hermes_home, "arm64").unwrap();
        assert_eq!(arm.archive_name, "PortableGit-2.54.0-arm64.7z.exe");
        let x86 = windows_git_runtime_stage_plan(&hermes_home, "x86").unwrap();
        assert_eq!(x86.archive_name, "MinGit-2.54.0-32-bit.zip");
        assert_eq!(x86.is_zip, true);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn python_runtime_stage_plan_prefers_managed_uv() {
        let root = std::env::temp_dir().join(format!("hermes-python-plan-{}", std::process::id()));
        let hermes_home = root.join("home");
        let path_tools = root.join("tools");
        std::fs::create_dir_all(hermes_home.join("bin")).unwrap();
        std::fs::create_dir_all(&path_tools).unwrap();
        std::fs::write(hermes_home.join("bin").join("uv.exe"), b"managed uv").unwrap();
        std::fs::write(path_tools.join("uv.exe"), b"path uv").unwrap();

        let plan = python_runtime_stage_plan(&hermes_home, &path_tools, ".EXE").unwrap();

        assert_eq!(plan.uv, hermes_home.join("bin").join("uv.exe"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn platform_sdks_skip_result_skips_only_when_no_platform_tokens_are_configured() {
        let root = std::env::temp_dir().join(format!(
            "hermes-platform-sdks-skip-test-{}",
            std::process::id()
        ));
        let hermes_home = root.join("home");
        std::fs::create_dir_all(&hermes_home).unwrap();

        let platform_sdks = stage_info(
            "platform-sdks",
            "Installing messaging platform SDKs",
            "finalize",
            false,
        );
        let missing_env = platform_sdks_skip_result(&platform_sdks, &hermes_home).unwrap();

        assert_eq!(missing_env.stage, "platform-sdks");
        assert_eq!(missing_env.ok, true);
        assert_eq!(missing_env.skipped, true);
        assert_eq!(
            missing_env.reason.as_deref(),
            Some("no messaging platform tokens configured")
        );

        std::fs::write(
            hermes_home.join(".env"),
            "TELEGRAM_BOT_TOKEN=your-token-here\nDISCORD_BOT_TOKEN=\n",
        )
        .unwrap();
        assert!(platform_sdks_skip_result(&platform_sdks, &hermes_home).is_some());

        std::fs::write(hermes_home.join(".env"), "TELEGRAM_BOT_TOKEN=abc123\n").unwrap();
        assert!(platform_sdks_skip_result(&platform_sdks, &hermes_home).is_none());

        let config = stage_info(
            "config-templates",
            "Writing configuration templates",
            "finalize",
            false,
        );
        assert!(platform_sdks_skip_result(&config, &hermes_home).is_none());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn platform_sdk_requirements_map_real_tokens_to_imports_and_specs() {
        let env = concat!(
            "# TELEGRAM_BOT_TOKEN=ignored\n",
            "TELEGRAM_BOT_TOKEN=abc123\n",
            "DISCORD_BOT_TOKEN=your-token-here\n",
            "SLACK_BOT_TOKEN=xoxb-test\n",
            "WHATSAPP_ENABLED=false\n",
            "SLACK_APP_TOKEN=\n",
        );

        let requirements = platform_sdk_requirements_from_env(env);
        let names = requirements
            .iter()
            .map(|sdk| (sdk.env_var, sdk.import_name, sdk.pip_spec))
            .collect::<Vec<_>>();

        assert_eq!(
            names,
            vec![
                (
                    "TELEGRAM_BOT_TOKEN",
                    "telegram",
                    "python-telegram-bot[webhooks]>=22.6,<23",
                ),
                ("SLACK_BOT_TOKEN", "slack_sdk", "slack-sdk>=3.27.0,<4"),
            ]
        );
    }

    #[test]
    fn platform_sdk_stage_plan_uses_venv_python_and_configured_requirements() {
        let root = std::env::temp_dir().join(format!("hermes-platform-plan-{}", std::process::id()));
        let hermes_home = root.join("home");
        let install_root = hermes_home.join("hermes-agent");
        let venv_python = venv_python_path(&install_root.join("venv"));
        std::fs::create_dir_all(venv_python.parent().unwrap()).unwrap();
        std::fs::write(&venv_python, b"python").unwrap();
        std::fs::create_dir_all(&hermes_home).unwrap();
        std::fs::write(hermes_home.join(".env"), "WHATSAPP_ENABLED=true\n").unwrap();

        let plan = platform_sdk_stage_plan(&hermes_home, &install_root).unwrap();

        assert_eq!(plan.python, venv_python);
        assert_eq!(plan.requirements.len(), 1);
        assert_eq!(plan.requirements[0].import_name, "qrcode");
        assert_eq!(plan.requirements[0].pip_spec, "qrcode>=7.0,<8");

        let _ = std::fs::remove_dir_all(&root);
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

    #[test]
    fn windows_path_stage_plan_reports_path_and_hermes_home_changes() {
        let hermes_home = PathBuf::from("C:/Users/example/AppData/Local/hermes");
        let install_root = hermes_home.join("hermes-agent");

        let report = windows_path_stage_plan(
            &hermes_home,
            &install_root,
            Some("C:/Windows/System32".to_string()),
            Some("C:/old/hermes".to_string()),
        );

        assert_eq!(
            report["hermesBin"],
            install_root.join("venv").join("Scripts").display().to_string()
        );
        assert_eq!(report["pathChanged"], true);
        assert_eq!(report["hermesHomeChanged"], true);
        assert_eq!(report["applied"], false);
    }

    #[test]
    fn unix_path_stage_writes_managed_profile_block() {
        let root = std::env::temp_dir().join(format!("hermes-unix-path-{}", std::process::id()));
        let home = root.join("home");
        let install_root = root.join("hermes-agent");
        let profile = home.join(".profile");
        std::fs::create_dir_all(&install_root).unwrap();
        std::fs::create_dir_all(&home).unwrap();
        std::fs::write(&profile, "alias ll='ls -la'\n").unwrap();

        let report = configure_unix_path_stage_with_profile(&install_root, &profile, None).unwrap();

        let text = std::fs::read_to_string(&profile).unwrap();
        assert!(text.contains("alias ll='ls -la'"));
        assert!(text.contains("Hermes Agent PATH"));
        assert!(text.contains(&install_root.join("venv").join("bin").display().to_string()));
        assert_eq!(report["profilePath"], profile.display().to_string());
        assert_eq!(report["profileChanged"], true);
        assert_eq!(report["pathChanged"], true);
        assert_eq!(
            report["hermesBin"],
            install_root.join("venv").join("bin").display().to_string()
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn install_method_stamp_writes_git_for_update_compatibility() {
        let root = std::env::temp_dir().join(format!("hermes-install-method-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();

        let report = write_install_method_stamp(&root).unwrap();

        assert_eq!(std::fs::read_to_string(root.join(".install_method")).unwrap(), "git\n");
        assert_eq!(report["installMethod"], "git");
        assert_eq!(report["stampPath"], root.join(".install_method").display().to_string());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn python_venv_stage_plan_targets_install_root_and_prefers_managed_uv() {
        let root = std::env::temp_dir().join(format!("hermes-venv-plan-{}", std::process::id()));
        let hermes_home = root.join("home");
        let install_root = hermes_home.join("hermes-agent");
        let path_tools = root.join("tools");
        std::fs::create_dir_all(hermes_home.join("bin")).unwrap();
        std::fs::create_dir_all(&install_root).unwrap();
        std::fs::create_dir_all(&path_tools).unwrap();
        std::fs::write(hermes_home.join("bin").join("uv.exe"), b"managed uv").unwrap();
        std::fs::write(path_tools.join("uv.exe"), b"path uv").unwrap();

        let plan = python_venv_stage_plan(&install_root, &hermes_home, &path_tools, ".EXE")
            .unwrap();

        assert_eq!(plan.uv, hermes_home.join("bin").join("uv.exe"));
        assert_eq!(plan.venv, install_root.join("venv"));
        assert_eq!(plan.cwd, install_root);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn python_dependencies_stage_plan_requires_lock_and_targets_venv() {
        let root = std::env::temp_dir().join(format!("hermes-deps-plan-{}", std::process::id()));
        let hermes_home = root.join("home");
        let install_root = hermes_home.join("hermes-agent");
        let path_tools = root.join("tools");
        std::fs::create_dir_all(hermes_home.join("bin")).unwrap();
        std::fs::create_dir_all(&install_root).unwrap();
        std::fs::create_dir_all(&path_tools).unwrap();
        std::fs::write(hermes_home.join("bin").join("uv.exe"), b"managed uv").unwrap();

        let missing_lock =
            python_dependencies_stage_plan(&install_root, &hermes_home, &path_tools, ".EXE")
                .unwrap_err();
        assert!(missing_lock.to_string().contains("uv.lock"));

        std::fs::write(install_root.join("uv.lock"), b"lock").unwrap();
        let plan =
            python_dependencies_stage_plan(&install_root, &hermes_home, &path_tools, ".EXE")
                .unwrap();

        assert_eq!(plan.uv, hermes_home.join("bin").join("uv.exe"));
        assert_eq!(plan.cwd, install_root);
        assert_eq!(plan.venv, plan.cwd.join("venv"));
        assert_eq!(plan.python, venv_python_path(&plan.venv));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn repository_archive_spec_prefers_commit_and_keeps_branch() {
        let spec = repository_archive_spec(Some("abcdef123"), Some("main"));

        assert_eq!(spec.owner, "NousResearch");
        assert_eq!(spec.repo, "hermes-agent");
        assert_eq!(spec.commit.as_deref(), Some("abcdef123"));
        assert_eq!(spec.branch.as_deref(), Some("main"));
    }
}
