use clap::{Parser, Subcommand};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, Read};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "hermes-windows-sandbox-wrapper")]
#[command(about = "Hermes wrapper for Codex windows-sandbox-rs")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Execute a single foreground command request read from stdin as JSON.
    Exec,
    /// Report current setup readiness for the requested CODEX_HOME.
    Status,
    /// Run the upstream elevated setup flow for the requested CODEX_HOME.
    Setup,
}

#[derive(Debug, Deserialize)]
struct SandboxRequest {
    cwd: String,
    mode: String,
    network_enabled: bool,
    writable_roots: Vec<String>,
    codex_home: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ExecuteRequest {
    #[serde(flatten)]
    sandbox: SandboxRequest,
    command: String,
    timeout_secs: u64,
    command_mode: String,
    stdin_data: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
struct Diagnostics {
    shell: Option<&'static str>,
    command_mode: Option<String>,
    executable: Option<String>,
    setup_code: Option<String>,
    setup_complete: Option<bool>,
    setup_executable: Option<String>,
    upstream_error: Option<String>,
}

#[derive(Debug, Serialize)]
struct ExecuteResponse {
    stdout: String,
    stderr: String,
    exit_code: i32,
    timed_out: bool,
    error: Option<String>,
    error_type: Option<&'static str>,
    diagnostics: Diagnostics,
}

#[derive(Debug, Serialize)]
struct StatusResponse {
    setup_complete: bool,
    error: Option<String>,
    error_type: Option<&'static str>,
    diagnostics: Diagnostics,
}

#[derive(Debug, Serialize)]
struct SetupResponse {
    ok: bool,
    error: Option<String>,
    error_type: Option<&'static str>,
    diagnostics: Diagnostics,
}

struct WrapperFailure {
    message: String,
    error_type: &'static str,
    diagnostics: Diagnostics,
    exit_code: i32,
    timed_out: bool,
}

#[cfg(windows)]
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
enum SandboxPolicyJson {
    #[serde(rename = "read-only")]
    ReadOnly {
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        network_access: bool,
    },
    #[serde(rename = "workspace-write")]
    WorkspaceWrite {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        writable_roots: Vec<PathBuf>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        network_access: bool,
        #[serde(default)]
        exclude_tmpdir_env_var: bool,
    },
}

#[cfg(windows)]
struct ValidatedSandboxRequest {
    cwd: PathBuf,
    codex_home: PathBuf,
    policy_json: String,
    policy: codex_windows_sandbox::SandboxPolicy,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Exec => run_exec(),
        Commands::Status => run_status(),
        Commands::Setup => run_setup(),
    }
}

fn run_exec() {
    let request: ExecuteRequest = match read_json_from_stdin("wrapper request") {
        Ok(request) => request,
        Err(message) => {
            emit_json(&invalid_config_failure(message, Diagnostics::default()).to_execute_response());
            return;
        }
    };

    emit_json(&execute_request(request));
}

fn run_status() {
    let request: SandboxRequest = match read_json_from_stdin("status request") {
        Ok(request) => request,
        Err(message) => {
            emit_json(&invalid_config_failure(message, Diagnostics::default()).to_status_response(false));
            return;
        }
    };

    emit_json(&status_request(request));
}

fn run_setup() {
    let request: SandboxRequest = match read_json_from_stdin("setup request") {
        Ok(request) => request,
        Err(message) => {
            emit_json(&invalid_config_failure(message, Diagnostics::default()).to_setup_response());
            return;
        }
    };

    emit_json(&setup_request(request));
}

fn read_json_from_stdin<T: DeserializeOwned>(label: &str) -> Result<T, String> {
    let mut raw = String::new();
    io::stdin()
        .read_to_string(&mut raw)
        .map_err(|err| format!("failed to read {label}: {err}"))?;

    serde_json::from_str(&raw).map_err(|err| format!("invalid {label} JSON: {err}"))
}

#[cfg(not(windows))]
fn execute_request(request: ExecuteRequest) -> ExecuteResponse {
    unsupported_failure(
        "windows-sandbox wrapper is only supported on Windows hosts",
        Diagnostics {
            command_mode: Some(request.command_mode),
            ..Diagnostics::default()
        },
    )
    .to_execute_response()
}

#[cfg(windows)]
fn execute_request(request: ExecuteRequest) -> ExecuteResponse {
    let diagnostics = Diagnostics {
        command_mode: Some(request.command_mode.clone()),
        ..Diagnostics::default()
    };

    if request.command_mode != "foreground" {
        return unsupported_failure(
            "windows-sandbox only supports foreground execution in the first release",
            diagnostics,
        )
        .to_execute_response();
    }

    if request.stdin_data.is_some() {
        return unsupported_failure(
            "windows-sandbox wrapper does not support stdin piping in the first release",
            diagnostics,
        )
        .to_execute_response();
    }

    let (validated, diagnostics) = match prepare_request(&request.sandbox, diagnostics) {
        Ok(values) => values,
        Err(failure) => return failure.to_execute_response(),
    };

    let mut diagnostics = match ensure_setup_ready(&validated, diagnostics) {
        Ok(diagnostics) => diagnostics,
        Err(failure) => return failure.to_execute_response(),
    };

    let (command_vec, shell_name) = match build_command_invocation(&request.command) {
        Ok(parts) => parts,
        Err(err) => {
            return invalid_config_failure(err, diagnostics).to_execute_response();
        }
    };

    diagnostics.shell = Some(shell_name);
    diagnostics.executable = command_vec.first().cloned();

    let timeout_ms = request.timeout_secs.saturating_mul(1000);
    let exec_result = codex_windows_sandbox::run_windows_sandbox_capture(
        validated.policy_json.as_str(),
        &validated.cwd,
        &validated.codex_home,
        command_vec,
        &validated.cwd,
        HashMap::new(),
        Some(timeout_ms),
        true,
    );

    match exec_result {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            let timed_out = output.timed_out;
            ExecuteResponse {
                stdout,
                stderr,
                exit_code: output.exit_code,
                timed_out,
                error: if timed_out {
                    Some(format!(
                        "Command timed out after {} seconds",
                        request.timeout_secs
                    ))
                } else {
                    None
                },
                error_type: if timed_out { Some("timeout") } else { None },
                diagnostics,
            }
        }
        Err(err) => classify_execution_error(&err, diagnostics).to_execute_response(),
    }
}

#[cfg(not(windows))]
fn status_request(_request: SandboxRequest) -> StatusResponse {
    unsupported_failure(
        "windows-sandbox wrapper is only supported on Windows hosts",
        Diagnostics::default(),
    )
    .to_status_response(false)
}

#[cfg(windows)]
fn status_request(request: SandboxRequest) -> StatusResponse {
    let (validated, mut diagnostics) = match prepare_request(&request, Diagnostics::default()) {
        Ok(values) => values,
        Err(failure) => return failure.to_status_response(false),
    };

    let setup_complete = codex_windows_sandbox::sandbox_setup_is_complete(&validated.codex_home);
    diagnostics.setup_complete = Some(setup_complete);

    StatusResponse {
        setup_complete,
        error: if setup_complete {
            None
        } else {
            Some("Windows sandbox setup is required before execution.".to_string())
        },
        error_type: if setup_complete {
            None
        } else {
            Some("setup_required")
        },
        diagnostics,
    }
}

#[cfg(not(windows))]
fn setup_request(_request: SandboxRequest) -> SetupResponse {
    unsupported_failure(
        "windows-sandbox wrapper is only supported on Windows hosts",
        Diagnostics::default(),
    )
    .to_setup_response()
}

#[cfg(windows)]
fn setup_request(request: SandboxRequest) -> SetupResponse {
    let (validated, diagnostics) = match prepare_request(&request, Diagnostics::default()) {
        Ok(values) => values,
        Err(failure) => return failure.to_setup_response(),
    };

    let mut diagnostics = diagnostics;
    if let Err(err) = codex_windows_sandbox::run_elevated_setup(
        &validated.policy,
        &validated.cwd,
        &validated.cwd,
        &HashMap::new(),
        &validated.codex_home,
        None,
        None,
    ) {
        return classify_setup_failure(&err, diagnostics, "Windows sandbox setup failed")
            .to_setup_response();
    }

    let setup_complete = codex_windows_sandbox::sandbox_setup_is_complete(&validated.codex_home);
    diagnostics.setup_complete = Some(setup_complete);
    if !setup_complete {
        return setup_required_failure(
            "Windows sandbox setup completed without a valid setup marker.",
            diagnostics,
        )
        .to_setup_response();
    }

    SetupResponse {
        ok: true,
        error: None,
        error_type: None,
        diagnostics,
    }
}

#[cfg(windows)]
fn prepare_request(
    request: &SandboxRequest,
    diagnostics: Diagnostics,
) -> Result<(ValidatedSandboxRequest, Diagnostics), WrapperFailure> {
    let mut diagnostics = diagnostics;
    diagnostics.setup_executable = Some(resolve_setup_executable().display().to_string());

    let cwd = if request.cwd.trim().is_empty() {
        std::env::current_dir().map_err(|err| {
            internal_error_failure(
                format!("failed to resolve current working directory: {err}"),
                diagnostics.clone(),
            )
        })?
    } else {
        PathBuf::from(&request.cwd)
    };

    if !cwd.is_absolute() {
        return Err(invalid_config_failure(
            format!("windows-sandbox cwd must be absolute: {}", cwd.display()),
            diagnostics,
        ));
    }

    let writable_roots = validate_writable_roots(&request.writable_roots, diagnostics.clone())?;
    let (policy_json, policy) = build_policy(request, &writable_roots, diagnostics.clone())?;

    let codex_home = request
        .codex_home
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(default_codex_home);

    Ok((
        ValidatedSandboxRequest {
            cwd,
            codex_home,
            policy_json,
            policy,
        },
        diagnostics,
    ))
}

#[cfg(windows)]
fn validate_writable_roots(
    writable_roots: &[String],
    diagnostics: Diagnostics,
) -> Result<Vec<PathBuf>, WrapperFailure> {
    let mut normalized: Vec<PathBuf> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for raw_root in writable_roots {
        let trimmed = raw_root.trim();
        if trimmed.is_empty() {
            continue;
        }

        let path = PathBuf::from(trimmed);
        if !path.is_absolute() {
            return Err(invalid_config_failure(
                format!("windows-sandbox writable roots must be absolute: {trimmed}"),
                diagnostics,
            ));
        }

        let normalized_key = path.to_string_lossy().to_ascii_lowercase();
        if seen.insert(normalized_key) {
            normalized.push(path);
        }
    }

    Ok(normalized)
}

#[cfg(windows)]
fn build_policy(
    request: &SandboxRequest,
    writable_roots: &[PathBuf],
    diagnostics: Diagnostics,
) -> Result<(String, codex_windows_sandbox::SandboxPolicy), WrapperFailure> {
    let policy_json = match request.mode.as_str() {
        "workspace-write" => serde_json::to_string(&SandboxPolicyJson::WorkspaceWrite {
            writable_roots: writable_roots.to_vec(),
            network_access: request.network_enabled,
            exclude_tmpdir_env_var: true,
        })
        .map_err(|err| {
            internal_error_failure(
                format!("failed to encode workspace-write policy JSON: {err}"),
                diagnostics.clone(),
            )
        })?,
        "read-only" => {
            if !writable_roots.is_empty() {
                return Err(invalid_config_failure(
                    "windows-sandbox read-only mode does not support writable_roots",
                    diagnostics,
                ));
            }
            serde_json::to_string(&SandboxPolicyJson::ReadOnly {
                network_access: request.network_enabled,
            })
            .map_err(|err| {
                internal_error_failure(
                    format!("failed to encode read-only policy JSON: {err}"),
                    diagnostics.clone(),
                )
            })?
        }
        other => {
            return Err(invalid_config_failure(
                format!("unsupported windows-sandbox mode: {other}"),
                diagnostics,
            ));
        }
    };

    let policy = codex_windows_sandbox::parse_policy(policy_json.as_str()).map_err(|err| {
        invalid_config_failure(
            format!("failed to parse sandbox policy JSON {policy_json:?}: {err}"),
            diagnostics,
        )
    })?;

    Ok((policy_json, policy))
}

#[cfg(windows)]
fn ensure_setup_ready(
    validated: &ValidatedSandboxRequest,
    mut diagnostics: Diagnostics,
) -> Result<Diagnostics, WrapperFailure> {
    let setup_complete = codex_windows_sandbox::sandbox_setup_is_complete(&validated.codex_home);
    diagnostics.setup_complete = Some(setup_complete);
    if !setup_complete {
        return Err(setup_required_failure(
            "Windows sandbox setup is required before execution.",
            diagnostics,
        ));
    }

    if let Err(err) = codex_windows_sandbox::run_setup_refresh(
        &validated.policy,
        &validated.cwd,
        &validated.cwd,
        &HashMap::new(),
        &validated.codex_home,
    ) {
        return Err(classify_setup_failure(
            &err,
            diagnostics,
            "Windows sandbox setup refresh failed",
        ));
    }

    Ok(diagnostics)
}

#[cfg(windows)]
fn classify_setup_failure(
    err: &anyhow::Error,
    mut diagnostics: Diagnostics,
    prefix: &str,
) -> WrapperFailure {
    if let Some(failure) = codex_windows_sandbox::extract_setup_failure(err) {
        diagnostics.setup_code = Some(failure.code.as_str().to_string());
        diagnostics.upstream_error = Some(failure.message.clone());
        return setup_required_failure(format!("{prefix}: {failure}"), diagnostics);
    }

    diagnostics.upstream_error = Some(err.to_string());
    setup_required_failure(format!("{prefix}: {err}"), diagnostics)
}

#[cfg(windows)]
fn classify_execution_error(err: &anyhow::Error, mut diagnostics: Diagnostics) -> WrapperFailure {
    if let Some(failure) = codex_windows_sandbox::extract_setup_failure(err) {
        diagnostics.setup_code = Some(failure.code.as_str().to_string());
        diagnostics.upstream_error = Some(failure.message.clone());
        return setup_required_failure(format!("Windows sandbox setup failed: {failure}"), diagnostics);
    }

    let message = err.to_string();
    diagnostics.upstream_error = Some(message.clone());
    let lower = message.to_ascii_lowercase();

    if lower.contains("timed out") || lower.contains("timeout") {
        return timeout_failure(message, diagnostics);
    }

    execution_failure(message, diagnostics)
}

#[cfg(windows)]
fn build_command_invocation(command: &str) -> Result<(Vec<String>, &'static str), String> {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return Err("command cannot be empty".to_string());
    }

    if let Some(rest) = strip_cmd_prefix(trimmed) {
        let exe = resolve_command_path(&["cmd.exe", "cmd"], r"C:\Windows\System32\cmd.exe");
        return Ok((
            vec![
                exe.display().to_string(),
                "/d".to_string(),
                "/s".to_string(),
                "/c".to_string(),
                rest.to_string(),
            ],
            "cmd",
        ));
    }

    let exe = resolve_command_path(
        &["pwsh.exe", "pwsh", "powershell.exe", "powershell"],
        r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
    );

    Ok((
        vec![
            exe.display().to_string(),
            "-NoLogo".to_string(),
            "-NoProfile".to_string(),
            "-NonInteractive".to_string(),
            "-Command".to_string(),
            trimmed.to_string(),
        ],
        "powershell",
    ))
}

#[cfg(windows)]
fn strip_cmd_prefix(command: &str) -> Option<&str> {
    for prefix in ["cmd /c ", "cmd.exe /c "] {
        if command.len() >= prefix.len() && command[..prefix.len()].eq_ignore_ascii_case(prefix) {
            return Some(command[prefix.len()..].trim_start());
        }
    }
    None
}

#[cfg(windows)]
fn resolve_command_path(candidates: &[&str], fallback: &str) -> PathBuf {
    for candidate in candidates {
        if let Ok(found) = which::which(candidate) {
            return found;
        }
    }
    PathBuf::from(fallback)
}

#[cfg(windows)]
fn resolve_setup_executable() -> PathBuf {
    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(dir) = current_exe.parent() {
            let candidate = dir.join("codex-windows-sandbox-setup.exe");
            if candidate.exists() {
                return candidate;
            }
        }
    }

    which::which("codex-windows-sandbox-setup.exe")
        .unwrap_or_else(|_| PathBuf::from("codex-windows-sandbox-setup.exe"))
}

#[cfg(windows)]
fn default_codex_home() -> PathBuf {
    if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(local_app_data)
            .join("Hermes")
            .join("windows-sandbox-codex-home");
    }
    std::env::temp_dir().join("hermes-windows-sandbox-codex-home")
}

#[cfg(not(windows))]
fn default_codex_home() -> PathBuf {
    std::env::temp_dir().join("hermes-windows-sandbox-codex-home")
}

fn invalid_config_failure(message: impl Into<String>, diagnostics: Diagnostics) -> WrapperFailure {
    WrapperFailure {
        message: message.into(),
        error_type: "invalid_config",
        diagnostics,
        exit_code: -1,
        timed_out: false,
    }
}

fn unsupported_failure(message: impl Into<String>, diagnostics: Diagnostics) -> WrapperFailure {
    WrapperFailure {
        message: message.into(),
        error_type: "unsupported_feature",
        diagnostics,
        exit_code: -1,
        timed_out: false,
    }
}

fn internal_error_failure(message: impl Into<String>, diagnostics: Diagnostics) -> WrapperFailure {
    WrapperFailure {
        message: message.into(),
        error_type: "internal_error",
        diagnostics,
        exit_code: -1,
        timed_out: false,
    }
}

fn setup_required_failure(message: impl Into<String>, diagnostics: Diagnostics) -> WrapperFailure {
    WrapperFailure {
        message: message.into(),
        error_type: "setup_required",
        diagnostics,
        exit_code: -1,
        timed_out: false,
    }
}

fn execution_failure(message: impl Into<String>, diagnostics: Diagnostics) -> WrapperFailure {
    WrapperFailure {
        message: message.into(),
        error_type: "execution_failed",
        diagnostics,
        exit_code: -1,
        timed_out: false,
    }
}

fn timeout_failure(message: impl Into<String>, diagnostics: Diagnostics) -> WrapperFailure {
    WrapperFailure {
        message: message.into(),
        error_type: "timeout",
        diagnostics,
        exit_code: 124,
        timed_out: true,
    }
}

impl WrapperFailure {
    fn to_execute_response(self) -> ExecuteResponse {
        ExecuteResponse {
            stdout: String::new(),
            stderr: self.message.clone(),
            exit_code: self.exit_code,
            timed_out: self.timed_out,
            error: Some(self.message),
            error_type: Some(self.error_type),
            diagnostics: self.diagnostics,
        }
    }

    fn to_status_response(self, setup_complete: bool) -> StatusResponse {
        StatusResponse {
            setup_complete,
            error: Some(self.message),
            error_type: Some(self.error_type),
            diagnostics: self.diagnostics,
        }
    }

    fn to_setup_response(self) -> SetupResponse {
        SetupResponse {
            ok: false,
            error: Some(self.message),
            error_type: Some(self.error_type),
            diagnostics: self.diagnostics,
        }
    }
}

fn emit_json<T: Serialize>(response: &T) {
    match serde_json::to_string(response) {
        Ok(encoded) => println!("{encoded}"),
        Err(err) => {
            let fallback = format!(
                "{{\"error\":\"failed to encode wrapper response: {err}\",\"error_type\":\"internal_error\"}}"
            );
            println!("{fallback}");
        }
    }
}



