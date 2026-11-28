//! Attach to the profile gateway the way Ink does.
//!
//! `ui-tui/scripts/gateway_bootstrap.py` prints one JSON object:
//! `{url, protocols, profile_id, instance_id}`. The ratatui client dials that
//! socket. It does not spawn `python -m tui_gateway.entry`.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

use crate::config::LaunchConfig;

const BOOTSTRAP_BUDGET: Duration = Duration::from_secs(45);

#[derive(Debug, Clone, Deserialize)]
pub struct GatewayAttach {
    pub url: String,
    pub protocols: Vec<String>,
}

pub async fn attach_gateway(cfg: &LaunchConfig) -> Result<GatewayAttach> {
    let script = cfg.src_root.join("ui-tui/scripts/gateway_bootstrap.py");
    if !script.is_file() {
        bail!(
            "missing {} — the native TUI attaches through Ink's gateway bootstrap",
            script.display()
        );
    }
    let pythonpath = match std::env::var("PYTHONPATH") {
        Ok(existing) if !existing.is_empty() => {
            format!("{}{}{}", cfg.src_root.display(), path_sep(), existing)
        }
        _ => cfg.src_root.display().to_string(),
    };
    let mut cmd = Command::new(&cfg.python);
    cmd.arg(&script)
        .arg("--start")
        .current_dir(&cfg.src_root)
        .env("HERMES_HOME", &cfg.hermes_home)
        .env("HERMES_PYTHON_SRC_ROOT", &cfg.src_root)
        .env("PYTHONPATH", pythonpath)
        .env("PYTHONUNBUFFERED", "1")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let child = cmd
        .spawn()
        .with_context(|| format!("failed to run {}", script.display()))?;
    let finished = timeout(BOOTSTRAP_BUDGET, child.wait_with_output())
        .await
        .context("gateway bootstrap timed out")?
        .context("gateway bootstrap failed")?;
    if !finished.status.success() {
        let err = String::from_utf8_lossy(&finished.stderr);
        let err = err.trim();
        bail!(
            "{}",
            if err.is_empty() {
                "gateway bootstrap exited without a ticket".to_string()
            } else {
                err.to_string()
            }
        );
    }
    let stdout = String::from_utf8_lossy(&finished.stdout);
    let line = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(stdout.trim());
    let value: Value = serde_json::from_str(line)
        .with_context(|| format!("gateway bootstrap did not print JSON ({line})"))?;
    let url = value
        .get("url")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .context("gateway bootstrap omitted url")?
        .to_string();
    let protocols = value
        .get("protocols")
        .and_then(|v| v.as_array())
        .map(|rows| {
            rows.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if protocols.is_empty() {
        bail!("gateway bootstrap omitted websocket protocols");
    }
    Ok(GatewayAttach { url, protocols })
}

fn path_sep() -> &'static str {
    if cfg!(windows) {
        ";"
    } else {
        ":"
    }
}
