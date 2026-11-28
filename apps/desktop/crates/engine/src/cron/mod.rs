//! Profile-scoped, in-app scheduled prompts, ported from public Hermes cron.
//! No daemon or system crontab is installed. Every run has an independent Session;
//! unattended tools are denied by the caller and foreground calls require approval.
//! Script/monitor execution and external-platform delivery are not implemented.
mod runner;
mod schedule;
mod store;
pub use runner::execute;
pub use schedule::{Schedule, instant, stamp};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};
pub use store::Store;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Repeat {
    pub times: Option<u32>,
    #[serde(default)]
    pub completed: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub name: String,
    pub prompt: String,
    pub schedule: Schedule,
    #[serde(default)]
    pub repeat: Repeat,
    #[serde(default = "yes")]
    pub enabled: bool,
    #[serde(default = "scheduled")]
    pub state: String,
    #[serde(default)]
    pub timezone: Option<String>,
    #[serde(default)]
    pub grace_minutes: Option<u32>,
    #[serde(default)]
    pub catch_up: Option<bool>,
    #[serde(default = "local")]
    pub deliver: String,
    #[serde(default)]
    pub workdir: Option<PathBuf>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub next_run_at: Option<String>,
    #[serde(default)]
    pub last_run_at: Option<String>,
    #[serde(default)]
    pub last_status: Option<String>,
    #[serde(default)]
    pub last_error: Option<String>,
    #[serde(default)]
    pub paused_at: Option<String>,
    #[serde(default)]
    pub definition_revision: u64,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}
fn yes() -> bool {
    true
}
fn scheduled() -> String {
    "scheduled".into()
}
fn local() -> String {
    "local".into()
}
impl Job {
    pub fn title(&self) -> &str {
        if self.name.trim().is_empty() {
            &self.prompt
        } else {
            &self.name
        }
    }
    pub fn timezone(&self) -> &str {
        self.timezone.as_deref().unwrap_or("UTC")
    }
    pub fn draft(&self) -> Draft {
        let schedule = match &self.schedule {
            Schedule::Once { run_at, .. } => run_at.clone(),
            Schedule::Interval { minutes, .. } => format!("every {minutes}m"),
            Schedule::Cron { expr, .. } => expr.clone(),
        };
        Draft {
            name: self.name.clone(),
            prompt: self.prompt.clone(),
            schedule,
            timezone: self.timezone().into(),
            workdir: self.workdir.clone(),
            model: self.model.clone(),
            provider: self.provider.clone(),
            repeat: self.repeat.times,
            deliver: self.deliver.clone(),
            catch_up: self.catch_up,
            grace_minutes: self.grace_minutes,
            paused: !self.enabled,
        }
    }
}
#[derive(Debug, Clone)]
pub struct Draft {
    pub name: String,
    pub prompt: String,
    pub schedule: String,
    pub timezone: String,
    pub workdir: Option<PathBuf>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub repeat: Option<u32>,
    pub deliver: String,
    pub catch_up: Option<bool>,
    pub grace_minutes: Option<u32>,
    pub paused: bool,
}
impl Default for Draft {
    fn default() -> Self {
        Self {
            name: String::new(),
            prompt: String::new(),
            schedule: "0 9 * * *".into(),
            timezone: "UTC".into(),
            workdir: None,
            model: None,
            provider: None,
            repeat: None,
            deliver: "local".into(),
            catch_up: None,
            grace_minutes: None,
            paused: false,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Claimed,
    Running,
    Completed,
    Failed,
    Cancelled,
    Interrupted,
    Skipped,
}
impl RunStatus {
    pub fn live(&self) -> bool {
        matches!(self, Self::Claimed | Self::Running)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub id: String,
    pub job_id: String,
    pub status: RunStatus,
    pub claimed_at: String,
    pub started_at: Option<String>,
    pub finished_at: Option<String>,
    pub scheduled_instant: Option<String>,
    pub manual: bool,
    pub repeat_counted: bool,
    pub owner: String,
    pub definition_revision: u64,
    pub session_dir: PathBuf,
    pub output: String,
    pub error: Option<String>,
    #[serde(default)]
    pub cancel_requested: bool,
}
#[derive(Debug, Clone)]
pub struct Claim {
    pub job: Job,
    pub run: Run,
}
#[derive(Debug, Clone)]
pub enum Outcome {
    Completed { output: String },
    Failed { error: String },
    Cancelled { reason: String },
}
