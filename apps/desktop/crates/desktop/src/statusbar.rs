//! Port of shell/statusbar-controls.tsx, hooks/use-statusbar-items.tsx,
//! approval-mode-menu.tsx and system-resources-statusbar.tsx. Visibility
//! defaults match STATUSBAR_HIDDEN_BY_DEFAULT; changing visibility never
//! changes execution or approval policy.
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, path::Path, time::Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Menu { Backend, Approval, Customize }

#[derive(Debug, Clone)]
pub(super) enum Action {
    Backend, Approval, Customize, Close, ToggleVisible, ToggleItem(&'static str), Reset,
    SetApproval(&'static str), Memory(Option<(u64, u64)>), Tick,
}

#[derive(Serialize, Deserialize)]
#[serde(default)]
pub(super) struct Preferences { pub visible: bool, pub hidden: BTreeSet<String> }
impl Default for Preferences {
    fn default() -> Self {
        Self { visible: true, hidden: ["agents", "cache-hit-rate", "context-usage", "cron", "running-timer", "session-timer", "system-resources", "terminal", "tokens-per-second", "webhooks"].into_iter().map(str::to_owned).collect() }
    }
}

/// Every toggleable item with its customize-menu label, left side first.
pub(super) const ITEMS: [(&str, &str); 12] = [
    ("gateway-health", "Backend"), ("workspace-cwd", "Workspace"), ("agents", "Agents"), ("cron", "Scheduled jobs"),
    ("running-timer", "Running timer"), ("context-usage", "Context usage"), ("cache-hit-rate", "Cache hit rate"),
    ("tokens-per-second", "Tokens per second"), ("session-timer", "Session timer"), ("system-resources", "System resources"),
    ("approval-mode", "Approval mode"), ("terminal", "Terminal"),
];

/// `(value, label, description)` from i18n `shell.approvalMode`.
pub(super) const APPROVAL_MODES: [(&str, &str, &str); 3] = [
    ("manual", "Manual", "Ask before actions that require approval"),
    ("smart", "Smart", "Automatically assess actions and ask when needed"),
    ("off", "Off", "Run without approval prompts"),
];

#[derive(Default)]
pub(super) struct State {
    pub preferences: Preferences, pub menu: Option<Menu>,
    /// System memory `(used, total)` in bytes, sampled while the item is shown.
    pub memory: Option<(u64, u64)>,
    pub turn_started: Option<Instant>, stream_started: Option<Instant>, last_text: Option<Instant>, stream_bytes: usize,
    /// Output rate of the latest reply, estimated from streamed text (about 4 bytes per token).
    pub tokens_per_second: Option<f64>,
    pub opened: Option<Instant>,
}
impl State {
    pub fn load(root: &Path) -> Self {
        Self { preferences: std::fs::read(root.join("statusbar-preferences.json")).ok().and_then(|bytes| serde_json::from_slice(&bytes).ok()).unwrap_or_default(),
            opened: Some(Instant::now()), ..Default::default() }
    }
    pub fn shows(&self, id: &str) -> bool { !self.preferences.hidden.contains(id) }
    pub fn turn_started(&mut self) { self.turn_started = Some(Instant::now()); self.stream_started = None; self.last_text = None; self.stream_bytes = 0; }
    pub fn streamed(&mut self, bytes: usize) {
        let now = Instant::now();
        let start = *self.stream_started.get_or_insert(now);
        self.stream_bytes += bytes; self.last_text = Some(now);
        let secs = now.duration_since(start).as_secs_f64();
        if secs >= 0.25 { self.tokens_per_second = Some(self.stream_bytes as f64 / 4.0 / secs); }
    }
    /// Settles the rate for the finished reply. A reply that streamed in one
    /// burst is timed from when the turn began, so short replies still get a rate.
    pub fn turn_finished(&mut self) {
        if self.stream_bytes > 0 && let (Some(first), Some(last)) = (self.stream_started, self.last_text) {
            let streaming = last.duration_since(first).as_secs_f64();
            let secs = if streaming >= 0.25 { streaming } else { self.turn_started.map_or(streaming, |t| last.duration_since(t).as_secs_f64()) };
            if secs > 0.0 { self.tokens_per_second = Some(self.stream_bytes as f64 / 4.0 / secs); }
        }
        self.turn_started = None; self.stream_started = None; self.last_text = None;
    }
    pub fn update(&mut self, action: Action, root: Option<&Path>) -> Result<(), String> {
        let menu = match action {
            Action::Backend => Some(Menu::Backend), Action::Approval => Some(Menu::Approval), Action::Customize => Some(Menu::Customize),
            Action::Close | Action::SetApproval(_) => { self.menu = None; return Ok(()); }
            Action::Memory(sample) => { self.memory = sample; return Ok(()); }
            Action::Tick => return Ok(()),
            Action::ToggleVisible => { self.preferences.visible = !self.preferences.visible; self.menu = None; None }
            Action::ToggleItem(id) => { if !self.preferences.hidden.remove(id) { self.preferences.hidden.insert(id.into()); } None }
            Action::Reset => { self.preferences.hidden = Preferences::default().hidden; None }
        };
        if let Some(menu) = menu { self.menu = if self.menu == Some(menu) { None } else { Some(menu) }; return Ok(()); }
        if let Some(root) = root {
            let path = root.join("statusbar-preferences.json"); let temp = root.join("statusbar-preferences.tmp");
            std::fs::write(&temp, serde_json::to_vec_pretty(&self.preferences).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            std::fs::rename(temp, path).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

pub(super) fn build_label() -> Option<(String, String)> {
    if let Some(version) = option_env!("HERMES_DESKTOP_VERSION").filter(|version| !version.trim().is_empty()) {
        return Some((format!("v{}", version.trim_start_matches('v')), format!("Hermes Desktop {version}")));
    }
    option_env!("HERMES_DESKTOP_SOURCE_REVISION")
        .filter(|revision| revision.len() >= 7 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .map(|revision| (format!("git.{}", &revision[..7]), format!("commit {revision}")))
}

/// `m:ss` / `h:mm:ss`, as LiveDuration renders it.
pub(super) fn duration(since: Instant) -> String {
    let secs = since.elapsed().as_secs();
    if secs >= 3600 { format!("{}:{:02}:{:02}", secs / 3600, secs / 60 % 60, secs % 60) } else { format!("{}:{:02}", secs / 60, secs % 60) }
}

/// Whole gigabytes, as the original RAM item shows them ("12G").
pub(super) fn gigabytes(bytes: u64) -> String { format!("{}G", (bytes as f64 / 1_073_741_824.0).round() as u64) }

/// Compact token counts: 950, 12.4k, 1.2M.
pub(super) fn tokens(n: u64) -> String {
    if n >= 1_000_000 { format!("{:.1}M", n as f64 / 1e6) } else if n >= 1000 { format!("{:.1}k", n as f64 / 1e3) } else { n.to_string() }
}

/// Used and total physical memory, read from the OS directly.
pub(super) fn sample_memory() -> Option<(u64, u64)> {
    #[cfg(target_os = "linux")]
    {
        let info = std::fs::read_to_string("/proc/meminfo").ok()?;
        let field = |name: &str| info.lines().find(|l| l.starts_with(name))?.split_whitespace().nth(1)?.parse::<u64>().ok().map(|kb| kb * 1024);
        let (total, available) = (field("MemTotal:")?, field("MemAvailable:")?);
        return Some((total.saturating_sub(available), total));
    }
    #[cfg(target_os = "macos")]
    {
        let run = |cmd: &str, args: &[&str]| std::process::Command::new(cmd).args(args).output().ok().and_then(|o| String::from_utf8(o.stdout).ok());
        let total = run("/usr/sbin/sysctl", &["-n", "hw.memsize"])?.trim().parse::<u64>().ok()?;
        let stats = run("/usr/bin/vm_stat", &[])?;
        let page = stats.lines().next().and_then(|l| l.split("page size of ").nth(1)).and_then(|s| s.split_whitespace().next()).and_then(|s| s.parse::<u64>().ok()).unwrap_or(16384);
        let pages = |name: &str| stats.lines().find(|l| l.starts_with(name)).and_then(|l| l.split(':').nth(1)).and_then(|v| v.trim().trim_end_matches('.').parse::<u64>().ok()).unwrap_or(0);
        // Activity Monitor "Memory Used" = app memory (anonymous minus purgeable) + wired + compressed.
        let used = (pages("Anonymous pages").saturating_sub(pages("Pages purgeable")) + pages("Pages wired down") + pages("Pages occupied by compressor")) * page;
        return Some((used.min(total), total));
    }
    #[allow(unreachable_code)]
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn metrics_and_formatting() {
        if let Some((used, total)) = sample_memory() { assert!(total > 0 && used <= total); }
        assert_eq!(gigabytes(17_179_869_184), "16G");
        assert_eq!(tokens(12_400), "12.4k");
        let mut state = State::default();
        state.turn_started(); std::thread::sleep(std::time::Duration::from_millis(20)); state.streamed(400);
        assert!(state.turn_started.is_some());
        state.turn_finished(); assert!(state.turn_started.is_none());
        assert!(state.tokens_per_second.is_some_and(|tps| tps > 0.0), "a one-burst reply still gets a rate");
    }
}
