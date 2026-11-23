//! Routines (app/routines): the redesigned Scheduled jobs page. The curated
//! templates are cron/blueprint_catalog.py's CATALOG (exported verbatim to
//! assets/routine-templates.json); a template's blanks fill its cron schedule
//! and prompt. Jobs group "problems first": Needs attention, Running,
//! Scheduled, Paused, Finished.
use serde::Deserialize;
use std::{collections::BTreeMap, sync::LazyLock};

#[derive(Debug, Clone, Deserialize)]
pub(super) struct Slot { pub name: String, #[serde(rename = "type")] pub kind: String, pub label: String, pub default: String, pub options: Vec<String>, pub help: String }
#[derive(Debug, Clone, Deserialize)]
pub(super) struct Template { #[cfg_attr(not(test), allow(dead_code))] pub key: String, pub title: String, pub description: String, pub category: String, pub schedule: String, pub prompt: String, pub slots: Vec<Slot>, pub skills: Vec<String> }

pub(super) static TEMPLATES: LazyLock<Vec<Template>> = LazyLock::new(|| serde_json::from_slice(include_bytes!("../assets/routine-templates.json")).unwrap_or_default());

/// The values a template starts with.
pub(super) fn defaults(t: &Template) -> BTreeMap<String, String> { t.slots.iter().map(|s| (s.name.clone(), s.default.clone())).collect() }

fn day_number(day: &str) -> Option<&'static str> {
    Some(match day.trim().to_lowercase().get(..3)? { "sun" => "0", "mon" => "1", "tue" => "2", "wed" => "3", "thu" => "4", "fri" => "5", "sat" => "6", _ => return None })
}
fn dow(value: &str) -> Option<String> {
    match value.trim().to_lowercase().as_str() {
        "everyday" | "every day" | "daily" => Some("*".into()),
        "weekdays" => Some("1-5".into()),
        "weekends" => Some("0,6".into()),
        other => other.split([',', ' ']).filter(|d| !d.is_empty()).map(day_number).collect::<Option<Vec<_>>>().map(|d| d.join(",")).filter(|d| !d.is_empty()),
    }
}

/// (name, cron schedule, prompt) for `t` filled with `values`, or why it can't be.
pub(super) fn fill(t: &Template, values: &BTreeMap<String, String>) -> Result<(String, String, String), String> {
    let mut schedule = t.schedule.clone();
    let mut prompt = t.prompt.clone();
    for slot in &t.slots {
        let value = values.get(&slot.name).map(|v| v.trim().to_owned()).filter(|v| !v.is_empty()).unwrap_or_else(|| slot.default.clone());
        match slot.kind.as_str() {
            "time" => {
                let (h, m) = value.split_once(':').ok_or(format!("{}: use HH:MM", slot.label))?;
                let (h, m): (u32, u32) = (h.trim().parse().map_err(|_| format!("{}: use HH:MM", slot.label))?, m.trim().parse().map_err(|_| format!("{}: use HH:MM", slot.label))?);
                if h > 23 || m > 59 { return Err(format!("{}: use a 24-hour time", slot.label)); }
                schedule = schedule.replace("{hour}", &h.to_string()).replace("{minute}", &m.to_string());
            }
            "weekdays" => { let d = dow(&value).ok_or(format!("{}: pick days like weekdays or mon,wed", slot.label))?; schedule = schedule.replace("{dow}", &d); }
            _ => {
                if slot.name == "day" && let Some(d) = day_number(&value) { schedule = schedule.replace("{dow}", d); }
                if slot.kind == "enum" && schedule.contains(&format!("{{{}}}", slot.name)) && !value.bytes().all(|b| b.is_ascii_digit()) {
                    return Err(format!("{}: choose a number", slot.label));
                }
                schedule = schedule.replace(&format!("{{{}}}", slot.name), &value);
            }
        }
        prompt = prompt.replace(&format!("{{{}}}", slot.name), &value);
    }
    if schedule.contains('{') { return Err("This template has a blank that isn't filled".into()); }
    if !t.skills.is_empty() { prompt.push_str(&format!("\n\nUse these skills if available: {}.", t.skills.join(", "))); }
    Ok((t.title.clone(), schedule, prompt))
}

/// The page's groups, problems first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Group { Attention, Running, Scheduled, Paused, Finished }
impl Group {
    pub fn label(self) -> &'static str { match self { Self::Attention => "NEEDS ATTENTION", Self::Running => "RUNNING", Self::Scheduled => "SCHEDULED", Self::Paused => "PAUSED", Self::Finished => "FINISHED" } }
}
/// (group, status chip) for a job.
pub(super) fn classify(job: &hermes_engine::cron::Job, running: bool, now: chrono::DateTime<chrono::Utc>) -> (Group, &'static str) {
    if running { return (Group::Running, "Running"); }
    let failed = job.last_status.as_deref().is_some_and(|s| matches!(s, "error" | "failed")) || job.state == "error";
    if failed { return (Group::Attention, "Failed"); }
    if job.state == "completed" { return (Group::Finished, "Finished"); }
    if job.state == "paused" || !job.enabled { return (Group::Paused, "Paused"); }
    let overdue = job.next_run_at.as_deref().and_then(|t| chrono::DateTime::parse_from_rfc3339(t).ok())
        .is_some_and(|t| now.signed_duration_since(t) > chrono::Duration::minutes(job.grace_minutes.unwrap_or(10) as i64));
    if overdue { return (Group::Attention, "Overdue"); }
    (Group::Scheduled, "On")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_template_fills_with_its_defaults_into_a_valid_schedule() {
        assert_eq!(TEMPLATES.len(), 16);
        for t in TEMPLATES.iter() {
            let (_, schedule, prompt) = fill(t, &defaults(t)).unwrap_or_else(|e| panic!("{}: {e}", t.key));
            assert!(hermes_engine::cron::Schedule::parse(&schedule, chrono::Utc::now(), "UTC").is_ok(), "{}: {schedule}", t.key);
            assert!(!prompt.contains('{'), "{}", t.key);
        }
        let brief = TEMPLATES.iter().find(|t| t.key == "morning-brief").unwrap();
        let mut v = defaults(brief); v.insert("time".into(), "07:15".into());
        assert_eq!(fill(brief, &v).unwrap().1, "15 7 * * *");
        let review = TEMPLATES.iter().find(|t| t.key == "weekly-review").unwrap();
        assert_eq!(fill(review, &defaults(review)).unwrap().1, "0 18 * * 0");
        let news = TEMPLATES.iter().find(|t| t.key == "news-digest").unwrap();
        let mut v = defaults(news); v.insert("recurrence".into(), "mon,thu".into());
        assert_eq!(fill(news, &v).unwrap().1, "0 18 * * 1,4");
        v.insert("time".into(), "25:00".into());
        assert!(fill(news, &v).is_err());
    }
}
