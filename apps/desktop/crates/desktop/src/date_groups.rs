//! Port of lib/session-date-groups.ts + lib/time.ts calendarBucket: the sidebar's
//! date dividers. The newest run of sessions has no divider; older rows are
//! bucketed Earlier today → Yesterday → Earlier this week → Last week →
//! Earlier this month → Month → Month Year, with a 4 AM day rollover.
use chrono::{DateTime, Datelike, Duration, Local, NaiveDate, TimeZone};

const MINUTE: i64 = 60_000;
const HOUR: i64 = 60 * MINUTE;
const TARGET_HEAD_SESSIONS: f64 = 5.0;
const MIN_RUN_BREAK_MS: i64 = 30 * MINUTE;
const MAX_RUN_GAP_MS: i64 = 8 * HOUR;
const DAY_ROLLOVER_HOURS: i64 = 4;

fn nominal_day(ms: i64) -> NaiveDate {
    Local.timestamp_millis_opt(ms - DAY_ROLLOVER_HOURS * HOUR).single().map_or_else(|| DateTime::UNIX_EPOCH.date_naive(), |d| d.date_naive())
}

/// `(key, label)` for a timestamp. Monday is the first day of the week.
pub(super) fn bucket(ms: i64, now_ms: i64) -> (String, String) {
    let day = nominal_day(ms); let today = nominal_day(now_ms);
    let diff = (today - day).num_days();
    if diff <= 0 { return ("today".into(), "Earlier today".into()); }
    if diff == 1 { return ("yesterday".into(), "Yesterday".into()); }
    let week_start = today - Duration::days(today.weekday().num_days_from_monday() as i64);
    if day >= week_start { return ("this-week".into(), "Earlier this week".into()); }
    if day >= week_start - Duration::days(7) { return ("last-week".into(), "Last week".into()); }
    if day.year() == today.year() && day.month() == today.month() { return ("this-month".into(), "Earlier this month".into()); }
    let month = day.format("%B").to_string();
    if day.year() == today.year() { (format!("m-{}-{}", day.year(), day.month()), month) }
    else { (format!("my-{}-{}", day.year(), day.month()), format!("{month} {}", day.year())) }
}

/// Everything at or above this time is the unlabelled head run.
fn head_cutoff(times: &[i64], now_ms: i64) -> i64 {
    let mut best: Option<usize> = None; let mut best_score = f64::INFINITY; let mut run_ended = false;
    for i in 1..times.len() {
        let gap = times[i - 1] - times[i];
        let ends = gap > MAX_RUN_GAP_MS;
        if gap >= MIN_RUN_BREAK_MS || ends {
            let score = (i as f64 / TARGET_HEAD_SESSIONS).ln().abs();
            if score < best_score { best_score = score; best = Some(i); run_ended = ends; }
        }
        if ends { break; }
    }
    let Some(index) = best else { return i64::MIN };
    if run_ended && bucket(times[0], now_ms).0 == bucket(times[index], now_ms).0 { return i64::MAX; }
    times[index - 1]
}

/// For recency-sorted (newest first) timestamps, the divider label to draw
/// before each row, if any.
pub(super) fn dividers(times: &[i64], now_ms: i64) -> Vec<Option<String>> {
    let cutoff = head_cutoff(times, now_ms);
    let mut out = Vec::with_capacity(times.len());
    let mut last: Option<String> = None; let mut emitted = std::collections::HashSet::new();
    for (index, &ms) in times.iter().enumerate() {
        if ms >= cutoff { out.push(None); last = Some("__recent__".into()); continue; }
        let (key, label) = bucket(ms, now_ms);
        let mut divider = None;
        if last.as_deref() != Some(key.as_str()) {
            if index > 0 && emitted.insert(key.clone()) { divider = Some(label); } else { emitted.insert(key.clone()); }
            last = Some(key);
        }
        out.push(divider);
    }
    out
}

pub(super) fn millis(rfc3339: &str) -> i64 {
    DateTime::parse_from_rfc3339(rfc3339).map_or(0, |d| d.timestamp_millis())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn at(y: i32, m: u32, d: u32, h: u32) -> i64 { Local.with_ymd_and_hms(y, m, d, h, 0, 0).unwrap().timestamp_millis() }
    #[test]
    fn head_run_is_unlabelled_then_calendar_buckets() {
        let now = at(2026, 11, 21, 15); // Saturday
        let times = [now - 5 * MINUTE, now - 10 * MINUTE, now - 20 * MINUTE, at(2026, 11, 21, 9), at(2026, 11, 20, 18), at(2026, 11, 18, 12), at(2026, 11, 11, 12), at(2026, 11, 2, 12), at(2026, 9, 2, 12), at(2025, 9, 2, 12)];
        let labels = dividers(&times, now);
        assert_eq!(labels, vec![None, None, None, None, Some("Yesterday".into()), Some("Earlier this week".into()),
            Some("Last week".into()), Some("Earlier this month".into()), Some("September".into()), Some("September 2025".into())]);
    }
    #[test]
    fn small_hours_belong_to_the_previous_day_and_one_cluster_gets_no_divider() {
        let now = at(2026, 11, 21, 15);
        assert_eq!(bucket(at(2026, 11, 21, 2), now).1, "Yesterday");
        assert_eq!(dividers(&[now, now - MINUTE, now - 2 * MINUTE], now), vec![None, None, None]);
    }
}
