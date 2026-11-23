//! Public Hermes schedule forms and IANA wall-clock matching, without a daemon.
use anyhow::{Result, bail, ensure};
use chrono::{DateTime, Datelike, Duration, LocalResult, NaiveDate, NaiveDateTime, TimeZone, Utc};
use chrono_tz::Tz;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Schedule {
    Once { run_at: String, display: String },
    Interval { minutes: u64, display: String },
    Cron { expr: String, display: String },
}
impl Schedule {
    pub fn parse(input: &str, now: DateTime<Utc>, timezone: &str) -> Result<Self> {
        let input = input.trim();
        ensure!(
            !input.is_empty() && input.len() <= 512,
            "Schedule is required (maximum 512 bytes)"
        );
        let zone = zone(timezone)?;
        let lower = input.to_lowercase();
        let rest = lower.strip_prefix("every ").unwrap_or(&lower);
        if let Some(expr) = natural(rest) {
            CronFields::parse(&expr)?;
            return Ok(Self::Cron {
                expr,
                display: input.into(),
            });
        }
        if let Some(duration) = lower.strip_prefix("in ") {
            let minutes = duration_minutes(duration)?;
            let at = now
                .checked_add_signed(Duration::minutes(i64::try_from(minutes)?))
                .ok_or_else(|| anyhow::anyhow!("Duration exceeds supported dates"))?;
            return Ok(Self::Once {
                run_at: stamp(at),
                display: format!("once in {duration}"),
            });
        }
        let parts: Vec<_> = input.split_whitespace().collect();
        if matches!(parts.len(), 5 | 6) {
            CronFields::parse(input)?;
            return Ok(Self::Cron {
                expr: parts.join(" "),
                display: input.into(),
            });
        }
        if input.contains('T')
            || (input.len() >= 10 && input.as_bytes()[4] == b'-' && input.as_bytes()[7] == b'-')
        {
            let at = parse_timestamp(input, zone)?;
            return Ok(Self::Once {
                run_at: stamp(at),
                display: format!(
                    "once at {}",
                    at.with_timezone(&zone).format("%Y-%m-%d %H:%M")
                ),
            });
        }
        let minutes = duration_minutes(rest)?;
        Ok(Self::Interval {
            minutes,
            display: format!("every {minutes}m"),
        })
    }
    pub fn display(&self) -> &str {
        match self {
            Self::Once { display, .. }
            | Self::Interval { display, .. }
            | Self::Cron { display, .. } => display,
        }
    }
    pub fn next_after(
        &self,
        after: DateTime<Utc>,
        timezone: &str,
    ) -> Result<Option<DateTime<Utc>>> {
        match self {
            Self::Once { run_at, .. } => {
                let at = parse_timestamp(run_at, zone(timezone)?)?;
                Ok((at > after).then_some(at))
            }
            Self::Interval { minutes, .. } => {
                ensure!(
                    *minutes > 0 && *minutes <= 525_960_000,
                    "Interval must be positive and no greater than 1000 years"
                );
                Ok(after.checked_add_signed(Duration::minutes(i64::try_from(*minutes)?)))
            }
            Self::Cron { expr, .. } => CronFields::parse(expr)?
                .next(after, zone(timezone)?)
                .map(Some),
        }
    }
    pub fn is_once(&self) -> bool {
        matches!(self, Self::Once { .. })
    }
    pub fn interval_seconds(&self) -> Option<i64> {
        if let Self::Interval { minutes, .. } = self {
            i64::try_from(*minutes).ok()?.checked_mul(60)
        } else {
            None
        }
    }
}
pub fn zone(name: &str) -> Result<Tz> {
    name.parse()
        .map_err(|_| anyhow::anyhow!("Unknown IANA timezone {name:?}"))
}
pub fn stamp(at: DateTime<Utc>) -> String {
    at.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}
pub fn instant(value: &str) -> Result<DateTime<Utc>> {
    Ok(DateTime::parse_from_rfc3339(value)?.with_timezone(&Utc))
}
fn parse_timestamp(value: &str, zone: Tz) -> Result<DateTime<Utc>> {
    if let Ok(at) = DateTime::parse_from_rfc3339(value) {
        return Ok(at.with_timezone(&Utc));
    }
    let naive = [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
    ]
    .iter()
    .find_map(|format| NaiveDateTime::parse_from_str(value, format).ok())
    .or_else(|| {
        NaiveDate::parse_from_str(value, "%Y-%m-%d")
            .ok()?
            .and_hms_opt(0, 0, 0)
    })
    .ok_or_else(|| anyhow::anyhow!("Invalid ISO timestamp"))?;
    match zone.from_local_datetime(&naive) {
        LocalResult::Single(at) => Ok(at.with_timezone(&Utc)),
        LocalResult::Ambiguous(first, _) => Ok(first.with_timezone(&Utc)),
        LocalResult::None => {
            bail!("This local time does not exist because of a daylight-saving transition")
        }
    }
}
fn duration_minutes(input: &str) -> Result<u64> {
    let input = input.trim();
    let split = input
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(input.len());
    let count = if split == 0 {
        1
    } else {
        input[..split].parse::<u64>()?
    };
    let unit = input[split..].trim();
    let multiplier = match unit {
        "m" | "min" | "mins" | "minute" | "minutes" => 1,
        "h" | "hr" | "hrs" | "hour" | "hours" => 60,
        "d" | "day" | "days" => 1440,
        _ => bail!(
            "Use a duration such as '30m', 'every 2 hours', an ISO timestamp, or a 5-field cron expression"
        ),
    };
    let minutes = count
        .checked_mul(multiplier)
        .ok_or_else(|| anyhow::anyhow!("Duration exceeds supported range"))?;
    ensure!(
        (1..=525_960_000).contains(&minutes),
        "Duration must be positive and no greater than 1000 years"
    );
    Ok(minutes)
}
fn weekday(input: &str) -> Option<u32> {
    match input {
        "sun" | "sunday" => Some(0),
        "mon" | "monday" => Some(1),
        "tue" | "tues" | "tuesday" => Some(2),
        "wed" | "weds" | "wednesday" => Some(3),
        "thu" | "thur" | "thurs" | "thursday" => Some(4),
        "fri" | "friday" => Some(5),
        "sat" | "saturday" => Some(6),
        _ => None,
    }
}
fn clock(input: &str) -> Option<(u32, u32)> {
    let input = input.replace(' ', "");
    if matches!(input.as_str(), "noon" | "midday") {
        return Some((12, 0));
    }
    if input == "midnight" {
        return Some((0, 0));
    }
    let (number, meridiem) = if let Some(number) = input.strip_suffix("am") {
        (number, Some(false))
    } else if let Some(number) = input.strip_suffix("pm") {
        (number, Some(true))
    } else {
        (input.as_str(), None)
    };
    let (hour, minute) = number.split_once(':').unwrap_or((number, "0"));
    let mut hour: u32 = hour.parse().ok()?;
    let minute: u32 = minute.parse().ok()?;
    if let Some(pm) = meridiem {
        if !(1..=12).contains(&hour) {
            return None;
        }
        hour = hour % 12 + if pm { 12 } else { 0 };
    }
    (hour <= 23 && minute <= 59).then_some((hour, minute))
}
fn natural(input: &str) -> Option<String> {
    let normalized = input.replace(',', " ");
    let tokens: Vec<_> = normalized.split_whitespace().collect();
    let first = *tokens.first()?;
    let (dow, index) = match first {
        "day" | "daily" | "everyday" => ("*".to_owned(), 1),
        "weekday" | "weekdays" => ("1-5".to_owned(), 1),
        "weekend" | "weekends" => ("0,6".to_owned(), 1),
        _ => {
            let mut days = BTreeSet::new();
            let mut index = 0;
            for token in &tokens {
                if *token == "and" {
                    index += 1;
                    continue;
                }
                let Some(day) = weekday(token) else {
                    break;
                };
                days.insert(day);
                index += 1;
            }
            if days.is_empty() {
                return None;
            }
            (
                days.iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
                index,
            )
        }
    };
    let time = tokens.get(index..)?;
    let time = if time.first() == Some(&"at") {
        &time[1..]
    } else {
        time
    };
    let (hour, minute) = clock(&time.join(" "))?;
    Some(format!("{minute} {hour} * * {dow}"))
}
#[derive(Debug)]
struct CronFields {
    minute: BTreeSet<u32>,
    hour: BTreeSet<u32>,
    day: BTreeSet<u32>,
    month: BTreeSet<u32>,
    weekday: BTreeSet<u32>,
    second: BTreeSet<u32>,
    any_day: bool,
    any_weekday: bool,
}
impl CronFields {
    fn parse(expr: &str) -> Result<Self> {
        let parts: Vec<_> = expr.split_whitespace().collect();
        ensure!(
            matches!(parts.len(), 5 | 6),
            "Cron expressions need 5 fields (or seconds as the sixth field)"
        );
        let fields = Self {
            minute: field(parts[0], 0, 59, 0)?,
            hour: field(parts[1], 0, 23, 0)?,
            day: field(parts[2], 1, 31, 0)?,
            month: field(parts[3], 1, 12, 1)?,
            weekday: field(parts[4], 0, 7, 2)?
                .into_iter()
                .map(|day| day % 7)
                .collect(),
            second: if parts.len() == 6 {
                field(parts[5], 0, 59, 0)?
            } else {
                BTreeSet::from([0])
            },
            any_day: parts[2] == "*",
            any_weekday: parts[4] == "*",
        };
        Ok(fields)
    }
    fn next(&self, after: DateTime<Utc>, zone: Tz) -> Result<DateTime<Utc>> {
        let mut date = after.with_timezone(&zone).date_naive();
        for _ in 0..366 * 8 {
            let dom = self.day.contains(&date.day());
            let dow = self
                .weekday
                .contains(&date.weekday().num_days_from_sunday());
            let day_matches = if self.any_day || self.any_weekday {
                dom && dow
            } else {
                dom || dow
            };
            if self.month.contains(&date.month()) && day_matches {
                let mut earliest = None;
                for &hour in &self.hour {
                    for &minute in &self.minute {
                        for &second in &self.second {
                            let wall = date
                                .and_hms_opt(hour, minute, second)
                                .ok_or_else(|| anyhow::anyhow!("Invalid clock time"))?;
                            let candidates: Vec<_> = match zone.from_local_datetime(&wall) {
                                LocalResult::None => vec![],
                                LocalResult::Single(at) => vec![at],
                                LocalResult::Ambiguous(a, b) => vec![a, b],
                            };
                            for at in candidates {
                                let at = at.with_timezone(&Utc);
                                if at > after && earliest.is_none_or(|earliest| at < earliest) {
                                    earliest = Some(at);
                                }
                            }
                        }
                    }
                }
                if let Some(at) = earliest {
                    return Ok(at);
                }
            }
            date = date
                .succ_opt()
                .ok_or_else(|| anyhow::anyhow!("Cron date exceeds supported range"))?;
        }
        bail!("Cron expression has no occurrence within eight years")
    }
}
fn field(input: &str, min: u32, max: u32, names: u8) -> Result<BTreeSet<u32>> {
    let number = |raw: &str| -> Result<u32> {
        let lower = raw.to_lowercase();
        let named = if names == 1 {
            [
                "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
            ]
            .iter()
            .position(|name| *name == lower)
            .map(|index| index as u32 + 1)
        } else if names == 2 {
            weekday(&lower)
        } else {
            None
        };
        let value = if let Some(value) = named {
            value
        } else {
            raw.parse::<u32>()?
        };
        ensure!(
            (min..=max).contains(&value),
            "Cron value {raw:?} must be between {min} and {max}"
        );
        Ok(value)
    };
    let mut result = BTreeSet::new();
    for part in input.split(',') {
        let (range, step) = part
            .split_once('/')
            .map_or((part, Ok(1)), |(range, step)| (range, step.parse::<u32>()));
        let step = step?;
        ensure!(step > 0 && step <= max + 1, "Invalid cron step");
        let (start, end) = if range == "*" {
            (min, max)
        } else if let Some((start, end)) = range.split_once('-') {
            (number(start)?, number(end)?)
        } else {
            let start = number(range)?;
            (start, if part.contains('/') { max } else { start })
        };
        ensure!(start <= end, "Cron range must be ascending");
        result.extend((start..=end).step_by(step as usize));
    }
    ensure!(!result.is_empty(), "Empty cron field");
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn at(value: &str) -> DateTime<Utc> {
        instant(value).unwrap()
    }
    #[test]
    fn public_forms_distinguish_recurring_duration_from_explicit_one_shot() {
        let now = at("2026-01-01T00:00:00Z");
        for input in ["30m", "every 30 minutes", "every hour"] {
            assert!(matches!(
                Schedule::parse(input, now, "UTC").unwrap(),
                Schedule::Interval { .. }
            ));
        }
        assert!(
            matches!(Schedule::parse("in 30m",now,"UTC").unwrap(),Schedule::Once{run_at,..} if run_at=="2026-01-01T00:30:00Z")
        );
        assert!(
            matches!(Schedule::parse("every monday, wednesday at 9:30am",now,"UTC").unwrap(),Schedule::Cron{expr,..} if expr=="30 9 * * 1,3")
        );
        assert!(Schedule::parse("0 25 * * *", now, "UTC").is_err());
        assert!(Schedule::parse("0m", now, "UTC").is_err());
        assert!(Schedule::parse("every 2h", now, "invalid/zone").is_err());
    }
    #[test]
    fn pinned_timezone_and_dst_gap_and_fold_produce_future_instants() {
        let daily =
            Schedule::parse("0 9 * * *", at("2026-01-01T00:00:00Z"), "America/New_York").unwrap();
        assert_eq!(
            daily
                .next_after(at("2026-03-07T15:00:00Z"), "America/New_York")
                .unwrap()
                .unwrap(),
            at("2026-03-08T13:00:00Z")
        );
        let repeated =
            Schedule::parse("30 1 * * *", at("2026-01-01T00:00:00Z"), "America/New_York").unwrap();
        assert_eq!(
            repeated
                .next_after(at("2026-11-01T05:45:00Z"), "America/New_York")
                .unwrap()
                .unwrap(),
            at("2026-11-01T06:30:00Z")
        );
        let gap =
            Schedule::parse("30 2 * * *", at("2026-01-01T00:00:00Z"), "America/New_York").unwrap();
        assert_eq!(
            gap.next_after(at("2026-03-08T00:00:00Z"), "America/New_York")
                .unwrap()
                .unwrap(),
            at("2026-03-09T06:30:00Z")
        );
    }
    #[test]
    fn named_fields_seconds_and_day_or_rule_follow_cron_contract() {
        let schedule =
            Schedule::parse("0 9 1 JAN MON 30", at("2026-01-01T09:00:31Z"), "UTC").unwrap();
        assert_eq!(
            schedule
                .next_after(at("2026-01-01T09:00:31Z"), "UTC")
                .unwrap()
                .unwrap(),
            at("2026-01-05T09:00:30Z")
        );
        let iso = Schedule::parse(
            "2026-01-02T09:00",
            at("2026-01-01T00:00:00Z"),
            "Australia/Sydney",
        )
        .unwrap();
        assert!(matches!(iso,Schedule::Once{run_at,..} if run_at=="2026-01-01T22:00:00Z"));
    }
}
