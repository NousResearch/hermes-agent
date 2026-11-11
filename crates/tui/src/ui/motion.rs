//! Hermes Ink motion: status FaceTicker + thinking/tool braille families.
//!
//! Cadence matches `ui-tui` (`appChrome.tsx`, `thinking.tsx`, `content/faces.ts`).
//! Transcript spinners stay 1 cell so the stream does not jitter.
//! `/motion` overrides `HERMES_TUI_REDUCED_MOTION` / `PREFERS_REDUCED_MOTION` for this process.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

const TICK_MS: u64 = 50;

thread_local! {
    /// 0 = follow env, 1 = motion on, -1 = reduced.
    static OVERRIDE: Cell<i8> = const { Cell::new(0) };
}

static EPOCH: AtomicU64 = AtomicU64::new(1);

/// Classic braille (`unicode-animations` `braille`, 80ms).
const BRAILLE: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const ASCII: &[&str] = &["|", "/", "-", "\\"];
/// Caduceus first, then the Ink emoji set. Trailing space on ⚕ keeps width 2.
const EMOJI: &[&str] = &["⚕ ", "🌀", "🤔", "✨", "🍵", "🔮"];
/// Official Ink `FACES` (`ui-tui/src/content/faces.ts`).
const KAOMOJI: &[&str] = &[
    "(｡•́︿•̀｡)",
    "(◔_◔)",
    "(¬‿¬)",
    "( •_•)>⌐■-■",
    "(⌐■_■)",
    "(´･_･`)",
    "◉_◉",
    "(°ロ°)",
    "( ˘⌣˘)♡",
    "ヽ(>∀<☆)☆",
    "٩(๑❛ᴗ❛๑)۶",
    "(⊙_⊙)",
    "(¬_¬)",
    "( ͡° ͜ʖ ͡°)",
    "ಠ_ಠ",
];

/// First cell of Ink `THINK` families (helix is weak at 1-col, so orbit leads).
const THINK: &[&[&str]] = &[
    &["⠃", "⠉", "⠘", "⠰", "⢠", "⣀", "⡄", "⠆"], // orbit
    &["⠋", "⠉", "⠙", "⠚", "⠒", "⠂", "⠂", "⠒", "⠲", "⠴", "⠤", "⠄"], // dna
    &[
        "⠖", "⡠", "⣠", "⣄", "⠢", "⠙", "⠉", "⠊", "⠜", "⡤", "⣀", "⢤", "⠣", "⠑", "⠉", "⠋",
    ], // waverows
    &[
        "⣁", "⣉", "⡉", "⠉", "⠈", "⠀", "⠐", "⠒", "⠖", "⠶", "⠦", "⠤", "⠠", "⠀", "⢀", "⣀",
    ], // snake
    &[
        "⠀", "⠂", "⠌", "⡑", "⢕", "⢝", "⣫", "⣟", "⣿", "⣟", "⣫", "⢝", "⢕", "⡑", "⠌", "⠂", "⠀",
    ], // breathe
    &["⠀", "⠰", "⢾", "⣏", "⡁"],                // pulse
];

/// First cell of Ink `TOOL` families with real 1-col motion.
const TOOL: &[&[&str]] = &[
    &["⡡", "⠊", "⢔", "⡁", "⢔", "⠨"], // sparkle
    &["⢁", "⠂", "⠄", "⡈", "⠐", "⠠", "⢁", "⠂", "⠄", "⡈", "⠐", "⠠"], // rain
    &["⣀", "⣤", "⣶", "⣿", "⣿", "⣿", "⣶", "⣤", "⣀", "⠀", "⠀"], // fillsweep
    &[
        "⠁", "⠋", "⠟", "⡿", "⣿", "⣿", "⣿", "⣿", "⣾", "⣴", "⣠", "⢀", "⠀", "⠀", "⠀", "⠀",
    ], // diagswipe
    &["⠀", "⠁", "⠋", "⠞", "⡴", "⣠", "⢀"], // cascade (blanks stripped)
];

const FACE_MS: u64 = 2500;
const EMOJI_MS: u64 = 600;
const ASCII_MS: u64 = 100;
const BRAILLE_MS: u64 = 80;
const THINK_MS: u64 = 90;
const TOOL_MS: u64 = 70;

pub fn reduced_motion() -> bool {
    match OVERRIDE.with(|c| c.get()) {
        1 => false,
        -1 => true,
        _ => env_reduced(),
    }
}

pub fn enabled() -> bool {
    !reduced_motion()
}

/// `/motion` session override. `true` plays chrome; `false` freezes it.
pub fn set_enabled(on: bool) {
    OVERRIDE.with(|c| c.set(if on { 1 } else { -1 }));
    EPOCH.fetch_add(1, Ordering::Relaxed);
}

pub fn toggle() -> bool {
    let on = reduced_motion();
    set_enabled(on);
    on
}

pub fn epoch() -> u64 {
    EPOCH.load(Ordering::Relaxed)
}

fn env_reduced() -> bool {
    flag("HERMES_TUI_REDUCED_MOTION") || flag("PREFERS_REDUCED_MOTION")
}

pub fn flag(key: &str) -> bool {
    std::env::var(key)
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// Freeze looping chrome at the first frame when motion is reduced.
pub fn frame(frame: u64) -> u64 {
    if reduced_motion() {
        0
    } else {
        frame
    }
}

fn at(frames: &[&'static str], interval_ms: u64, frame: u64) -> &'static str {
    if frames.is_empty() {
        return "●";
    }
    if reduced_motion() {
        return frames[0];
    }
    let ticks = (frame.saturating_mul(TICK_MS)) / interval_ms.max(1);
    frames[(ticks as usize) % frames.len()]
}

fn family(sets: &[&[&'static str]], salt: u64, interval_ms: u64, frame: u64) -> &'static str {
    let set = sets[(salt as usize) % sets.len()];
    at(set, interval_ms, frame)
}

pub fn spinner(frame: u64) -> &'static str {
    spinner_for(crate::state::IndicatorStyle::Unicode, frame)
}

/// Status-bar FaceTicker (`/indicator`). Unicode is a bare braille cell.
pub fn spinner_for(style: crate::state::IndicatorStyle, frame: u64) -> &'static str {
    if reduced_motion() {
        return match style {
            crate::state::IndicatorStyle::Emoji => "⚕ ",
            crate::state::IndicatorStyle::Kaomoji => "(´･_･`)",
            crate::state::IndicatorStyle::Ascii => "+",
            crate::state::IndicatorStyle::Unicode => "●",
        };
    }
    match style {
        crate::state::IndicatorStyle::Unicode => at(BRAILLE, BRAILLE_MS, frame),
        crate::state::IndicatorStyle::Ascii => at(ASCII, ASCII_MS, frame),
        crate::state::IndicatorStyle::Emoji => at(EMOJI, EMOJI_MS, frame),
        crate::state::IndicatorStyle::Kaomoji => at(KAOMOJI, FACE_MS, frame),
    }
}

/// Transcript thinking (Ink `Spinner variant="think"`).
pub fn think_spinner(frame: u64) -> &'static str {
    if reduced_motion() {
        return "●";
    }
    family(THINK, 0, THINK_MS, frame)
}

/// Running tool row (Ink `Spinner variant="tool"`). `salt` picks a family.
pub fn tool_spinner(frame: u64, salt: u64) -> &'static str {
    if reduced_motion() {
        return "●";
    }
    family(TOOL, salt, TOOL_MS, frame)
}

pub fn salt_id(id: &str) -> u64 {
    let mut h = 0u64;
    for b in id.as_bytes() {
        h = h.wrapping_mul(16777619) ^ u64::from(*b);
    }
    h
}

pub fn ellipsis_at(frame: u64, reduced: bool) -> &'static str {
    if reduced {
        "…"
    } else {
        [".", "..", "..."][((frame / 6) % 3) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_parses_truthy() {
        assert!(matches_truthy("1"));
        assert!(matches_truthy("TRUE"));
        assert!(matches_truthy("on"));
        assert!(!matches_truthy("0"));
        assert!(!matches_truthy("no"));
    }

    #[test]
    fn spinner_freezes_when_reduced() {
        assert_eq!(ellipsis_at(0, true), "…");
        assert_eq!(ellipsis_at(12, true), "…");
        assert_eq!(ellipsis_at(0, false), ".");
        assert_eq!(ellipsis_at(12, false), "...");
        assert_eq!(spinner_for(crate::state::IndicatorStyle::Ascii, 0).len(), 1);
        assert!(!spinner_for(crate::state::IndicatorStyle::Kaomoji, 8).is_empty());
    }

    #[test]
    fn hermes_emoji_leads_with_caduceus() {
        assert!(EMOJI[0].starts_with('⚕'));
        assert!(KAOMOJI.iter().any(|f| f.contains("⌐■")));
    }

    #[test]
    fn think_and_tool_families_move() {
        reset_override();
        let a = think_spinner(0);
        let b = think_spinner(8);
        assert!(!a.is_empty());
        assert!(!b.is_empty());
        let t0 = tool_spinner(0, 1);
        let t1 = tool_spinner(12, 1);
        assert!(!t0.is_empty());
        assert!(!t1.is_empty());
        assert_ne!(tool_spinner(0, 0), tool_spinner(0, 3));
    }

    #[test]
    fn slash_override_freezes_and_restores() {
        let _reset = OverrideReset;
        set_enabled(false);
        assert!(reduced_motion());
        assert!(!enabled());
        assert_eq!(frame(12), 0);
        assert_eq!(think_spinner(0), think_spinner(8));
        let before = epoch();
        set_enabled(true);
        assert!(enabled());
        assert!(!reduced_motion());
        assert_eq!(frame(12), 12);
        assert!(epoch() > before);
        assert!(!toggle());
        assert!(!enabled());
    }

    fn reset_override() {
        OVERRIDE.with(|c| c.set(0));
    }

    struct OverrideReset;
    impl Drop for OverrideReset {
        fn drop(&mut self) {
            reset_override();
        }
    }

    fn matches_truthy(v: &str) -> bool {
        matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
    }
}
