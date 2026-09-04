//! Honor reduced motion (Vercel: `prefers-reduced-motion`).
//!
//! `HERMES_TUI_REDUCED_MOTION` or `PREFERS_REDUCED_MOTION` = 1/true/on.

const SPIN: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const ASCII: &[&str] = &["|", "/", "-", "\\"];
const EMOJI: &[&str] = &["⚕", "🌀", "🤔", "✨", "🍵", "🔮"];
const KAOMOJI: &[&str] = &[
    "(｡•́︿•̀｡)",
    "(◔_◔)",
    "(¬‿¬)",
    "(´･_･`)",
    "◉_◉",
    "(°ロ°)",
    "(⊙_⊙)",
    "(¬_¬)",
    "ಠ_ಠ",
];

pub fn reduced_motion() -> bool {
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

pub fn spinner(frame: u64) -> &'static str {
    spinner_for(crate::state::IndicatorStyle::Unicode, frame)
}

pub fn spinner_for(style: crate::state::IndicatorStyle, frame: u64) -> &'static str {
    if reduced_motion() {
        return match style {
            crate::state::IndicatorStyle::Emoji => "⚕",
            crate::state::IndicatorStyle::Kaomoji => "(´･_･`)",
            crate::state::IndicatorStyle::Ascii => "+",
            crate::state::IndicatorStyle::Unicode => "●",
        };
    }
    let tick = self::frame(frame);
    match style {
        crate::state::IndicatorStyle::Unicode => SPIN[(tick / 2 % SPIN.len() as u64) as usize],
        crate::state::IndicatorStyle::Ascii => ASCII[(tick / 2 % ASCII.len() as u64) as usize],
        crate::state::IndicatorStyle::Emoji => EMOJI[(tick / 6 % EMOJI.len() as u64) as usize],
        crate::state::IndicatorStyle::Kaomoji => {
            KAOMOJI[(tick / 8 % KAOMOJI.len() as u64) as usize]
        }
    }
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

    fn matches_truthy(v: &str) -> bool {
        matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
    }
}
