use ratatui::style::Color;

/// Live palette. `Theme::bg_base()` etc. read the active skin each frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Palette {
    pub id: &'static str,
    pub label: &'static str,
    pub blurb: &'static str,
    pub aliases: &'static [&'static str],
    pub bg_base: Color,
    pub bg_surface: Color,
    pub bg_header: Color,
    pub bg_popup: Color,
    pub bg_highlight: Color,
    pub border_subtle: Color,
    pub border_focus: Color,
    pub brand_gold: Color,
    pub brand_orange: Color,
    pub accent_green: Color,
    pub accent_red: Color,
    pub accent_yellow: Color,
    pub text_primary: Color,
    pub text_secondary: Color,
    pub text_muted: Color,
    pub text_dim: Color,
}

const fn rgb(r: u8, g: u8, b: u8) -> Color {
    Color::Rgb(r, g, b)
}

impl Palette {
    pub const fn gold() -> Self {
        Self {
            id: "gold",
            label: "Caduceus gold",
            blurb: "Antique gold on warm near-black — this TUI's original metal",
            aliases: &[],
            bg_base: rgb(10, 10, 11),
            bg_surface: rgb(16, 16, 18),
            bg_header: rgb(12, 12, 13),
            bg_popup: rgb(18, 18, 20),
            bg_highlight: rgb(32, 26, 16),
            border_subtle: rgb(42, 38, 32),
            border_focus: rgb(168, 132, 72),
            brand_gold: rgb(212, 175, 106),
            brand_orange: rgb(168, 132, 72),
            accent_green: rgb(140, 160, 110),
            accent_red: rgb(180, 90, 70),
            accent_yellow: rgb(180, 150, 80),
            text_primary: rgb(232, 226, 214),
            text_secondary: rgb(140, 132, 118),
            text_muted: rgb(98, 92, 82),
            text_dim: rgb(64, 60, 54),
        }
    }

    /// Website / dashboard "Hermes Teal": deep teal canvas, cream type, gold ⚕.
    pub const fn hermes() -> Self {
        Self {
            id: "hermes",
            label: "Hermes site",
            blurb: "hermes-agent.nousresearch.com — dark teal, cream, caduceus gold",
            aliases: &[],
            bg_base: rgb(6, 16, 18),
            bg_surface: rgb(10, 24, 26),
            bg_header: rgb(8, 20, 22),
            bg_popup: rgb(12, 28, 30),
            bg_highlight: rgb(18, 48, 50),
            border_subtle: rgb(28, 58, 60),
            border_focus: rgb(94, 196, 188),
            brand_gold: rgb(212, 175, 106),
            brand_orange: rgb(62, 168, 164),
            accent_green: rgb(94, 196, 164),
            accent_red: rgb(196, 92, 86),
            accent_yellow: rgb(212, 175, 106),
            text_primary: rgb(242, 235, 220),
            text_secondary: rgb(156, 186, 184),
            text_muted: rgb(92, 122, 122),
            text_dim: rgb(56, 78, 80),
        }
    }

    pub const fn midnight() -> Self {
        Self {
            id: "midnight",
            label: "Midnight",
            blurb: "Dashboard midnight — blue-violet, Inter-on-terminal",
            aliases: &[],
            bg_base: rgb(8, 10, 22),
            bg_surface: rgb(14, 16, 32),
            bg_header: rgb(10, 12, 26),
            bg_popup: rgb(16, 18, 36),
            bg_highlight: rgb(32, 36, 72),
            border_subtle: rgb(48, 52, 88),
            border_focus: rgb(140, 150, 220),
            brand_gold: rgb(176, 184, 232),
            brand_orange: rgb(120, 130, 200),
            accent_green: rgb(110, 180, 150),
            accent_red: rgb(200, 100, 120),
            accent_yellow: rgb(180, 170, 120),
            text_primary: rgb(220, 224, 240),
            text_secondary: rgb(140, 148, 180),
            text_muted: rgb(96, 102, 132),
            text_dim: rgb(64, 68, 92),
        }
    }

    pub const fn ember() -> Self {
        Self {
            id: "ember",
            label: "Ember",
            blurb: "Dashboard ember — warm crimson and bronze",
            aliases: &["ares"],
            bg_base: rgb(14, 8, 8),
            bg_surface: rgb(22, 12, 12),
            bg_header: rgb(18, 10, 10),
            bg_popup: rgb(26, 14, 14),
            bg_highlight: rgb(48, 22, 18),
            border_subtle: rgb(72, 36, 28),
            border_focus: rgb(196, 92, 64),
            brand_gold: rgb(212, 140, 88),
            brand_orange: rgb(168, 72, 48),
            accent_green: rgb(140, 150, 100),
            accent_red: rgb(200, 70, 60),
            accent_yellow: rgb(212, 150, 80),
            text_primary: rgb(236, 220, 208),
            text_secondary: rgb(168, 124, 108),
            text_muted: rgb(112, 80, 70),
            text_dim: rgb(72, 48, 42),
        }
    }

    pub const fn mono() -> Self {
        Self {
            id: "mono",
            label: "Mono",
            blurb: "Official skin — grayscale, screen-recording clean",
            aliases: &[],
            bg_base: rgb(12, 12, 12),
            bg_surface: rgb(18, 18, 18),
            bg_header: rgb(14, 14, 14),
            bg_popup: rgb(20, 20, 20),
            bg_highlight: rgb(36, 36, 36),
            border_subtle: rgb(64, 64, 64),
            border_focus: rgb(160, 160, 160),
            brand_gold: rgb(201, 209, 217),
            brand_orange: rgb(140, 140, 140),
            accent_green: rgb(160, 160, 160),
            accent_red: rgb(180, 180, 180),
            accent_yellow: rgb(180, 180, 180),
            text_primary: rgb(220, 220, 220),
            text_secondary: rgb(150, 150, 150),
            text_muted: rgb(100, 100, 100),
            text_dim: rgb(70, 70, 70),
        }
    }

    pub const fn daylight() -> Self {
        Self {
            id: "daylight",
            label: "Daylight",
            blurb: "Official skin — light canvas, slate text, cool blue",
            aliases: &[],
            bg_base: rgb(244, 246, 248),
            bg_surface: rgb(255, 255, 255),
            bg_header: rgb(232, 236, 240),
            bg_popup: rgb(255, 255, 255),
            bg_highlight: rgb(220, 230, 245),
            border_subtle: rgb(190, 198, 208),
            border_focus: rgb(65, 105, 225),
            brand_gold: rgb(40, 70, 160),
            brand_orange: rgb(70, 90, 160),
            accent_green: rgb(40, 130, 80),
            accent_red: rgb(180, 50, 50),
            accent_yellow: rgb(160, 110, 20),
            text_primary: rgb(28, 32, 40),
            text_secondary: rgb(70, 78, 90),
            text_muted: rgb(110, 118, 128),
            text_dim: rgb(150, 156, 164),
        }
    }

    /// Omarchy `themes/tokyo-night/colors.toml`.
    pub const fn tokyo() -> Self {
        Self {
            id: "tokyo",
            label: "Tokyo Night",
            blurb: "Omarchy — #1a1b26 canvas, #7aa2f7 accent",
            aliases: &["tokyo-night", "tokyonight"],
            bg_base: rgb(26, 27, 38),
            bg_surface: rgb(36, 40, 59),
            bg_header: rgb(22, 22, 30),
            bg_popup: rgb(31, 35, 53),
            bg_highlight: rgb(41, 46, 66),
            border_subtle: rgb(68, 75, 106),
            border_focus: rgb(122, 162, 247),
            brand_gold: rgb(122, 162, 247),
            brand_orange: rgb(255, 158, 100),
            accent_green: rgb(158, 206, 106),
            accent_red: rgb(247, 118, 142),
            accent_yellow: rgb(224, 175, 104),
            text_primary: rgb(192, 202, 245),
            text_secondary: rgb(169, 177, 214),
            text_muted: rgb(86, 95, 137),
            text_dim: rgb(65, 72, 104),
        }
    }

    /// Omarchy `themes/catppuccin/colors.toml` (mocha).
    pub const fn mocha() -> Self {
        Self {
            id: "mocha",
            label: "Catppuccin",
            blurb: "Omarchy mocha — #1e1e2e, latte-blue #89b4fa",
            aliases: &["catppuccin", "catppuccin-mocha"],
            bg_base: rgb(30, 30, 46),
            bg_surface: rgb(49, 50, 68),
            bg_header: rgb(24, 24, 37),
            bg_popup: rgb(49, 50, 68),
            bg_highlight: rgb(69, 71, 90),
            border_subtle: rgb(88, 91, 112),
            border_focus: rgb(137, 180, 250),
            brand_gold: rgb(137, 180, 250),
            brand_orange: rgb(250, 179, 135),
            accent_green: rgb(166, 227, 161),
            accent_red: rgb(243, 139, 168),
            accent_yellow: rgb(249, 226, 175),
            text_primary: rgb(205, 214, 244),
            text_secondary: rgb(186, 194, 222),
            text_muted: rgb(108, 112, 134),
            text_dim: rgb(69, 71, 90),
        }
    }

    /// Omarchy `themes/catppuccin-latte/colors.toml`.
    pub const fn latte() -> Self {
        Self {
            id: "latte",
            label: "Latte",
            blurb: "Omarchy Catppuccin Latte — paper #eff1f5, blue #1e66f5",
            aliases: &["catppuccin-latte"],
            bg_base: rgb(239, 241, 245),
            bg_surface: rgb(255, 255, 255),
            bg_header: rgb(230, 233, 239),
            bg_popup: rgb(255, 255, 255),
            bg_highlight: rgb(204, 208, 218),
            border_subtle: rgb(172, 176, 190),
            border_focus: rgb(30, 102, 245),
            brand_gold: rgb(30, 102, 245),
            brand_orange: rgb(254, 100, 11),
            accent_green: rgb(64, 160, 43),
            accent_red: rgb(210, 15, 57),
            accent_yellow: rgb(223, 142, 29),
            text_primary: rgb(76, 79, 105),
            text_secondary: rgb(92, 95, 119),
            text_muted: rgb(108, 111, 133),
            text_dim: rgb(156, 160, 176),
        }
    }

    /// Omarchy `themes/gruvbox/colors.toml` (material medium dark).
    pub const fn gruvbox() -> Self {
        Self {
            id: "gruvbox",
            label: "Gruvbox",
            blurb: "Omarchy — #282828, aqua #7daea3",
            aliases: &["gruvbox-material"],
            bg_base: rgb(40, 40, 40),
            bg_surface: rgb(50, 48, 47),
            bg_header: rgb(29, 32, 33),
            bg_popup: rgb(60, 56, 54),
            bg_highlight: rgb(80, 73, 69),
            border_subtle: rgb(80, 73, 69),
            border_focus: rgb(125, 174, 163),
            brand_gold: rgb(125, 174, 163),
            brand_orange: rgb(214, 93, 14),
            accent_green: rgb(169, 182, 101),
            accent_red: rgb(234, 105, 98),
            accent_yellow: rgb(216, 166, 87),
            text_primary: rgb(212, 190, 152),
            text_secondary: rgb(189, 174, 147),
            text_muted: rgb(146, 131, 116),
            text_dim: rgb(102, 92, 84),
        }
    }

    /// Omarchy `themes/nord/colors.toml`.
    pub const fn nord() -> Self {
        Self {
            id: "nord",
            label: "Nord",
            blurb: "Omarchy — polar night #2e3440, frost #81a1c1",
            aliases: &[],
            bg_base: rgb(46, 52, 64),
            bg_surface: rgb(59, 66, 82),
            bg_header: rgb(46, 52, 64),
            bg_popup: rgb(67, 76, 94),
            bg_highlight: rgb(76, 86, 106),
            border_subtle: rgb(76, 86, 106),
            border_focus: rgb(129, 161, 193),
            brand_gold: rgb(129, 161, 193),
            brand_orange: rgb(208, 135, 112),
            accent_green: rgb(163, 190, 140),
            accent_red: rgb(191, 97, 106),
            accent_yellow: rgb(235, 203, 139),
            text_primary: rgb(236, 239, 244),
            text_secondary: rgb(216, 222, 233),
            text_muted: rgb(143, 188, 187),
            text_dim: rgb(76, 86, 106),
        }
    }

    /// Omarchy `themes/everforest/colors.toml`.
    pub const fn forest() -> Self {
        Self {
            id: "forest",
            label: "Everforest",
            blurb: "Omarchy — #2d353b, aqua #7fbbb3",
            aliases: &["everforest"],
            bg_base: rgb(45, 53, 59),
            bg_surface: rgb(52, 63, 68),
            bg_header: rgb(39, 46, 51),
            bg_popup: rgb(61, 72, 77),
            bg_highlight: rgb(71, 82, 88),
            border_subtle: rgb(71, 82, 88),
            border_focus: rgb(127, 187, 179),
            brand_gold: rgb(127, 187, 179),
            brand_orange: rgb(230, 126, 128),
            accent_green: rgb(167, 192, 128),
            accent_red: rgb(230, 126, 128),
            accent_yellow: rgb(219, 188, 127),
            text_primary: rgb(211, 198, 170),
            text_secondary: rgb(167, 192, 128),
            text_muted: rgb(133, 146, 137),
            text_dim: rgb(71, 82, 88),
        }
    }

    /// Omarchy `themes/kanagawa/colors.toml`.
    pub const fn kanagawa() -> Self {
        Self {
            id: "kanagawa",
            label: "Kanagawa",
            blurb: "Omarchy — sumi ink #1f1f28, wave blue #7e9cd8",
            aliases: &["kanagawa-wave", "wave"],
            bg_base: rgb(31, 31, 40),
            bg_surface: rgb(42, 42, 55),
            bg_header: rgb(22, 22, 29),
            bg_popup: rgb(42, 42, 55),
            bg_highlight: rgb(45, 79, 103),
            border_subtle: rgb(114, 113, 105),
            border_focus: rgb(126, 156, 216),
            brand_gold: rgb(126, 156, 216),
            brand_orange: rgb(255, 160, 102),
            accent_green: rgb(118, 148, 106),
            accent_red: rgb(195, 64, 67),
            accent_yellow: rgb(192, 163, 110),
            text_primary: rgb(220, 215, 186),
            text_secondary: rgb(200, 192, 147),
            text_muted: rgb(114, 113, 105),
            text_dim: rgb(84, 84, 89),
        }
    }

    /// Omarchy `themes/rose-pine/colors.toml` is dawn (light).
    pub const fn dawn() -> Self {
        Self {
            id: "dawn",
            label: "Rosé Pine Dawn",
            blurb: "Omarchy rose-pine — paper #faf4ed, foam #56949f",
            aliases: &["rose-pine", "rose-pine-dawn", "rosepine"],
            bg_base: rgb(250, 244, 237),
            bg_surface: rgb(255, 250, 243),
            bg_header: rgb(242, 233, 225),
            bg_popup: rgb(255, 250, 243),
            bg_highlight: rgb(223, 218, 217),
            border_subtle: rgb(152, 147, 165),
            border_focus: rgb(86, 148, 159),
            brand_gold: rgb(86, 148, 159),
            brand_orange: rgb(234, 157, 52),
            accent_green: rgb(40, 105, 131),
            accent_red: rgb(180, 99, 122),
            accent_yellow: rgb(234, 157, 52),
            text_primary: rgb(87, 82, 121),
            text_secondary: rgb(121, 117, 147),
            text_muted: rgb(152, 147, 165),
            text_dim: rgb(182, 177, 191),
        }
    }

    /// Rosé Pine main (dark). Omarchy ships dawn; this is the moon people expect in a TUI.
    pub const fn rose() -> Self {
        Self {
            id: "rose",
            label: "Rosé Pine",
            blurb: "Rosé Pine moon — #191724, iris #c4a7e7, gold #f6c177",
            aliases: &["rose-pine-moon", "moon", "rosepine-moon"],
            bg_base: rgb(25, 23, 36),
            bg_surface: rgb(31, 29, 46),
            bg_header: rgb(25, 23, 36),
            bg_popup: rgb(38, 35, 58),
            bg_highlight: rgb(64, 61, 82),
            border_subtle: rgb(82, 79, 103),
            border_focus: rgb(196, 167, 231),
            brand_gold: rgb(196, 167, 231),
            brand_orange: rgb(246, 193, 119),
            accent_green: rgb(156, 207, 216),
            accent_red: rgb(235, 111, 146),
            accent_yellow: rgb(246, 193, 119),
            text_primary: rgb(224, 222, 244),
            text_secondary: rgb(144, 140, 170),
            text_muted: rgb(110, 106, 134),
            text_dim: rgb(82, 79, 103),
        }
    }

    /// Omarchy `themes/matte-black/colors.toml`.
    pub const fn matte() -> Self {
        Self {
            id: "matte",
            label: "Matte Black",
            blurb: "Omarchy — #121212, amber #e68e0d",
            aliases: &["matte-black", "matteblack"],
            bg_base: rgb(18, 18, 18),
            bg_surface: rgb(26, 26, 26),
            bg_header: rgb(18, 18, 18),
            bg_popup: rgb(33, 33, 33),
            bg_highlight: rgb(81, 81, 81),
            border_subtle: rgb(81, 81, 81),
            border_focus: rgb(230, 142, 13),
            brand_gold: rgb(230, 142, 13),
            brand_orange: rgb(245, 158, 11),
            accent_green: rgb(255, 193, 7),
            accent_red: rgb(211, 95, 95),
            accent_yellow: rgb(255, 193, 7),
            text_primary: rgb(234, 234, 234),
            text_secondary: rgb(190, 190, 190),
            text_muted: rgb(138, 138, 141),
            text_dim: rgb(80, 80, 80),
        }
    }

    /// Omarchy `themes/osaka-jade/colors.toml`.
    pub const fn jade() -> Self {
        Self {
            id: "jade",
            label: "Osaka Jade",
            blurb: "Omarchy — #111c18, jade #509475",
            aliases: &["osaka-jade", "osaka"],
            bg_base: rgb(17, 28, 24),
            bg_surface: rgb(26, 42, 34),
            bg_header: rgb(17, 28, 24),
            bg_popup: rgb(35, 55, 43),
            bg_highlight: rgb(35, 55, 43),
            border_subtle: rgb(83, 104, 91),
            border_focus: rgb(80, 148, 117),
            brand_gold: rgb(80, 148, 117),
            brand_orange: rgb(229, 199, 54),
            accent_green: rgb(84, 158, 106),
            accent_red: rgb(255, 83, 69),
            accent_yellow: rgb(229, 199, 54),
            text_primary: rgb(246, 245, 221),
            text_secondary: rgb(193, 196, 151),
            text_muted: rgb(83, 104, 91),
            text_dim: rgb(53, 72, 61),
        }
    }

    /// Omarchy `themes/ristretto/colors.toml`.
    pub const fn ristretto() -> Self {
        Self {
            id: "ristretto",
            label: "Ristretto",
            blurb: "Omarchy — espresso #2c2525, coral #f38d70",
            aliases: &[],
            bg_base: rgb(44, 37, 37),
            bg_surface: rgb(64, 62, 65),
            bg_header: rgb(44, 37, 37),
            bg_popup: rgb(64, 62, 65),
            bg_highlight: rgb(64, 62, 65),
            border_subtle: rgb(148, 138, 139),
            border_focus: rgb(243, 141, 112),
            brand_gold: rgb(243, 141, 112),
            brand_orange: rgb(249, 204, 108),
            accent_green: rgb(173, 218, 120),
            accent_red: rgb(253, 104, 131),
            accent_yellow: rgb(249, 204, 108),
            text_primary: rgb(230, 217, 219),
            text_secondary: rgb(195, 183, 184),
            text_muted: rgb(148, 138, 139),
            text_dim: rgb(114, 105, 106),
        }
    }

    /// Omarchy `themes/flexoki-light/colors.toml`.
    pub const fn flexoki() -> Self {
        Self {
            id: "flexoki",
            label: "Flexoki Light",
            blurb: "Omarchy — paper #fffcf0, ink #100f0f, blue #205ea6",
            aliases: &["flexoki-light"],
            bg_base: rgb(255, 252, 240),
            bg_surface: rgb(255, 255, 255),
            bg_header: rgb(242, 240, 229),
            bg_popup: rgb(255, 255, 255),
            bg_highlight: rgb(218, 216, 206),
            border_subtle: rgb(183, 181, 172),
            border_focus: rgb(32, 94, 166),
            brand_gold: rgb(32, 94, 166),
            brand_orange: rgb(218, 93, 151),
            accent_green: rgb(135, 154, 57),
            accent_red: rgb(209, 77, 65),
            accent_yellow: rgb(208, 162, 21),
            text_primary: rgb(16, 15, 15),
            text_secondary: rgb(64, 62, 56),
            text_muted: rgb(110, 107, 94),
            text_dim: rgb(183, 181, 172),
        }
    }

    /// Grok Build TUI — true black canvas, white type, violet accent.
    pub const fn grok() -> Self {
        Self {
            id: "grok",
            label: "Grok black",
            blurb: "xAI Grok — #050506 canvas, white type, violet accent",
            aliases: &["xai", "void", "black"],
            bg_base: rgb(5, 5, 6),
            bg_surface: rgb(18, 18, 20),
            bg_header: rgb(8, 8, 9),
            bg_popup: rgb(16, 16, 18),
            bg_highlight: rgb(28, 28, 34),
            border_subtle: rgb(42, 42, 50),
            border_focus: rgb(139, 124, 247),
            brand_gold: rgb(196, 181, 253),
            brand_orange: rgb(167, 139, 250),
            accent_green: rgb(94, 234, 212),
            accent_red: rgb(251, 113, 133),
            accent_yellow: rgb(251, 191, 36),
            text_primary: rgb(244, 244, 245),
            text_secondary: rgb(161, 161, 170),
            text_muted: rgb(113, 113, 122),
            text_dim: rgb(63, 63, 70),
        }
    }

    /// Dracula — widest editor/terminal coverage in 2026.
    pub const fn dracula() -> Self {
        Self {
            id: "dracula",
            label: "Dracula",
            blurb: "Dracula — #282a36, pink #ff79c6, cyan #8be9fd",
            aliases: &[],
            bg_base: rgb(40, 42, 54),
            bg_surface: rgb(68, 71, 90),
            bg_header: rgb(33, 34, 44),
            bg_popup: rgb(52, 55, 70),
            bg_highlight: rgb(68, 71, 90),
            border_subtle: rgb(98, 114, 164),
            border_focus: rgb(189, 147, 249),
            brand_gold: rgb(189, 147, 249),
            brand_orange: rgb(255, 184, 108),
            accent_green: rgb(80, 250, 123),
            accent_red: rgb(255, 85, 85),
            accent_yellow: rgb(241, 250, 140),
            text_primary: rgb(248, 248, 242),
            text_secondary: rgb(189, 147, 249),
            text_muted: rgb(98, 114, 164),
            text_dim: rgb(68, 71, 90),
        }
    }

    /// Atom One Dark / One Dark Pro — VS Code's default dark.
    pub const fn onedark() -> Self {
        Self {
            id: "onedark",
            label: "One Dark",
            blurb: "Atom One Dark — #282c34, blue #61afef",
            aliases: &["one-dark", "atom", "one"],
            bg_base: rgb(40, 44, 52),
            bg_surface: rgb(49, 54, 64),
            bg_header: rgb(33, 37, 43),
            bg_popup: rgb(49, 54, 64),
            bg_highlight: rgb(62, 68, 82),
            border_subtle: rgb(75, 82, 99),
            border_focus: rgb(97, 175, 239),
            brand_gold: rgb(97, 175, 239),
            brand_orange: rgb(209, 154, 102),
            accent_green: rgb(152, 195, 121),
            accent_red: rgb(224, 108, 117),
            accent_yellow: rgb(229, 192, 123),
            text_primary: rgb(171, 178, 191),
            text_secondary: rgb(171, 178, 191),
            text_muted: rgb(92, 99, 112),
            text_dim: rgb(75, 82, 99),
        }
    }

    /// IBM Carbon oxocarbon — true black, IBM blue.
    pub const fn carbon() -> Self {
        Self {
            id: "carbon",
            label: "Oxocarbon",
            blurb: "IBM Carbon — #161616, blue #78a9ff",
            aliases: &["oxocarbon", "ibm"],
            bg_base: rgb(22, 22, 22),
            bg_surface: rgb(38, 38, 38),
            bg_header: rgb(22, 22, 22),
            bg_popup: rgb(38, 38, 38),
            bg_highlight: rgb(57, 57, 57),
            border_subtle: rgb(82, 82, 82),
            border_focus: rgb(120, 169, 255),
            brand_gold: rgb(120, 169, 255),
            brand_orange: rgb(190, 149, 255),
            accent_green: rgb(66, 190, 101),
            accent_red: rgb(238, 83, 150),
            accent_yellow: rgb(190, 149, 255),
            text_primary: rgb(242, 244, 248),
            text_secondary: rgb(210, 210, 210),
            text_muted: rgb(111, 111, 111),
            text_dim: rgb(57, 57, 57),
        }
    }

    /// Solarized Dark — Ethan Schoonover, 2011, still a top-5 terminal.
    pub const fn solar() -> Self {
        Self {
            id: "solar",
            label: "Solarized",
            blurb: "Solarized Dark — #002b36, cyan #2aa198",
            aliases: &["solarized", "solarized-dark"],
            bg_base: rgb(0, 43, 54),
            bg_surface: rgb(7, 54, 66),
            bg_header: rgb(0, 43, 54),
            bg_popup: rgb(7, 54, 66),
            bg_highlight: rgb(7, 54, 66),
            border_subtle: rgb(88, 110, 117),
            border_focus: rgb(38, 139, 210),
            brand_gold: rgb(181, 137, 0),
            brand_orange: rgb(203, 75, 22),
            accent_green: rgb(133, 153, 0),
            accent_red: rgb(220, 50, 47),
            accent_yellow: rgb(181, 137, 0),
            text_primary: rgb(147, 161, 161),
            text_secondary: rgb(131, 148, 150),
            text_muted: rgb(88, 110, 117),
            text_dim: rgb(7, 54, 66),
        }
    }

    fn matches(&self, needle: &str) -> bool {
        let label = self.label.to_ascii_lowercase().replace([' ', '_'], "-");
        self.id == needle
            || label == needle
            || self.aliases.iter().any(|a| a.eq_ignore_ascii_case(needle))
    }
}

pub const CATALOG: [Palette; 24] = [
    Palette::gold(),
    Palette::hermes(),
    Palette::grok(),
    Palette::midnight(),
    Palette::ember(),
    Palette::mono(),
    Palette::daylight(),
    Palette::tokyo(),
    Palette::mocha(),
    Palette::latte(),
    Palette::gruvbox(),
    Palette::nord(),
    Palette::forest(),
    Palette::kanagawa(),
    Palette::dawn(),
    Palette::rose(),
    Palette::matte(),
    Palette::jade(),
    Palette::ristretto(),
    Palette::flexoki(),
    Palette::dracula(),
    Palette::onedark(),
    Palette::carbon(),
    Palette::solar(),
];

pub const THEME_COUNT: usize = CATALOG.len();

pub fn catalog() -> &'static [Palette] {
    &CATALOG
}

pub fn lookup(id: &str) -> Palette {
    let needle = id.trim().to_ascii_lowercase().replace(['_', ' '], "-");
    CATALOG
        .iter()
        .copied()
        .find(|p| p.matches(&needle))
        .unwrap_or_else(Palette::gold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_len_matches_const() {
        assert_eq!(catalog().len(), THEME_COUNT);
        assert_eq!(THEME_COUNT, 24);
    }

    #[test]
    fn lookup_aliases() {
        assert_eq!(lookup("tokyo-night").id, "tokyo");
        assert_eq!(lookup("catppuccin").id, "mocha");
        assert_eq!(lookup("everforest").id, "forest");
        assert_eq!(lookup("rose-pine").id, "dawn");
        assert_eq!(lookup("rose-pine-moon").id, "rose");
        assert_eq!(lookup("osaka-jade").id, "jade");
        assert_eq!(lookup("black").id, "grok");
        assert_eq!(lookup("oxocarbon").id, "carbon");
        assert_eq!(lookup("one-dark").id, "onedark");
        assert_eq!(lookup("solarized").id, "solar");
        assert_eq!(lookup("HERMES SITE").id, "hermes");
        assert_eq!(lookup("nope").id, "gold");
    }

    #[test]
    fn omarchy_ids_present() {
        let ids: Vec<_> = catalog().iter().map(|p| p.id).collect();
        for id in [
            "gold",
            "hermes",
            "grok",
            "tokyo",
            "dracula",
            "onedark",
            "carbon",
            "solar",
            "mocha",
            "gruvbox",
            "nord",
            "forest",
            "kanagawa",
            "dawn",
            "rose",
            "matte",
            "jade",
            "ristretto",
            "flexoki",
        ] {
            assert!(ids.contains(&id), "missing {id}");
        }
    }
}
