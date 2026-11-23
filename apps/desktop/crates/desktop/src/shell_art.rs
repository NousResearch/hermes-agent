//! Port of the redesign's portal shell artwork (app/redesign/artwork.ts and
//! portal-artwork.css) and its dark colour roles (tokens/color.css, raw.css).
//! One artwork is picked per launch, in ARTWORKS order after the previous
//! launch's; its geometry is Figma's, in frame units, scaled to cover the window.
use iced::{Color, Element, Length, widget::{Space, container, image, pin, responsive, stack, svg}};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Artwork { Ascend, Cathedral, Forest, Angel, Sword, Throne, Lattice, Wing, Karborn, Vanilla }

pub(super) const ARTWORKS: [Artwork; 10] = [Artwork::Ascend, Artwork::Cathedral, Artwork::Forest, Artwork::Angel, Artwork::Sword,
    Artwork::Throne, Artwork::Lattice, Artwork::Wing, Artwork::Karborn, Artwork::Vanilla];

const fn rgb(r: u8, g: u8, b: u8) -> Color { Color::from_rgb8(r, g, b) }
/// The artworks' grounds (--portal-artwork-ground-*), flattened.
const GROUND_VANILLA: Color = rgb(13, 13, 27);
const GROUND_WING: Color = rgb(13, 13, 36);
const GROUND_PHOTO: Color = rgb(13, 13, 22);
const GROUND_KARBORN: Color = rgb(0, 0, 29);
const GROUND_LIGHT: Color = rgb(242, 242, 242);

struct Placement { x: f32, y: f32, w: f32, h: f32, shift: f32, over: f32, ground: Color }

impl Artwork {
    pub fn id(self) -> &'static str {
        match self { Self::Ascend => "ascend", Self::Cathedral => "cathedral", Self::Forest => "forest", Self::Angel => "angel", Self::Sword => "sword",
            Self::Throne => "throne", Self::Lattice => "lattice", Self::Wing => "wing", Self::Karborn => "karborn", Self::Vanilla => "vanilla" }
    }
    fn placement(self) -> Placement {
        let square = |y: f32| Placement { x: -5.0, y, w: 1280.0, h: 1280.0, shift: 0.0, over: 0.0, ground: GROUND_PHOTO };
        match self {
            Self::Vanilla => Placement { x: 78.25, y: -121.76, w: 1113.51, h: 1113.51, shift: 0.0, over: 0.902, ground: GROUND_VANILLA },
            Self::Wing => Placement { x: -169.09, y: -307.08, w: 1548.18, h: 1548.18, shift: 0.0, over: 0.0, ground: GROUND_WING },
            Self::Angel => Placement { x: -226.77, y: -60.53, w: 1557.1, h: 990.63, shift: 166.43, over: 0.30, ground: GROUND_PHOTO },
            Self::Ascend => Placement { x: 10.21, y: -60.02, w: 1296.33, h: 990.11, shift: 151.73, over: 0.0, ground: GROUND_PHOTO },
            Self::Karborn => Placement { x: 45.02, y: -6.0, w: 1179.95, h: 882.0, shift: 0.0, over: 0.0, ground: GROUND_KARBORN },
            Self::Throne => square(-165.0),
            Self::Cathedral | Self::Forest | Self::Sword | Self::Lattice => square(-205.0),
        }
    }
    pub fn ground(self) -> Color { self.placement().ground }
    fn image(self) -> Option<image::Handle> {
        let bytes: &'static [u8] = match self {
            Self::Ascend => include_bytes!("../assets/artwork/ascend.jpg"), Self::Cathedral => include_bytes!("../assets/artwork/cathedral.jpg"),
            Self::Forest => include_bytes!("../assets/artwork/forest.jpg"), Self::Angel => include_bytes!("../assets/artwork/angel.jpg"),
            Self::Sword => include_bytes!("../assets/artwork/sword.jpg"), Self::Throne => include_bytes!("../assets/artwork/throne.jpg"),
            Self::Lattice => include_bytes!("../assets/artwork/lattice.jpg"), Self::Karborn => include_bytes!("../assets/artwork/karborn.jpg"),
            Self::Wing | Self::Vanilla => return None,
        };
        Some(image::Handle::from_bytes(bytes))
    }
    fn vector(self) -> Option<svg::Handle> {
        match self {
            Self::Wing => Some(svg::Handle::from_memory(include_bytes!("../assets/artwork/wing.svg").as_slice())),
            Self::Vanilla => Some(svg::Handle::from_memory(include_bytes!("../assets/artwork/wing-ghost.svg").as_slice())),
            _ => None,
        }
    }
}

/// artworkForMode: light mode draws only Vanilla and Ascend (artwork.ts MODE_ARTWORKS).
pub(super) fn for_mode(artwork: Artwork, dark: bool) -> Artwork {
    if dark || matches!(artwork, Artwork::Vanilla | Artwork::Ascend) { artwork } else { Artwork::Vanilla }
}

/// artworkForLaunch: the artwork after the previous launch's, remembered in the library.
pub(super) fn for_launch(root: &Path) -> Artwork {
    let path = root.join("artwork.json");
    let previous = std::fs::read_to_string(&path).ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| v["artwork"].as_str().map(str::to_owned))
        .and_then(|id| ARTWORKS.iter().position(|a| a.id() == id));
    let next = ARTWORKS[previous.map_or(0, |i| (i + 1) % ARTWORKS.len())];
    let _ = std::fs::write(&path, serde_json::json!({ "artwork": next.id() }).to_string());
    next
}

/// The window-filling art layer: the ground, the picture where Figma places
/// it (scaled to cover, shifted beside a docked sidebar), and its tint.
pub(super) fn layer<'a, M: 'a>(artwork: Artwork, dark: bool) -> Element<'a, M> {
    responsive(move |size| {
        let mut place = artwork.placement();
        // Light paints #f2f2f2 under its artwork, the wing ghost veiled at .951.
        if !dark { place.ground = GROUND_LIGHT; place.over = if artwork == Artwork::Vanilla { 0.951 } else { 0.85 }; }
        let unit = (size.width / 1280.0).max(size.height / 880.0);
        let docked = if size.width >= 1024.0 { 1.0 } else { 0.0 };
        let x = size.width / 2.0 + (place.x + place.shift * docked - 635.0) * unit;
        let y = size.height / 2.0 + (place.y - 435.0) * unit;
        let (w, h) = (Length::Fixed(place.w * unit), Length::Fixed(place.h * unit));
        let picture: Element<'a, M> = match (artwork.image(), artwork.vector()) {
            (Some(handle), _) => image(handle).width(w).height(h).content_fit(iced::ContentFit::Fill).into(),
            (None, Some(handle)) => svg(handle).width(w).height(h).into(),
            _ => Space::new().into(),
        };
        let ground = place.ground; let over = place.over;
        stack![
            container(Space::new().width(Length::Fill).height(Length::Fill)).style(move |_| container::Style { background: Some(ground.into()), ..Default::default() }),
            pin(picture).x(x).y(y),
            container(Space::new().width(Length::Fill).height(Length::Fill)).style(move |_| container::Style { background: (over > 0.0).then(|| Color { a: over, ..ground }.into()), ..Default::default() }),
        ].width(Length::Fill).height(Length::Fill).into()
    }).into()
}

/// Dark colour roles of the portal shell, flattened from tokens/color.css.
pub(super) mod roles {
    use iced::Color;
    /// --portal-surface-stage-veil-conversation: white .05 / black .85 / blue .60, ≈94% opaque.
    pub const STAGE_VEIL: Color = Color { r: 14.0 / 255.0, g: 14.0 / 255.0, b: 35.0 / 255.0, a: 0.943 };
    /// The stage veil for the mode: navy at .943 in dark, white .80 in light.
    pub fn stage_veil(dark: bool) -> Color { if dark { STAGE_VEIL } else { Color { r: 1.0, g: 1.0, b: 1.0, a: 0.80 } } }
    /// --portal-surface-stage, solid.
    pub const STAGE: Color = Color { r: 13.0 / 255.0, g: 13.0 / 255.0, b: 35.0 / 255.0, a: 1.0 };
    /// --portal-sidebar-background over the photo ground.
    pub const SIDEBAR: Color = Color { r: 6.0 / 255.0, g: 6.0 / 255.0, b: 31.0 / 255.0, a: 1.0 };
    /// --portal-composer-fill.
    pub const COMPOSER: Color = Color { r: 22.0 / 255.0, g: 22.0 / 255.0, b: 48.0 / 255.0, a: 1.0 };
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn launches_rotate_through_the_artworks() {
        let dir = tempfile::tempdir().unwrap();
        let first = for_launch(dir.path());
        let second = for_launch(dir.path());
        assert_eq!(first, Artwork::Ascend);
        assert_eq!(second, Artwork::Cathedral);
        for _ in 0..8 { for_launch(dir.path()); }
        assert_eq!(for_launch(dir.path()), Artwork::Ascend, "wraps after Vanilla");
    }
}

// ---------------------------------------------------------------- accents ("Make it yours")
/// store/portal-accent.ts PORTAL_ACCENTS, with raw.css's fill, ink-on-fill and brand inks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Accent { Blue, Violet, Pink, Red, Orange, Yellow, Green, Cyan }
pub(super) const ACCENTS: [Accent; 8] = [Accent::Blue, Accent::Violet, Accent::Pink, Accent::Red, Accent::Orange, Accent::Yellow, Accent::Green, Accent::Cyan];
const NAVY: Color = rgb(11, 11, 36);
impl Accent {
    pub fn id(self) -> &'static str { match self { Self::Blue => "blue", Self::Violet => "violet", Self::Pink => "pink", Self::Red => "red", Self::Orange => "orange", Self::Yellow => "yellow", Self::Green => "green", Self::Cyan => "cyan" } }
    pub fn label(self) -> &'static str { match self { Self::Blue => "Blue", Self::Violet => "Violet", Self::Pink => "Pink", Self::Red => "Red", Self::Orange => "Orange", Self::Yellow => "Yellow", Self::Green => "Green", Self::Cyan => "Cyan" } }
    /// --portal-raw-accent: brand fills (buttons, switches, selection).
    pub fn fill(self) -> Color { match self { Self::Blue => rgb(0, 0, 242), Self::Violet => rgb(0x6d, 0x28, 0xd9), Self::Pink => rgb(0xb0, 0x12, 0x4f), Self::Red => rgb(0xc4, 0x0d, 0x0d),
        Self::Orange => rgb(0xa3, 0x3f, 0x00), Self::Yellow => rgb(0xf2, 0xf2, 0x00), Self::Green => rgb(0x00, 0xf2, 0x79), Self::Cyan => rgb(0x00, 0xf2, 0xf2) } }
    /// --portal-raw-accent-on: the ink on a brand fill.
    pub fn on(self) -> Color { if matches!(self, Self::Yellow | Self::Green | Self::Cyan) { NAVY } else { Color::WHITE } }
    /// --portal-raw-accent-ink-{dark,light}: rings, underlines and brand text.
    pub fn ink(self, dark: bool) -> Color {
        if dark { match self { Self::Blue => rgb(0x85, 0x85, 0xff), Self::Violet => rgb(0xb7, 0x9c, 0xff), Self::Pink => rgb(0xff, 0x80, 0xb0), Self::Red => rgb(0xff, 0x80, 0x80),
            Self::Orange => rgb(0xff, 0x9a, 0x52), Self::Yellow => rgb(0xf2, 0xf2, 0x00), Self::Green => rgb(0x3f, 0xd6, 0x8a), Self::Cyan => rgb(0x00, 0xf2, 0xf2) } }
        else { match self { Self::Red => rgb(0xb0, 0x0c, 0x0c), Self::Yellow => rgb(0x66, 0x60, 0x00), Self::Green => rgb(0x08, 0x6b, 0x36), Self::Cyan => rgb(0x00, 0x69, 0x6b), other => other.fill() } }
    }
}
static ACCENT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub(super) fn accent() -> Accent { ACCENTS[ACCENT.load(std::sync::atomic::Ordering::Relaxed).min(ACCENTS.len() - 1)] }

/// The "Make it yours" choices, remembered in the library.
#[derive(Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub(super) struct Choices { pub accent: String, pub artwork: Option<String> }
fn choices_path(root: &Path) -> std::path::PathBuf { root.join("portal-preferences.json") }
pub(super) fn load_choices(root: &Path) -> Choices {
    let choices: Choices = std::fs::read(choices_path(root)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    ACCENT.store(ACCENTS.iter().position(|a| a.id() == choices.accent).unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
    choices
}
pub(super) fn chosen_artwork(choices: &Choices) -> Option<Artwork> { choices.artwork.as_deref().and_then(|id| ARTWORKS.iter().copied().find(|a| a.id() == id)) }
pub(super) fn choose(root: &Path, accent: Option<Accent>, artwork: Option<Artwork>) -> Result<(), String> {
    let mut choices = load_choices(root);
    if let Some(a) = accent { choices.accent = a.id().into(); ACCENT.store(ACCENTS.iter().position(|x| *x == a).unwrap_or(0), std::sync::atomic::Ordering::Relaxed); }
    if let Some(a) = artwork { choices.artwork = Some(a.id().into()); }
    std::fs::write(choices_path(root), serde_json::to_vec(&choices).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
/// artworksFor(mode): light offers only Vanilla and Ascend.
pub(super) fn artworks_for(dark: bool) -> Vec<Artwork> { ARTWORKS.into_iter().filter(|a| dark || matches!(a, Artwork::Vanilla | Artwork::Ascend)).collect() }
/// A small tile of the artwork for the chooser.
pub(super) fn thumbnail<'a, M: 'a>(artwork: Artwork, dark: bool) -> Element<'a, M> {
    let ground = if dark { artwork.ground() } else { GROUND_LIGHT };
    let picture: Element<'a, M> = match (artwork.image(), artwork.vector()) {
        (Some(handle), _) => image(handle).width(Length::Fill).height(Length::Fill).content_fit(iced::ContentFit::Cover).into(),
        (None, Some(handle)) => svg(handle).width(Length::Fill).height(Length::Fill).into(),
        _ => Space::new().into(),
    };
    container(picture).width(Length::Fill).height(Length::Fixed(72.0)).clip(true)
        .style(move |_| container::Style { background: Some(ground.into()), ..Default::default() }).into()
}
