//! Embedded native preview for HTML artifacts: a `wry` child webview
//! (WKWebView / WebView2 / WebKitGTK) laid over the Iced preview pane. The
//! shell stays Iced; only this pane is a webview.
//!
//! Guest rules follow AGENTS.md "Guest content never opens anything by itself":
//! the page may load only files inside its own folder; every other navigation
//! and every popup is refused. An http(s) link the page tries to open is
//! offered in the native preview header, and only a real click on that Iced
//! button opens the OS browser. There is no IPC handler, no devtools in release
//! builds, and the data store is non-persistent (incognito).
use iced::{Color, Rectangle, Task, advanced::widget::{self, Id, Operation, operation::Outcome}, window};
use std::{cell::RefCell, path::{Path, PathBuf}, sync::{Arc, Mutex}};

/// The container id of the preview pane the webview covers.
pub(super) const PANE: &str = "artifact-webview";

// ---------------------------------------------------------------- navigation policy
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Decision {
    /// A file inside the artifact's folder (or about:blank).
    Allow,
    /// An http(s) link: not loaded, offered to the user for the OS browser.
    External(String),
    /// Anything else: refused silently.
    Block,
}

/// Decide a navigation or popup request for a guest scoped to `root`.
pub(super) fn decide(url: &str, root: &Path) -> Decision {
    if matches!(url, "about:blank" | "about:srcdoc") { return Decision::Allow; }
    let Ok(parsed) = url::Url::parse(url) else { return Decision::Block };
    match parsed.scheme() {
        "file" => {
            let Ok(path) = parsed.to_file_path() else { return Decision::Block };
            // Resolve `..` and symlinks; a path that does not exist cannot be checked, so it is refused.
            let (Ok(path), Ok(root)) = (path.canonicalize(), root.canonicalize()) else { return Decision::Block };
            if path.starts_with(&root) { Decision::Allow } else { Decision::Block }
        }
        "http" | "https" if parsed.host_str().is_some() => Decision::External(parsed.to_string()),
        _ => Decision::Block,
    }
}

// ---------------------------------------------------------------- host
/// What the preview pane should show, and where.
#[derive(Default)]
pub(super) struct Host {
    pub window: Option<window::Id>,
    /// The file currently loaded, if the webview is up.
    shown: Option<PathBuf>,
    visible: bool,
    bounds: Option<Rectangle>,
    /// The last http(s) link the guest tried to open, for the header's button.
    pub offered: Arc<Mutex<Option<String>>>,
}

struct Live { webview: wry::WebView, file: PathBuf }
thread_local! {
    // The webview is created and driven only on the event-loop thread via window::run.
    static LIVE: RefCell<Option<Live>> = const { RefCell::new(None) };
}

/// `&dyn Window` → a sized handle for `build_as_child`.
struct Parent<'a>(&'a dyn window::Window);
impl raw_window_handle::HasWindowHandle for Parent<'_> {
    fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> { self.0.window_handle() }
}

fn rect(bounds: Rectangle, scale: f32) -> wry::Rect {
    wry::Rect {
        position: wry::dpi::LogicalPosition::new((bounds.x * scale) as f64, (bounds.y * scale) as f64).into(),
        size: wry::dpi::LogicalSize::new((bounds.width * scale).max(1.0) as f64, (bounds.height * scale).max(1.0) as f64).into(),
    }
}

struct FindPane(Option<Rectangle>);
impl Operation<Option<Rectangle>> for FindPane {
    fn traverse(&mut self, f: &mut dyn FnMut(&mut dyn Operation<Option<Rectangle>>)) { f(self) }
    fn container(&mut self, id: Option<&Id>, bounds: Rectangle) { if id == Some(&Id::new(PANE)) { self.0 = Some(bounds); } }
    fn finish(&self) -> Outcome<Option<Rectangle>> { Outcome::Some(self.0) }
}

impl Host {
    /// Bring the webview in line with what the UI shows. `file` is the HTML
    /// artifact to show (None hides it); `scale` maps Iced units to window points.
    pub fn sync<M: Send + 'static>(&mut self, file: Option<PathBuf>, background: Color, scale: f32, bounds_message: fn(Option<Rectangle>) -> M) -> Task<M> {
        #[cfg(debug_assertions)]
        if std::env::var_os("HERMES_RUST_WEBVIEW_TRACE").is_some() { eprintln!("webview sync: window={:?} file={:?} bounds={:?} visible={}", self.window, file, self.bounds, self.visible); }
        let Some(id) = self.window else { return Task::none() };
        match file {
            None => {
                if !self.visible { return Task::none(); }
                self.visible = false;
                window::run(id, |_| LIVE.with(|live| { if let Some(live) = live.borrow().as_ref() { let _ = live.webview.set_visible(false); } })).discard()
            }
            Some(file) => {
                let reload = self.shown.as_ref() != Some(&file);
                if reload { self.shown = Some(file.clone()); if let Ok(mut offered) = self.offered.lock() { *offered = None; } }
                let Some(bounds) = self.bounds else {
                    // Measure the pane first; place() runs when the bounds arrive.
                    return widget::operate(FindPane(None)).map(bounds_message);
                };
                let offered = self.offered.clone();
                let first = !self.visible || reload;
                self.visible = true;
                let measure = widget::operate(FindPane(None)).map(bounds_message);
                let place = window::run(id, move |window| place(window, &file, bounds, background, scale, offered, reload)).discard();
                if first { Task::batch([place, measure]) } else { place }
            }
        }
    }
    /// New pane bounds from the layout. Returns true when they changed.
    pub fn measured(&mut self, bounds: Option<Rectangle>) -> bool {
        let changed = self.bounds != bounds;
        self.bounds = bounds;
        changed
    }
    pub fn offered_link(&self) -> Option<String> { self.offered.lock().ok().and_then(|o| o.clone()) }
    pub fn clear_offer(&self) { if let Ok(mut offered) = self.offered.lock() { *offered = None; } }
}

fn place(window: &dyn window::Window, file: &Path, bounds: Rectangle, background: Color, scale: f32, offered: Arc<Mutex<Option<String>>>, reload: bool) {
    LIVE.with(|live| {
        let mut live = live.borrow_mut();
        let rebuild = reload || live.as_ref().is_none_or(|l| l.file != file);
        if rebuild {
            *live = None; // Drop the previous guest entirely: a new file gets a fresh, empty store.
            let root = file.parent().map(Path::to_path_buf).unwrap_or_default();
            let Ok(url) = url::Url::from_file_path(file) else { return };
            let bg = ((background.r * 255.0) as u8, (background.g * 255.0) as u8, (background.b * 255.0) as u8, 255);
            let (nav_root, nav_offer) = (root.clone(), offered.clone());
            let popup_root = root.clone();
            let builder = wry::WebViewBuilder::new()
                .with_url(url.as_str())
                .with_bounds(rect(bounds, scale))
                .with_background_color(bg)
                .with_incognito(true)
                .with_devtools(cfg!(debug_assertions))
                .with_clipboard(false)
                .with_navigation_handler(move |url| match decide(&url, &nav_root) {
                    Decision::Allow => true,
                    Decision::External(link) => { if let Ok(mut o) = nav_offer.lock() { *o = Some(link); } false }
                    Decision::Block => false,
                })
                .with_new_window_req_handler(move |url, _| {
                    if let Decision::External(link) = decide(&url, &popup_root) && let Ok(mut o) = offered.lock() { *o = Some(link); }
                    wry::NewWindowResponse::Deny
                });
            match builder.build_as_child(&Parent(window)) {
                Ok(webview) => { #[cfg(debug_assertions)] eprintln!("preview webview created for {}", file.display()); *live = Some(Live { webview, file: file.to_path_buf() }) }
                Err(error) => eprintln!("Could not create the preview webview: {error}"),
            }
        }
        if let Some(live) = live.as_ref() {
            let _ = live.webview.set_bounds(rect(bounds, scale));
            let _ = live.webview.set_visible(true);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn navigation_policy() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("site");
        std::fs::create_dir_all(root.join("pages")).unwrap();
        std::fs::write(root.join("index.html"), "<a href=x>").unwrap();
        std::fs::write(root.join("pages/two.html"), "").unwrap();
        std::fs::write(dir.path().join("secret.html"), "").unwrap();
        let file = |p: &Path| url::Url::from_file_path(p).unwrap().to_string();
        // Inside the artifact's folder: allowed, including a fragment.
        assert_eq!(decide(&file(&root.join("index.html")), &root), Decision::Allow);
        assert_eq!(decide(&(file(&root.join("pages/two.html")) + "#top"), &root), Decision::Allow);
        assert_eq!(decide("about:blank", &root), Decision::Allow);
        // Outside it, by sibling path or `..`: blocked.
        assert_eq!(decide(&file(&dir.path().join("secret.html")), &root), Decision::Block);
        assert_eq!(decide(&format!("{}/../secret.html", url::Url::from_file_path(&root).unwrap()), &root), Decision::Block);
        assert_eq!(decide("file:///etc/passwd", &root), Decision::Block);
        // Web links are offered, never loaded.
        assert_eq!(decide("https://example.com/a?b=1", &root), Decision::External("https://example.com/a?b=1".into()));
        assert_eq!(decide("http://localhost:3000/", &root), Decision::External("http://localhost:3000/".into()));
        // Other schemes are refused.
        for url in ["javascript:alert(1)", "data:text/html,<b>x</b>", "blob:https://x/1", "ftp://example.com/", "hermes://x", "mailto:a@b.c", "not a url"] {
            assert_eq!(decide(url, &root), Decision::Block, "{url}");
        }
    }
}
